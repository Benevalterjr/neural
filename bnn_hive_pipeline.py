import os
# Configuração de thread único para otimização do NumPy no Windows
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import time
from tensorflow.keras.datasets import mnist

from hivestore import DiskHiveStore, HiveBrain, StableSparseBNN

def extract_bnn_embeddings(model, X):
    # Passa os dados pelo BNN até a camada latente de 256 neurônios
    a = X.T.astype(np.float32)
    for i in range(len(model.latent_weights)):
        Wb = model.get_binary_weights(i)
        z_raw = Wb.dot(a)
        if i < len(model.latent_weights) - 1:
            mean = z_raw.mean(axis=1, keepdims=True)
            std = z_raw.std(axis=1, keepdims=True) + 1e-6
            z = (z_raw - mean) / std
        else:
            z = z_raw
            
        if i == 0:
            a = z
        elif i < len(model.latent_weights) - 1:
            a = model.stable_sign(z)
        else:
            a = z
            
        # O output do layer i = 2 é a ativação binária de 256 dimensões
        if i == 2:
            return a.T

if __name__ == "__main__":
    print("=== INICIANDO PIPELINE DE INTEGRAÇÃO BNN + HIVESTORE ===")
    
    # 1. Carregar dados do MNIST
    print("\n[1/5] Carregando e preparando dataset MNIST...")
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.reshape(-1, 784).astype(np.float32) / 255.0 - 0.5) * 2
    X_test  = (X_test.reshape(-1, 784).astype(np.float32) / 255.0 - 0.5) * 2
    
    # Rótulos originais para checagem do RAG/KNN
    y_train_labels = y_train.copy()
    y_test_labels = y_test.copy()
    
    # One-hot encoding para o BNN
    y_train = np.eye(10)[y_train]
    y_test  = np.eye(10)[y_test]

    # 2. Treinar o extrator de características BNN
    print("\n[2/5] Treinando extrator de características BNN ([784, 512, 512, 256, 10])...")
    bnn_model = StableSparseBNN([784, 512, 512, 256, 10], sparsity=0.15)
    
    # Com Adam, BN completo e surrogate gradient, lr = 0.005 é ideal
    lr = 0.005
    for epoch in range(12):
        current_lr = lr * (0.5 ** (epoch // 4))
        for i in range(0, len(X_train), 256):
            bnn_model.train_step(X_train[i:i+256], y_train[i:i+256], lr=current_lr)
        test_acc = bnn_model.evaluate(X_test, y_test)
        print(f"Época {epoch:2d} | Acurácia Direta do BNN: {test_acc:.2%}")

    # 3. Extrair Embeddings Binários do BNN
    # Vamos indexar 50.000 imagens de treino no HiveStore
    N_index = 50000
    print(f"\n[3/5] Extraindo embeddings de {N_index} imagens de treino...")
    train_embeddings = extract_bnn_embeddings(bnn_model, X_train[:N_index])
    
    # Garantir que são vetores binários (sinais -1 ou 1)
    assert train_embeddings.shape == (N_index, 256), "Formato incorreto de embeddings!"
    print("Normalizando embeddings binários...")
    train_embeddings /= np.linalg.norm(train_embeddings, axis=1, keepdims=True)
    # Ensure they are float32
    train_embeddings = train_embeddings.astype(np.float32)

    # Limpeza de arquivos antigos
    print("Limpando arquivos antigos do indexador...")
    for f in os.listdir('.'):
        if f.startswith("hive_bnn"):
            try: os.remove(f)
            except: pass

    # 4. Criar e Indexar no DiskHiveStore de forma online incremental (O(N log N))
    print("\n[4/5] Indexando embeddings no DiskHiveStore de forma online incremental (O(N log N))...")
    store = DiskHiveStore("hive_bnn", 256)
    brain = HiveBrain(store)
    
    start_time = time.time()
    label_map = {}
    
    # Insere o primeiro elemento
    brain.insert_vector(train_embeddings[0], 0, k_neighbors=12)
    brain.update_sentinels(k_sentinels=500)
    label_map[0] = y_train_labels[0]
    
    for idx in range(1, N_index):
        brain.insert_vector(train_embeddings[idx], idx, k_neighbors=12)
        label_map[idx] = y_train_labels[idx]
        # Recalcular as sentinelas à medida que cresce para otimizar portões de busca online
        if idx % 5000 == 0:
            brain.update_sentinels(500)
            print(f"   Indexados {idx}/{N_index} elementos online...")
            
    print(f"Indexação de {N_index} elementos concluída em {time.time() - start_time:.2f}s.")

    # 5. Avaliação do Pipeline Integrado (BNN + HiveStore)
    N_test = 5000
    print(f"\n[5/5] Executando busca vetorial baseada em grafo em LOTE para classificar {N_test} imagens de teste...")
    test_embeddings = extract_bnn_embeddings(bnn_model, X_test[:N_test])
    test_embeddings /= np.linalg.norm(test_embeddings, axis=1, keepdims=True)
    test_embeddings = test_embeddings.astype(np.float32)
    
    brain = HiveBrain(store)
    np.random.seed(42)
    brain.update_sentinels(500) # Mais sentinelas para maior escala
    
    start_time = time.time()
    # Executar busca paralela via ThreadPoolExecutor no HiveBrain usando distância Hamming
    res_indices = brain.search_batch_hamming(test_embeddings, beam_width=10, n_entry_points=3)
    search_duration = time.time() - start_time
    
    hits = 0
    for idx, res in enumerate(res_indices):
        predicted_label = label_map[res]
        actual_label = y_test_labels[idx]
        if predicted_label == actual_label:
            hits += 1
            
    print(f"Acurácia de Classificação via Recuperação (BNN + HiveStore): {hits/N_test:.2%}")
    print(f"Tempo de busca: {search_duration:.4f}s (QPS de busca: {N_test/search_duration:.2f})")
    
    # 6. Teste de fogo da persistência
    print("\n--- TESTANDO PERSISTÊNCIA EM DISCO ---")
    store.close()
    del store
    print("Banco fechado com sucesso.")
    
    # Reabrir do disco
    store_reopened = DiskHiveStore("hive_bnn", 256)
    brain_reopened = HiveBrain(store_reopened)
    np.random.seed(42)
    brain_reopened.update_sentinels(500)
    
    start_time = time.time()
    res_indices_reopened = brain_reopened.search_batch_hamming(test_embeddings, beam_width=10, n_entry_points=3)
    search_duration_reopened = time.time() - start_time
    
    hits_reopened = 0
    for idx, res in enumerate(res_indices_reopened):
        predicted_label = label_map[res]
        actual_label = y_test_labels[idx]
        if predicted_label == actual_label:
            hits_reopened += 1
            
    print(f"Acurácia após reabertura do disco: {hits_reopened/N_test:.2%}")
    print(f"Tempo de busca: {search_duration_reopened:.4f}s (QPS de busca: {N_test/search_duration_reopened:.2f})")
    
    assert hits == hits_reopened, "ERRO: Resultados de recuperação divergem após persistência!"
    assert res_indices == res_indices_reopened, "ERRO: Índices exatos recuperados divergem após persistência!"
    print("Persistência do pipeline BNN + HiveStore validada com 100% de sucesso!")
    
    store_reopened.close()
    print("\n=== PIPELINE FINALIZADO COM SUCESSO ===")
