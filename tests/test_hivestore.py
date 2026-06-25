import numpy as np
import os
import time
import sys
import threading

# Garante que o diretório raiz esteja no path para que possamos importar hivestore
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hivestore import DiskHiveStore, HiveBrain

def test_hivestore_suite():
    print("--- INICIANDO TESTE DE FOGO DO HIVESTORE ---")
    
    # 1. Limpeza de arquivos anteriores
    print("Limpando arquivos antigos...")
    for f in os.listdir('.'):
        if f.startswith("hive_test") or f.startswith("hive_fixed") or f.startswith("hive_rag"):
            try:
                os.remove(f)
            except:
                pass

    # 2. Teste de Estresse de Escrita e Redimensionamento
    N, D = 2000, 128
    print(f"\n[1/4] Indexando {N} vetores de dimensão {D} com limite inicial de 256KB (força resizes)...")
    
    X = np.random.randn(N, D).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    
    start_time = time.time()
    store = DiskHiveStore("hive_test", D)
    
    # Calculando Ground Truth para conectar os nós
    print("Calculando similaridades reais do grafo de vizinhos locais (k-NN)...")
    sim_matrix = np.dot(X, X.T)
    
    for i in range(N):
        v_off = store.append_vector(X[i])
        real_neighbors = np.argsort(-sim_matrix[i])[1:16]
        n_off = store.append_graph_edges(real_neighbors)
        store.write_cell_meta(i, v_off, n_off, len(real_neighbors))
        
    write_duration = time.time() - start_time
    print(f"Indexação concluída em {write_duration:.2f}s (QPS de escrita: {N/write_duration:.2f})")
    
    # 3. Teste de Busca (Com Sentinelas)
    print("\n[2/4] Executando buscas direcionadas via Sentinelas em RAM...")
    brain = HiveBrain(store)
    start_sentinels = time.time()
    brain.update_sentinels(200)
    print(f"Sentinelas construídas em {time.time() - start_sentinels:.4f}s.")
    
    test_queries = np.random.randn(100, D).astype(np.float32)
    test_queries /= np.linalg.norm(test_queries, axis=1, keepdims=True)
    
    start_time = time.time()
    for q in test_queries:
        res = brain.search(q, beam_width=10, n_entry_points=3)
    search_duration = time.time() - start_time
    print(f"100 buscas concluídas em {search_duration:.4f}s (QPS de busca: {100/search_duration:.2f})")
    
    # 4. Teste de Persistência
    print("\n[3/4] Testando persistência do banco em disco...")
    target_idx = 1234
    target_vec = store.read_vector(target_idx)
    target_meta = store.read_cell_meta(target_idx)
    target_neighs = store.read_neighbors(target_meta)
    
    store.close()
    print("Conexões fechadas com sucesso.")
    
    # Reabrir banco
    print("Reabrindo banco a partir do disco...")
    store_reopened = DiskHiveStore("hive_test", D)
    
    vec_reopened = store_reopened.read_vector(target_idx)
    meta_reopened = store_reopened.read_cell_meta(target_idx)
    neighs_reopened = store_reopened.read_neighbors(meta_reopened)
    
    assert np.allclose(target_vec, vec_reopened), "ERRO: Vetores divergem após persistência!"
    assert target_meta == meta_reopened, "ERRO: Metadados divergem após persistência!"
    assert np.all(target_neighs == neighs_reopened), "ERRO: Grafo de vizinhos diverge após persistência!"
    print("Persistência validada com 100% de integridade binária!")
    
    # 5. Busca no banco de dados persistido (com Sentinelas recriadas)
    print("\n[4/4] Testando busca direcionada no banco persistido (recuperação exata)...")
    brain_reopened = HiveBrain(store_reopened)
    brain_reopened.update_sentinels(200)
    
    hits = 0
    for idx in range(100):
        res = brain_reopened.search(X[idx], beam_width=10, n_entry_points=3)
        sim = np.dot(X[idx], store_reopened.read_vector(res))
        if res == idx or sim > 0.99:
            hits += 1
            
    print(f"Acurácia de recuperação exata após reabertura: {hits}%")
    store_reopened.close()
    
    # Limpeza de arquivos de teste
    for f in os.listdir('.'):
        if f.startswith("hive_test") or f.startswith("hive_fixed") or f.startswith("hive_rag"):
            try:
                os.remove(f)
            except:
                pass
                
    # 6. Teste de Concorrência Multithread Read-Write sob Carga
    print("\n[5/5] Iniciando Teste de Concorrência Estressada...")
    store_concurrent = DiskHiveStore("hive_concurrent", 64)
    # Inicializar com alguns dados
    for i in range(100):
        v = np.random.randn(64).astype(np.float32)
        v_off = store_concurrent.append_vector(v)
        store_concurrent.write_cell_meta(i, v_off, 0, 0)
        
    stop_event = threading.Event()
    errors = []

    def writer_thread():
        idx = 100
        while not stop_event.is_set():
            try:
                v = np.random.randn(64).astype(np.float32)
                v_off = store_concurrent.append_vector(v)
                store_concurrent.write_cell_meta(idx, v_off, 0, 0)
                idx += 1
                time.sleep(0.001)
            except Exception as e:
                errors.append(f"Erro no escritor: {e}")
                break

    def reader_thread(tid):
        while not stop_event.is_set():
            try:
                total = store_concurrent.c_buf._tail // store_concurrent.c_stride
                if total > 0:
                    idx = np.random.randint(0, total)
                    meta = store_concurrent.read_cell_meta(idx)
                    vec = store_concurrent.read_vector(meta["v_off"])
                    assert vec.shape == (64,), "Shape incorreto!"
            except Exception as e:
                errors.append(f"Erro no leitor {tid}: {e}")
                break

    threads = []
    w_t = threading.Thread(target=writer_thread)
    w_t.start()
    threads.append(w_t)

    for i in range(4):
        r_t = threading.Thread(target=reader_thread, args=(i,))
        r_t.start()
        threads.append(r_t)

    time.sleep(3)
    stop_event.set()

    for t in threads:
        t.join()

    store_concurrent.close()
    
    # Limpeza dos arquivos concorrentes
    for f in os.listdir('.'):
        if f.startswith("hive_concurrent"):
            try:
                os.remove(f)
            except:
                pass

    if errors:
        print(f"ERRO: Concorrência falhou com {len(errors)} erros:")
        for err in errors[:5]:
            print(err)
        assert False, "Erro crítico no controle de concorrência!"
    else:
        print("Concorrência validada com 100% de sucesso! Nenhum crash ou deadlock.")

    print("\n--- TESTE DE FOGO CONCLUÍDO COM SUCESSO ---")

if __name__ == "__main__":
    test_hivestore_suite()
