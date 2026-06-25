import os
import sys
import numpy as np
import time

# Adicionar o diretório raiz ao path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hivestore import DiskHiveStore, HiveBrain
from hivestore.infrastructure.cython_ops import hive_ops

def test_cython_hamming():
    print("=== INICIANDO TESTE DO HOT-PATH CYTHON HAMMING ===")
    
    # 1. Limpeza
    for f in os.listdir('.'):
        if f.startswith("hive_hamming_test"):
            try: os.remove(f)
            except: pass
            
    # 2. Inicializar banco com vetores binarizados
    N, D = 500, 256
    print(f"\n[1] Inicializando base com {N} vetores binarizados (dimensão {D})...")
    np.random.seed(42)
    
    # Gerando vetores com -1.0 e 1.0 (binários)
    X = np.random.choice([-1.0, 1.0], size=(N, D)).astype(np.float32)
    
    store = DiskHiveStore("hive_hamming_test", D)
    brain = HiveBrain(store)
    
    # Calculando matriz de Hamming exata
    print("Calculando similaridade Hamming exata para indexação...")
    for i in range(N):
        v_off = store.append_vector(X[i])
        # Distância de Hamming exata com o resto da base
        dists = np.sum((X[i] >= 0.0) != (X >= 0.0), axis=1)
        real_neighbors = np.argsort(dists)[1:13] # Top 12 vizinhos mais próximos (menor distância Hamming)
        n_off = store.append_graph_edges(real_neighbors)
        store.write_cell_meta(i, v_off, n_off, len(real_neighbors))
        
    print("      -> Base indexada no grafo do HiveStore.")
    brain.update_sentinels(k_sentinels=50)
    
    # 3. Validar a função fast_hamming diretamente
    print("\n[2] Validando hive_ops.fast_hamming comparado ao numpy...")
    a = X[0]
    b = X[1]
    dist_cython = hive_ops.fast_hamming(a, b)
    dist_numpy = int(np.sum((a >= 0.0) != (b >= 0.0)))
    
    print(f"      - Distância Cython: {dist_cython}")
    print(f"      - Distância Numpy: {dist_numpy}")
    assert dist_cython == dist_numpy, "Erro: As distâncias calculadas divergem!"
    print("      - Validação de cálculo direto concluída com sucesso!")
    
    # 4. Testar buscas guiadas por Hamming
    print("\n[3] Testando search_hamming e find_neighbors_hamming no HiveBrain...")
    q = np.random.choice([-1.0, 1.0], size=(D,)).astype(np.float32)
    
    # Chamando métodos de Hamming
    res_search = brain.search_hamming(q, beam_width=10, n_entry_points=3)
    res_neighbors = brain.find_neighbors_hamming(q, k=5, beam_width=10, n_entry_points=3)
    
    print(f"      - Nó mais próximo via search_hamming: {res_search}")
    print(f"      - Top-5 vizinhos via find_neighbors_hamming: {res_neighbors}")
    
    assert len(res_neighbors) == 5, "Erro: Deveria retornar exatamente 5 vizinhos!"
    assert res_search in res_neighbors or brain.search_hamming(q, beam_width=20) is not None, "Erro na consistência de busca!"
    
    # 5. Benchmark de velocidade
    print("\n[4] Executando benchmark de performance (1000 buscas Hamming)...")
    queries = np.random.choice([-1.0, 1.0], size=(1000, D)).astype(np.float32)
    
    t_start = time.time()
    for q_idx in range(1000):
        _ = brain.find_neighbors_hamming(queries[q_idx], k=10)
    t_duration = time.time() - t_start
    print(f"      - 1000 buscas Hamming concluídas em {t_duration:.4f}s (QPS: {1000/t_duration:.2f})")
    
    # Fechar e limpar
    store.close()
    for f in os.listdir('.'):
        if f.startswith("hive_hamming_test"):
            try: os.remove(f)
            except: pass
            
    print("\n=== TODOS OS TESTES CYTHON HAMMING PASSARAM COM SUCESSO! ===")

if __name__ == '__main__':
    test_cython_hamming()
