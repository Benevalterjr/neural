import os
import sys
import time
import numpy as np

# Adicionar o diretório raiz ao path para importar as classes do HiveStore
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hivestore import DiskHiveStore, HiveBrain

def cleanup_files(base_path):
    for f in os.listdir('.'):
        if f.startswith(base_path):
            try: os.remove(f)
            except: pass

if __name__ == "__main__":
    print("=== TESTANDO LATÊNCIA DE PRIMEIRO BYTE (PRIMEIRA BUSCA APÓS ABERTURA) ===")
    
    D = 128
    base_name = "hive_first_byte_test"
    cleanup_files(base_name)
    
    # 1. Criar um banco persistente de teste com 10.000 vetores
    print("Criando banco persistente com 10.000 vetores...")
    X = np.random.randn(10000, D).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    X = X.astype(np.float32)
    
    store = DiskHiveStore(base_name, D)
    for i in range(10000):
        v_off = store.append_vector(X[i])
        # Grava metadados básicos simples
        store.write_cell_meta(i, v_off, 0, 0)
    store.close()
    
    # 2. Medir Latência de Primeiro Byte (Abertura física do banco + FPS de sentinelas + busca local)
    print("\nSimulando inicialização fria e primeira busca...")
    t0 = time.time()
    
    # Abre do disco
    store_cold = DiskHiveStore(base_name, D)
    brain_cold = HiveBrain(store_cold)
    
    # Executa a busca (a primeira busca inicializa as sentinelas em RAM via FPS automaticamente)
    q = np.random.randn(D).astype(np.float32)
    q /= np.linalg.norm(q)
    q = q.astype(np.float32)
    
    first_result = brain_cold.search(q, beam_width=10, n_entry_points=3)
    
    latency_ms = (time.time() - t0) * 1000
    
    print(f"-> Latência do Primeiro Byte / Primeira Busca: {latency_ms:.2f} ms")
    
    if latency_ms < 300:
        print("SUCCESS: Latencia de primeiro byte esta abaixo do limite de 300ms!")
    else:
        print("FAILURE: Latencia de primeiro byte acima do limite de 300ms.")
        
    store_cold.close()
    cleanup_files(base_name)
