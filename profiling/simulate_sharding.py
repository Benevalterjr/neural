import os
import sys
import time
import numpy as np

# Adicionar o diretório raiz ao path para importar as classes do HiveStore
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hivestore import DiskHiveStore, HiveBrain

def cleanup_files(pattern):
    for f in os.listdir('.'):
        if f.startswith(pattern):
            try: os.remove(f)
            except: pass

class SimulatedShardNode:
    """Representa uma máquina física secundária (Shard Node) rodando uma partição do HiveStore."""
    def __init__(self, shard_id, dimension):
        self.shard_id = shard_id
        self.db_name = f"hive_shard_{shard_id}"
        cleanup_files(self.db_name)
        
        self.store = DiskHiveStore(self.db_name, dimension)
        self.brain = HiveBrain(self.store)
        
        # Mapeamento local-para-global para suportar IDs distribuídos
        self.global_to_local = {}
        self.local_to_global = {}
        self.local_count = 0

    def insert(self, vec, video_id):
        local_id = self.local_count
        self.global_to_local[video_id] = local_id
        self.local_to_global[local_id] = video_id
        self.local_count += 1
        
        # Cada shard insere localmente usando o ID sequencial local
        self.brain.insert_vector(vec, local_id, k_neighbors=12)

    def search_local(self, query, k=3):
        total = self.store.c_buf._tail // self.store.c_stride
        if total == 0:
            return []
        # Garante que sentinelas locais estejam sincronizadas com os dados locais do nó
        self.brain.update_sentinels(k_sentinels=max(1, min(50, total)))
        return self.brain.find_neighbors(query, k=k)

    def close(self):
        self.store.close()
        cleanup_files(self.db_name)


class SimulatedGatewayRouter:
    """Nó coordenador que gerencia o roteamento espacial de leituras e escritas para os Shards."""
    def __init__(self, dimension, num_shards=3):
        self.D = dimension
        self.num_shards = num_shards
        
        # Inicializa os Shards físicos simulados
        self.shards = [SimulatedShardNode(i, dimension) for i in range(num_shards)]
        
        # Define 3 centróides espaciais em RAM para simular a partição Voronoi global
        np.random.seed(42)
        centroids = np.random.randn(num_shards, dimension).astype(np.float32)
        self.centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

    def route_write(self, vec, video_id):
        # 1. Calcula a similaridade de cosseno com as 3 sentinelas de sharding
        sims = np.dot(self.centroids, vec)
        # 2. Escolhe o shard da sentinela com maior similaridade (célula Voronoi mais próxima)
        target_shard_id = np.argmax(sims)
        
        # 3. Roteia a escrita para o nó alvo
        self.shards[target_shard_id].insert(vec, video_id)
        return target_shard_id

    def route_search(self, query_vec, top_shards_to_query=2, k=3):
        # 1. Calcula similaridade com os centróides globais
        sims = np.dot(self.centroids, query_vec)
        # 2. Ordena e seleciona apenas os Shards mais próximos (poda de roteamento!)
        active_shard_ids = np.argsort(-sims)[:top_shards_to_query]
        
        # 3. Consulta apenas os Shards selecionados
        aggregated_results = []
        for shard_id in active_shard_ids:
            shard = self.shards[shard_id]
            local_hits = shard.search_local(query_vec, k=k)
            for local_idx in local_hits:
                # Converte o ID local de volta para o ID global mapeado
                global_idx = shard.local_to_global[local_idx]
                v = shard.brain.get_vector(local_idx)
                sim = np.dot(query_vec, v)
                aggregated_results.append((global_idx, sim, shard_id))
                
        # 4. Agrega e ordena para retornar as recomendações globais mais próximas
        aggregated_results.sort(key=lambda x: x[1], reverse=True)
        return aggregated_results[:k], active_shard_ids

    def close(self):
        for shard in self.shards:
            shard.close()


if __name__ == "__main__":
    print("=== INICIANDO SIMULAÇÃO DE SHARDING DO HIVESTORE ===")
    
    D = 128
    N = 1000
    
    # Inicializa o cluster simulado (Gateway + 3 Shards)
    gateway = SimulatedGatewayRouter(dimension=D, num_shards=3)
    
    # Gerar 1000 embeddings de vídeos sintéticos
    np.random.seed(123)
    X = np.random.randn(N, D).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    X = X.astype(np.float32)
    
    # 1. Roteamento de Ingestão
    print(f"\n[1/3] Roteando e indexando {N} vídeos nos 3 Shards (Particionamento Voronoi)...")
    distribution = [0, 0, 0]
    
    for i in range(N):
        shard_routed = gateway.route_write(X[i], video_id=i)
        distribution[shard_routed] += 1
        
    print("\nDistribuição física dos vídeos cadastrados:")
    for shard_id, count in enumerate(distribution):
        print(f"      - Shard {shard_id}: {count} vídeos indexados ({count/N:.1%})")
        
    # 2. Testar busca distribuída com poda de roteamento (Pruning)
    print("\n[2/3] Executando busca distribuída com poda (top_shards=2)...")
    
    test_query = X[42]  # Usamos o vídeo ID 42 como query
    results, active_shards = gateway.route_search(test_query, top_shards_to_query=2, k=4)
    
    print(f"      - Vídeos consultados nos Shards ativos: {list(active_shards)} (O Shard {list(set([0,1,2]) - set(active_shards))[0]} foi podado/ignorado!)")
    print(f"      - Resultados agregados mais próximos (Top 4):")
    for rank, (idx, sim, shard_id) in enumerate(results):
        print(f"        {rank+1}º. Vídeo ID {idx:<3} | Similaridade: {sim:.6f} | Localizado no Shard {shard_id}")
        
    # Validar se o próprio ID 42 foi encontrado com similaridade de ~1.0 no shard correto
    assert results[0][0] == 42, "Erro: A busca distribuída falhou ao recuperar o próprio vetor da query!"
    print("SUCCESS: O próprio vetor foi recuperado na primeira posição com similaridade perfeita!")

    # 3. Testar vazão e eficiência do particionamento
    print("\n[3/3] Executando 100 buscas distribuídas para validar o ganho de carga...")
    queries = np.random.randn(100, D).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    queries = queries.astype(np.float32)
    
    shards_queried_count = 0
    start_time = time.time()
    
    for q in queries:
        _, active_shards = gateway.route_search(q, top_shards_to_query=2, k=3)
        shards_queried_count += len(active_shards)
        
    duration = time.time() - start_time
    
    total_potential_calls = 100 * 3 # Se fizéssemos broadcast para os 3 nós
    actual_calls = shards_queried_count
    reduction = (1 - actual_calls / total_potential_calls) * 100
    
    print(f"      - 100 buscas distribuídas concluídas em {duration*1000:.2f} ms.")
    print(f"      - Total de consultas que seriam feitas (Broadcast): {total_potential_calls}")
    print(f"      - Consultas físicas de fato efetuadas nos shards : {actual_calls}")
    print(f"      - Redução da carga de consultas no cluster       : {reduction:.1f}%")
    
    # Fechar recursos
    gateway.close()
    print("\n=== SIMULAÇÃO DE SHARDING FINALIZADA COM SUCESSO! ===")
