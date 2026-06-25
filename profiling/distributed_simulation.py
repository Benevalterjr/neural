import os
import sys
import time
import numpy as np

# Add parent directory to path to import hivestore package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hivestore import ShardNode, GatewayCoordinator

if __name__ == "__main__":
    print("=== INICIANDO SIMULAÇÃO EM LARGA ESCALA DE CLUSTER DISTRIBUÍDO ===")
    
    D = 128
    # Initialize with 5 shards, 0ms latency for ingestion, and 40% rebalancing threshold
    gateway = GatewayCoordinator(dimension=D, num_shards=5, latency_ms=0, balance_threshold=0.40)
    
    np.random.seed(999)
    
    # 1. Ingest 4,000 normal videos (balanced load)
    print("\n[1/5] Ingerindo 4.000 vídeos normais (Carga balanceada)...")
    t_start = time.time()
    for i in range(4000):
        v = np.random.randn(D).astype(np.float32)
        v /= np.linalg.norm(v)
        gateway.route_write(v, video_id=i)
        
    print("      Distribuição de carga inicial:")
    for s in gateway.shards:
        print(f"      - Shard {s.shard_id}: {s.local_count} vídeos")
        
    # 2. Test imbalance with 6,000 viral videos
    print("\n[2/5] Ingerindo 6.000 vídeos Semelhantes (Vídeo Viral direcionado ao Shard 1)...")
    centroid_1 = gateway.centroids[1]
    
    for i in range(4000, 10000):
        noise = np.random.normal(0, 0.08, D).astype(np.float32)
        viral_vec = centroid_1 + noise
        viral_vec /= np.linalg.norm(viral_vec)
        # Route writes with rebalancing/splits enabled
        gateway.route_write(viral_vec, video_id=i)
        
    print(f"Indexação de 10.000 vetores concluída em {time.time() - t_start:.2f}s.")
        
    # Show final workload after multiple dynamic Voronoi splits
    print("\nDistribuição final pós-Rebalanceamento Dinâmico (Múltiplos Splits):")
    for s in gateway.shards:
        print(f"      - Shard {s.shard_id}: {s.local_count} vídeos (Carga: {s.local_count / 10000:.1%})")
        
    # 3. Test pruning with larger cluster size
    print(f"\n[3/5] Testando eficiência de busca (Pruning) com {len(gateway.shards)} Shards...")
    query = np.random.randn(D).astype(np.float32)
    query /= np.linalg.norm(query)
    
    # Search top 2 shards out of the expanded active cluster
    results, shards_hit = gateway.route_search(query, top_shards_to_query=2, k=3)
    print(f"      - Busca distribuída avaliou apenas {shards_hit} Shards de {len(gateway.shards)} disponíveis.")
    print(f"      - Redução da carga de trabalho do cluster: {(1 - shards_hit / len(gateway.shards)):.1%}")

    # 4. Apply artificial network latency to shards for read operations
    print("\n[4/5] Aplicando latência de rede artificial (10ms) nos Shards para leitura...")
    for shard in gateway.shards:
        shard.latency_sec = 0.010  # 10ms network delay per hop
        
    t0 = time.time()
    # Execute routed search hitting 2 shards (resulting in 2 * 10ms = 20ms network latency overhead)
    gateway.route_search(query, top_shards_to_query=2, k=3)
    duration_ms = (time.time() - t0) * 1000
    print(f"      - Tempo total de busca distribuída (com delay de rede): {duration_ms:.2f} ms")
    
    # 5. Failover routing under crashed state
    print("\n[5/5] Testando resiliência e Failover (Derrubando Shard ativo)...")
    sims = [np.dot(c, query) for c in gateway.centroids]
    closest_shard_id = int(np.argmax(sims))
    
    print(f"      - Derrubando o Shard {closest_shard_id} (Marcado como OFFLINE)...")
    gateway.shards[closest_shard_id].failed = True
    
    # Execute query
    print("      - Executando busca com o Shard offline:")
    results_failover, hit_count = gateway.route_search(query, top_shards_to_query=2, k=3)
    
    print(f"      - Busca concluída consultando {hit_count} shards ativos.")
    print(f"      - Resultados do failover retornados com sucesso:")
    for rank, (idx, sim, shard_id) in enumerate(results_failover):
        print(f"        {rank+1}º. Vídeo ID {idx:<4} | Similaridade: {sim:.4f} | Shard {shard_id}")
        
    gateway.close()
    print("\n=== SIMULAÇÃO AVANÇADA DE 10K VÍDEOS FINALIZADA COM SUCESSO! ===")
