import numpy as np
import os
import sys

# Ensure root directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hivestore import GatewayCoordinator

def test_sharding_suite():
    print("--- INICIANDO TESTE DE SHARDING E DISTRIBUIÇÃO ---")
    
    D = 64
    # Começa com 3 Shards, delay de 0ms para testes rápidos e threshold de split a 40%
    gateway = GatewayCoordinator(dimension=D, num_shards=3, latency_ms=0, balance_threshold=0.40)
    
    # 1. Validar Roteamento de Escrita Inicial
    print("\n[1/4] Testando roteamento e consistência inicial...")
    np.random.seed(123)
    for i in range(50):
        vec = np.random.randn(D).astype(np.float32)
        vec /= np.linalg.norm(vec)
        gateway.route_write(vec, video_id=i, trigger_rebalance=False)
        
    total_in_shards = sum(s.local_count for s in gateway.shards)
    assert total_in_shards == 50, f"Erro: Esperava 50 vetores nos shards, encontrou {total_in_shards}."
    print("      Roteamento e inserções iniciais bem-sucedidos.")

    # 2. Validar Split Dinâmico
    print("\n[2/4] Testando split de shard sob desbalanceamento...")
    # Ingerindo vetores idênticos/próximos direcionados para forçar split
    target_vec = gateway.centroids[0].copy()
    for i in range(50, 150):
        # Pequeno ruído para não serem idênticos
        noise = np.random.normal(0, 0.01, D).astype(np.float32)
        vec = target_vec + noise
        vec /= np.linalg.norm(vec)
        gateway.route_write(vec, video_id=i, trigger_rebalance=True)
        
    # Como inserimos muitos vetores no centroid 0, deve ter disparado um ou mais splits
    print(f"      Total de shards após splits: {len(gateway.shards)}")
    assert len(gateway.shards) > 3, "Erro: Esperava que novos shards fossem criados a partir dos splits."
    print("      Rebalanceamento dinâmico e split de célula Voronoi validados!")

    # 3. Validar Pruning (Poda)
    print("\n[3/4] Testando pruning na busca distribuída...")
    q = np.random.randn(D).astype(np.float32)
    q /= np.linalg.norm(q)
    
    # Buscar nos top 2 mais próximos de todos os disponíveis
    results, shards_hit = gateway.route_search(q, top_shards_to_query=2, k=3)
    assert shards_hit == 2, f"Erro: Esperava consultar 2 shards, consultou {shards_hit}."
    assert len(results) <= 3, "Erro: Resultados retornados maior do que k."
    print(f"      Pruning validado: consultou {shards_hit} de {len(gateway.shards)} shards.")

    # 4. Validar Failover
    print("\n[4/4] Testando resiliência e failover...")
    # Descobrir qual o shard mais próximo da nossa query e marcá-lo como offline
    sims = [np.dot(c, q) for c in gateway.centroids]
    closest_shard_id = int(np.argsort(-np.array(sims))[0])
    
    print(f"      Marcando Shard {closest_shard_id} (o mais próximo) como offline...")
    gateway.shards[closest_shard_id].failed = True
    
    # Executar busca. Ela deve pular o offline e consultar os próximos ativos sem falhar
    results_failover, hit_count = gateway.route_search(q, top_shards_to_query=2, k=3)
    assert hit_count == 2, "Erro: Falha na recuperação de nós ativos durante o failover."
    # Garantir que nenhum resultado veio do shard falho
    for idx, sim, shard_id in results_failover:
        assert shard_id != closest_shard_id, f"Erro: Recebeu resultado do shard offline {closest_shard_id}!"
        
    print("      Failover validado com sucesso!")
    
    # Limpeza
    gateway.close()
    print("\n--- TESTE DE SHARDING CONCLUÍDO COM SUCESSO ---")

if __name__ == "__main__":
    test_sharding_suite()
