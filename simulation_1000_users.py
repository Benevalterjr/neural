"""
===============================================================================
🐝 SIMULAÇÃO DE LARGA ESCALA: 1000 USUÁRIOS × FEED DE VÍDEOS CURTOS
===============================================================================
Simula o comportamento real de uma plataforma de vídeos curtos (estilo TikTok)
usando o ecossistema completo HiveStore + Neural Engine:

- 500 vídeos cadastrados em 10 clusters semânticos
- 1000 usuários com preferências de conteúdo distintas
- Sessões de scroll no feed "For You" com 15-25 vídeos cada
- Métricas de cache (HOT/WARM/COLD), latência, recomendação e engajamento
===============================================================================
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import time
import shutil
import sys
from collections import defaultdict, Counter

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from hivestore import (
    DiskHiveStore,
    HiveBrain,
    DiskVideoStorage,
    TieredCacheManager,
    VideoDeliveryBrain,
)

# ============================================================================
# CONFIGURAÇÃO DA SIMULAÇÃO
# ============================================================================
N_VIDEOS         = 500       # Total de vídeos no catálogo
N_CLUSTERS       = 10        # Categorias semânticas (humor, música, culinária, etc.)
EMBED_DIM        = 256       # Dimensão dos embeddings
N_USERS          = 1000      # Número de usuários simultâneos na simulação
VIDEOS_PER_SESSION = (15, 25) # Faixa de vídeos assistidos por sessão
HOT_CACHE_SIZE   = 50        # Limite de vídeos no Hot RAM Cache
SEED             = 42

CLUSTER_NAMES = [
    "🎵 Música", "😂 Humor", "🍳 Culinária", "💪 Fitness",
    "📚 Educação", "🎮 Gaming", "✈️ Viagens", "🐶 Pets",
    "💄 Beleza", "⚽ Esportes"
]

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================
def generate_clustered_embeddings(n_videos, n_clusters, dim, seed=42):
    """Gera embeddings com estrutura de cluster (centróides + ruído gaussiano)."""
    rng = np.random.RandomState(seed)
    
    # Centróides ortogonalizados para máxima separação
    centroids = rng.randn(n_clusters, dim).astype(np.float32)
    # Gram-Schmidt simplificado para separar melhor os clusters
    for i in range(1, n_clusters):
        for j in range(i):
            centroids[i] -= np.dot(centroids[i], centroids[j]) / (np.dot(centroids[j], centroids[j]) + 1e-8) * centroids[j]
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    
    embeddings = np.zeros((n_videos, dim), dtype=np.float32)
    labels = np.zeros(n_videos, dtype=np.int32)
    
    videos_per_cluster = n_videos // n_clusters
    for c in range(n_clusters):
        start = c * videos_per_cluster
        end = start + videos_per_cluster if c < n_clusters - 1 else n_videos
        count = end - start
        # Embedding = centróide + ruído gaussiano (sigma=0.3 para sobreposição parcial)
        noise = rng.randn(count, dim).astype(np.float32) * 0.3
        embeddings[start:end] = centroids[c] + noise
        labels[start:end] = c
        
    # Normalizar todos para a esfera unitária
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = (embeddings / norms).astype(np.float32)
    
    return embeddings, labels, centroids


def generate_user_preferences(n_users, n_clusters, seed=42):
    """Cada usuário tem 1-3 clusters de preferência (com pesos)."""
    rng = np.random.RandomState(seed + 1000)
    prefs = []
    for _ in range(n_users):
        n_interests = rng.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
        clusters = rng.choice(n_clusters, size=n_interests, replace=False)
        weights = rng.dirichlet(np.ones(n_interests) * 2.0)
        prefs.append(dict(zip(clusters.tolist(), weights.tolist())))
    return prefs


def simulate_watch_completion(user_pref, video_cluster):
    """
    Simula a taxa de conclusão de assistência baseada na relevância.
    - Se o vídeo pertence ao cluster preferido do usuário: alta conclusão (70-100%)
    - Se não: baixa conclusão (5-40%), simulando skip rápido
    """
    if video_cluster in user_pref:
        weight = user_pref[video_cluster]
        return np.clip(np.random.beta(5.0 * weight + 2, 2.0), 0.3, 1.0)
    else:
        return np.clip(np.random.beta(1.5, 5.0), 0.02, 0.4)


# ============================================================================
# MAIN: SIMULAÇÃO
# ============================================================================
if __name__ == "__main__":
    total_start = time.time()
    
    print("=" * 80)
    print("🐝 SIMULAÇÃO DE LARGA ESCALA: 1000 USUÁRIOS × FEED DE VÍDEOS CURTOS")
    print("=" * 80)
    
    # ---- FASE 1: Setup do Catálogo ----
    print(f"\n{'─'*60}")
    print(f"📦 FASE 1: Criando catálogo de {N_VIDEOS} vídeos em {N_CLUSTERS} clusters")
    print(f"{'─'*60}")
    
    # Limpar dados antigos
    for f in os.listdir('.'):
        if f.startswith("sim_hive_") or f.startswith("sim_video_"):
            try: os.remove(f)
            except: pass
    if os.path.exists("sim_video_storage"):
        shutil.rmtree("sim_video_storage", ignore_errors=True)
    
    # Gerar embeddings clusterizados
    embeddings, video_labels, centroids = generate_clustered_embeddings(N_VIDEOS, N_CLUSTERS, EMBED_DIM, SEED)
    
    print(f"   Distribuição de vídeos por cluster:")
    for c in range(N_CLUSTERS):
        count = np.sum(video_labels == c)
        print(f"     {CLUSTER_NAMES[c]:20s}: {count} vídeos")
    
    # Inicializar camadas do sistema
    store = DiskHiveStore("sim_hive_store", EMBED_DIM)
    brain = HiveBrain(store, max_cache_size=20000)
    video_storage = DiskVideoStorage(base_dir="sim_video_storage", hnerv_dim=64, thumb_dim=16)
    cache_mgr = TieredCacheManager(video_storage, max_hot_videos=HOT_CACHE_SIZE)
    delivery = VideoDeliveryBrain(brain, video_storage, cache_mgr)
    
    # Cadastrar todos os vídeos (silenciosamente)
    print(f"\n   Indexando {N_VIDEOS} vídeos no grafo HiveStore...")
    t_index_start = time.time()
    
    for vid in range(N_VIDEOS):
        av1_data = os.urandom(1024)  # 1KB mock AV1
        hnerv_w = np.random.normal(0, 0.1, 64).astype(np.float32)
        thumb = np.random.uniform(0, 1, (16, 16)).astype(np.float32)
        
        # Inserir vetor + dados físicos
        brain.insert_vector(embeddings[vid], vid, k_neighbors=12)
        video_storage.store_video(vid, av1_data, hnerv_w, thumb)
        
        if (vid + 1) % 100 == 0:
            brain.update_sentinels(k_sentinels=min(200, vid + 1))
            elapsed = time.time() - t_index_start
            print(f"     {vid+1}/{N_VIDEOS} indexados ({elapsed:.1f}s)")
    
    brain.update_sentinels(k_sentinels=200)
    t_index = time.time() - t_index_start
    print(f"   ✅ Catálogo indexado em {t_index:.2f}s ({N_VIDEOS/t_index:.0f} vídeos/s)")
    
    # ---- FASE 2: Geração de Usuários ----
    print(f"\n{'─'*60}")
    print(f"👥 FASE 2: Gerando perfis de {N_USERS} usuários")
    print(f"{'─'*60}")
    
    user_prefs = generate_user_preferences(N_USERS, N_CLUSTERS, SEED)
    
    interest_dist = Counter()
    for pref in user_prefs:
        for c in pref:
            interest_dist[c] += 1
    
    print(f"   Distribuição de interesses dos usuários:")
    for c in range(N_CLUSTERS):
        bar = "█" * (interest_dist[c] // 10)
        print(f"     {CLUSTER_NAMES[c]:20s}: {interest_dist[c]:4d} usuários  {bar}")
    
    # ---- FASE 3: Simulação das Sessões ----
    print(f"\n{'─'*60}")
    print(f"🎬 FASE 3: Simulando {N_USERS} sessões de feed \"For You\"")
    print(f"{'─'*60}")
    
    rng = np.random.RandomState(SEED + 2000)
    
    # Métricas coletadas
    all_search_times = []
    all_cache_hits = defaultdict(int)
    all_completion_rates = []
    relevant_recommendations = 0
    total_recommendations = 0
    videos_watched_global = Counter()
    user_satisfaction_scores = []
    prefetch_hits = 0
    total_prefetch_checks = 0
    cluster_recommendation_matrix = np.zeros((N_CLUSTERS, N_CLUSTERS), dtype=np.int32)
    
    session_start = time.time()
    
    for user_id in range(N_USERS):
        # Determinar quantos vídeos este usuário assiste
        n_watch = rng.randint(VIDEOS_PER_SESSION[0], VIDEOS_PER_SESSION[1] + 1)
        
        # Primeiro vídeo: escolher aleatoriamente de um cluster preferido
        pref = user_prefs[user_id]
        favorite_cluster = max(pref, key=pref.get)
        cluster_videos = np.where(video_labels == favorite_cluster)[0]
        current_video = rng.choice(cluster_videos)
        
        prev_video = None
        prev_completion = None
        user_completions = []
        user_relevant_recs = 0
        
        for step in range(n_watch):
            # ---- Assistir o vídeo atual ----
            videos_watched_global[current_video] += 1
            video_cluster = video_labels[current_video]
            
            # Simular comportamento de assistência
            completion = simulate_watch_completion(pref, video_cluster)
            all_completion_rates.append(completion)
            user_completions.append(completion)
            
            # Registrar watch no cache manager
            if prev_video is not None and prev_completion is not None:
                cache_mgr.record_watch_completion(prev_video, prev_completion)
            
            # ---- Carregar vídeo via Tiered Cache ----
            t_search_start = time.time()
            payload = cache_mgr.access_video(current_video)
            
            cache_source = payload["cache_hit"]
            all_cache_hits[cache_source] += 1
            
            # ---- Buscar recomendações via Waggle Dance ----
            emb = brain.get_vector(current_video)
            
            # Calcular feromônios (simplificado para performance)
            now = time.time()
            pheromones_dict = {}
            for vid_p, views in cache_mgr.watch_counts.items():
                last_time = cache_mgr.watch_timestamps.get(vid_p, now)
                dt = now - last_time
                completions = cache_mgr.watch_completions[vid_p]
                avg_c = np.mean(completions) if completions else 0.5
                quality_w = avg_c ** 3
                eff_views = views * quality_w
                dynamic_lambda = 0.00005 / np.log(np.e + eff_views)
                decayed = eff_views * np.exp(-dynamic_lambda * dt)
                pheromones_dict[vid_p] = float(np.log1p(decayed))
            
            neighbors = brain.find_neighbors(
                emb, k=4, beam_width=8, n_entry_points=3,
                pheromones=pheromones_dict, pheromone_weight=0.05
            )
            neighbors = [n for n in neighbors if n != current_video][:4]
            
            t_search = time.time() - t_search_start
            all_search_times.append(t_search)
            
            # ---- Avaliar qualidade da recomendação ----
            for rec_id in neighbors:
                rec_cluster = video_labels[rec_id]
                total_recommendations += 1
                cluster_recommendation_matrix[video_cluster, rec_cluster] += 1
                if rec_cluster in pref:
                    relevant_recommendations += 1
                    user_relevant_recs += 1
            
            # ---- Prefetch dos vizinhos ----
            cache_mgr.prefetch_videos(neighbors)
            
            # ---- Verificar se o PRÓXIMO vídeo escolhido era prefetched ----
            if step < n_watch - 1:
                total_prefetch_checks += 1
                # O usuário escolhe o próximo vídeo baseado na recomendação (70%) ou aleatório (30%)
                if rng.random() < 0.7 and len(neighbors) > 0:
                    # Segue a recomendação (com peso para primeiros)
                    weights = np.array([0.4, 0.3, 0.2, 0.1][:len(neighbors)])
                    weights /= weights.sum()
                    next_video = rng.choice(neighbors, p=weights)
                else:
                    # Escolhe aleatório de um cluster preferido
                    fav_c = rng.choice(list(pref.keys()))
                    c_vids = np.where(video_labels == fav_c)[0]
                    next_video = rng.choice(c_vids)
                
                if next_video in cache_mgr.hot_cache:
                    prefetch_hits += 1
            else:
                next_video = current_video  # placeholder, sessão acabou
            
            prev_video = current_video
            prev_completion = completion
            current_video = next_video
        
        # Satisfação do usuário = média das taxas de conclusão da sessão
        user_satisfaction_scores.append(np.mean(user_completions))
        
        # Progresso
        if (user_id + 1) % 200 == 0:
            elapsed = time.time() - session_start
            avg_latency = np.mean(all_search_times[-500:]) * 1000
            print(f"   👤 {user_id+1}/{N_USERS} usuários simulados "
                  f"({elapsed:.1f}s, latência média: {avg_latency:.1f}ms)")
    
    total_session_time = time.time() - session_start
    total_time = time.time() - total_start
    
    # ============================================================================
    # FASE 4: RELATÓRIO DE RESULTADOS
    # ============================================================================
    print(f"\n{'═'*80}")
    print(f"📊 RELATÓRIO FINAL DA SIMULAÇÃO")
    print(f"{'═'*80}")
    
    total_views = sum(videos_watched_global.values())
    search_times_ms = np.array(all_search_times) * 1000
    completions = np.array(all_completion_rates)
    satisfactions = np.array(user_satisfaction_scores)
    
    # ---- Escala ----
    print(f"\n📐 ESCALA DA SIMULAÇÃO")
    print(f"   {'Vídeos no catálogo:':<35s} {N_VIDEOS}")
    print(f"   {'Clusters semânticos:':<35s} {N_CLUSTERS}")
    print(f"   {'Usuários simulados:':<35s} {N_USERS}")
    print(f"   {'Total de visualizações:':<35s} {total_views:,}")
    print(f"   {'Média de vídeos/sessão:':<35s} {total_views/N_USERS:.1f}")
    print(f"   {'Tempo total de simulação:':<35s} {total_time:.2f}s")
    
    # ---- Performance ----
    print(f"\n⚡ PERFORMANCE DO MOTOR DE BUSCA")
    print(f"   {'Latência média (busca+cache+rec):':<35s} {np.mean(search_times_ms):.2f} ms")
    print(f"   {'Latência mediana:':<35s} {np.median(search_times_ms):.2f} ms")
    print(f"   {'Latência p95:':<35s} {np.percentile(search_times_ms, 95):.2f} ms")
    print(f"   {'Latência p99:':<35s} {np.percentile(search_times_ms, 99):.2f} ms")
    print(f"   {'QPS total (queries/segundo):':<35s} {total_views/total_session_time:.0f}")
    
    # ---- Cache ----
    print(f"\n💾 EFICIÊNCIA DO CACHE DE 3 CAMADAS")
    total_accesses = sum(all_cache_hits.values())
    for source in ["HOT_RAM", "COLD_DISK_WARM_MMAP"]:
        count = all_cache_hits.get(source, 0)
        pct = count / total_accesses * 100 if total_accesses > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"   {source:<30s}: {count:>6,} ({pct:5.1f}%) {bar}")
    
    prefetch_rate = prefetch_hits / total_prefetch_checks * 100 if total_prefetch_checks > 0 else 0
    print(f"\n   {'Taxa de acerto do prefetching:':<35s} {prefetch_rate:.1f}% ({prefetch_hits:,}/{total_prefetch_checks:,})")
    print(f"   {'Tamanho final do Hot Cache:':<35s} {len(cache_mgr.hot_cache)} vídeos")
    
    # ---- Recomendação ----
    print(f"\n🎯 QUALIDADE DA RECOMENDAÇÃO (WAGGLE DANCE + FEROMÔNIOS)")
    rec_precision = relevant_recommendations / total_recommendations * 100 if total_recommendations > 0 else 0
    print(f"   {'Recomendações totais geradas:':<35s} {total_recommendations:,}")
    print(f"   {'Recomendações relevantes:':<35s} {relevant_recommendations:,}")
    print(f"   {'Precisão de recomendação:':<35s} {rec_precision:.1f}%")
    
    # Precisão esperada por acaso (baseline)
    avg_interest_clusters = np.mean([len(p) for p in user_prefs])
    random_baseline = avg_interest_clusters / N_CLUSTERS * 100
    lift = rec_precision / random_baseline if random_baseline > 0 else 0
    print(f"   {'Baseline aleatório:':<35s} {random_baseline:.1f}%")
    print(f"   {'Lift sobre aleatório:':<35s} {lift:.1f}×")
    
    # ---- Engajamento ----
    print(f"\n📈 ENGAJAMENTO DOS USUÁRIOS")
    print(f"   {'Taxa média de conclusão (watch %):':<35s} {np.mean(completions)*100:.1f}%")
    print(f"   {'Mediana de conclusão:':<35s} {np.median(completions)*100:.1f}%")
    print(f"   {'Satisfação média do usuário:':<35s} {np.mean(satisfactions)*100:.1f}%")
    print(f"   {'Satisfação p10 (piores):':<35s} {np.percentile(satisfactions, 10)*100:.1f}%")
    print(f"   {'Satisfação p90 (melhores):':<35s} {np.percentile(satisfactions, 90)*100:.1f}%")
    
    # ---- Distribuição de visualizações (Long tail) ----
    print(f"\n📊 DISTRIBUIÇÃO DE VISUALIZAÇÕES (LONG TAIL)")
    view_counts = np.array([videos_watched_global.get(i, 0) for i in range(N_VIDEOS)])
    
    top10_ids = np.argsort(-view_counts)[:10]
    print(f"   Top 10 vídeos mais assistidos:")
    for rank, vid in enumerate(top10_ids):
        cluster = CLUSTER_NAMES[video_labels[vid]]
        views = view_counts[vid]
        bar = "█" * min(int(views / 2), 40)
        print(f"     #{rank+1:2d}  Vídeo {vid:3d} ({cluster:20s}): {views:4d} views  {bar}")
    
    watched_at_least_once = np.sum(view_counts > 0)
    never_watched = np.sum(view_counts == 0)
    print(f"\n   {'Vídeos assistidos ≥1 vez:':<35s} {watched_at_least_once}/{N_VIDEOS} ({watched_at_least_once/N_VIDEOS*100:.1f}%)")
    print(f"   {'Vídeos nunca assistidos:':<35s} {never_watched}/{N_VIDEOS}")
    print(f"   {'Gini de concentração:':<35s} ", end="")
    
    # Coeficiente de Gini
    sorted_views = np.sort(view_counts)
    n = len(sorted_views)
    cumulative = np.cumsum(sorted_views)
    gini = 1.0 - 2.0 * np.sum(cumulative) / (n * np.sum(sorted_views)) if np.sum(sorted_views) > 0 else 0
    print(f"{gini:.3f} (0=perfeita igualdade, 1=concentração total)")
    
    # ---- Matriz de Recomendação Inter-Cluster ----
    print(f"\n🔄 MATRIZ DE RECOMENDAÇÃO INTER-CLUSTER (origem → destino)")
    # Normalizar por linha
    row_sums = cluster_recommendation_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    norm_matrix = cluster_recommendation_matrix / row_sums * 100
    
    # Header
    header = "   " + " " * 22
    for c in range(N_CLUSTERS):
        header += f"C{c:d}  "
    print(header)
    
    for c in range(N_CLUSTERS):
        row = f"   {CLUSTER_NAMES[c]:20s}  "
        for c2 in range(N_CLUSTERS):
            val = norm_matrix[c, c2]
            if val >= 20:
                row += f"\033[92m{val:3.0f}%\033[0m "
            elif val >= 10:
                row += f"{val:3.0f}% "
            else:
                row += f"\033[90m{val:3.0f}%\033[0m "
        print(row)
    
    # ---- Efeito do Feromônio ----
    print(f"\n🧪 EFEITO DO SISTEMA DE FEROMÔNIOS")
    # Diagonal dominance = intra-cluster recommendations
    diag_sum = np.trace(cluster_recommendation_matrix)
    total_recs = cluster_recommendation_matrix.sum()
    intra_cluster_rate = diag_sum / total_recs * 100 if total_recs > 0 else 0
    print(f"   {'Recomendações intra-cluster:':<35s} {intra_cluster_rate:.1f}% (diagonal da matriz)")
    print(f"   {'Recomendações inter-cluster:':<35s} {100-intra_cluster_rate:.1f}% (serendipidade)")
    
    # Vídeos com mais feromônio (mais assistidos com alta conclusão)
    print(f"\n   Top 5 vídeos com maior feromônio acumulado:")
    now = time.time()
    pheromone_scores = {}
    for vid, views in cache_mgr.watch_counts.items():
        last_time = cache_mgr.watch_timestamps.get(vid, now)
        dt = now - last_time
        completions_list = cache_mgr.watch_completions[vid]
        avg_c = np.mean(completions_list) if completions_list else 0.5
        quality_w = avg_c ** 3
        eff_views = views * quality_w
        dynamic_lambda = 0.00005 / np.log(np.e + eff_views)
        decayed = eff_views * np.exp(-dynamic_lambda * dt)
        pheromone_scores[vid] = float(np.log1p(decayed))
    
    top_phero = sorted(pheromone_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    for rank, (vid, score) in enumerate(top_phero):
        cluster = CLUSTER_NAMES[video_labels[vid]]
        views = cache_mgr.watch_counts[vid]
        comps = cache_mgr.watch_completions[vid]
        avg_comp = np.mean(comps) if comps else 0
        print(f"     #{rank+1}  Vídeo {vid:3d} ({cluster:20s}): feromônio={score:.3f}  "
              f"views={views:3d}  conclusão_média={avg_comp*100:.0f}%")
    
    # ---- Resumo Final ----
    print(f"\n{'═'*80}")
    print(f"✅ SIMULAÇÃO CONCLUÍDA COM SUCESSO")
    print(f"{'═'*80}")
    print(f"   {N_USERS} usuários consumiram {total_views:,} vídeos em {total_time:.1f}s")
    print(f"   O Waggle Dance com feromônios atingiu {rec_precision:.1f}% de precisão ({lift:.1f}× sobre aleatório)")
    print(f"   Taxa de acerto do prefetching preditivo: {prefetch_rate:.1f}%")
    print(f"   Latência p99: {np.percentile(search_times_ms, 99):.2f}ms")
    print(f"{'═'*80}")
    
    # ---- Cleanup ----
    store.close()
    video_storage.close()
    for f in os.listdir('.'):
        if f.startswith("sim_hive_") or f.startswith("sim_video_"):
            try: os.remove(f)
            except: pass
    if os.path.exists("sim_video_storage"):
        shutil.rmtree("sim_video_storage", ignore_errors=True)
