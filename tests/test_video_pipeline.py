import os
import sys
import numpy as np
import time

# Adicionar o diretório raiz ao path para importar as classes do HiveStore
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hivestore import (
    DiskHiveStore,
    HiveBrain,
    StableSparseBNN,
    DiskVideoStorage,
    TieredCacheManager,
    VideoDeliveryBrain,
    HNeRVCodec
)

def cleanup_files():
    for f in os.listdir('.'):
        if f.startswith("video_meta_test"):
            try: os.remove(f)
            except: pass
    import shutil
    if os.path.exists("test_video_storage"):
        try: shutil.rmtree("test_video_storage")
        except: pass

def test_video_tiered_pipeline():
    print("=== INICIANDO PIPELINE DE TESTE DE VÍDEO CONCERN-SEPARATED ===")
    cleanup_files()
    
    # 1. Inicializar as camadas da arquitetura limpa
    print("\n[1] Inicializando camadas do sistema...")
    # HiveStore apenas para embeddings + metadados (D=256)
    metadata_store = DiskHiveStore("video_meta_test", 256)
    hive_brain = HiveBrain(metadata_store)
    
    # Storage otimizado para leitura sequencial (warm mmap + cold av1 files)
    video_storage = DiskVideoStorage(base_dir="test_video_storage", hnerv_dim=1024, thumb_dim=64)
    
    # Cache híbrido em camadas (Hot RAM cache limitado a 3 vídeos para demonstrar despejo LRU)
    cache_manager = TieredCacheManager(video_storage, max_hot_videos=3)
    
    # Coordenador de casos de uso
    delivery_brain = VideoDeliveryBrain(hive_brain, video_storage, cache_manager)
    
    # 2. Gerar Mock de 10 vídeos (Embeddings + AV1 + HNeRV Preview + Thumbnail)
    print("\n[2] Cadastrando 10 vídeos no sistema (Separation of Concerns)...")
    np.random.seed(42)
    
    # Vetores de embeddings de 256 dimensões com similaridade espacial projetada
    embeddings = np.random.randn(10, 256).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings.astype(np.float32)
    
    for i in range(10):
        # Simulando segmento completo AV1 (100KB de bytes)
        av1_data = os.urandom(100 * 1024)
        
        # Simulando Preview Neural HNeRV (1024 parâmetros float32)
        hnerv_preview = np.random.normal(0, 0.1, 1024).astype(np.float32)
        
        # Simulando Thumbnail Neural pequena (grade de 64x64 pixels)
        thumb_data = np.random.uniform(0, 1, (64, 64)).astype(np.float32)
        
        # Registrando no sistema
        delivery_brain.register_video(
            video_id=i,
            embedding=embeddings[i],
            av1_segment=av1_data,
            hnerv_weights=hnerv_preview,
            thumbnail=thumb_data
        )
        
    # Inicializa as sentinelas com os 10 vídeos cadastrados
    hive_brain.update_sentinels(k_sentinels=10)
        
    print("      -> 10 vídeos indexados no grafo HiveStore e gravados no cold/warm storage.")

    # 3. Simular primeira reprodução (Cold start)
    print("\n[3] Usuário reproduz o Vídeo ID 0 pela primeira vez (Cold Start)...")
    play1 = delivery_brain.play_video(video_id=0, num_prefetch=3)
    
    assert play1["cache_source"] == "COLD_DISK_WARM_MMAP", "Erro: O primeiro carregamento deveria vir do disco/mmap!"
    assert len(play1["prefetched_neighbors"]) >= 2, "Erro: Deveria recomendar e prefetchar pelo menos 2 vizinhos!"
    
    # 4. Verificar se os vizinhos recomendados foram prefetados em RAM
    prefetched_ids = play1["prefetched_neighbors"]
    print(f"      - Verificando se os vizinhos {prefetched_ids} estão no Hot RAM Cache:")
    for pid in prefetched_ids:
        in_cache = pid in cache_manager.hot_cache
        print(f"        * Vídeo ID {pid} no Hot Cache RAM? {in_cache}")
        assert in_cache, f"Erro: Vizinho {pid} deveria ter sido prefetado!"
        
    # 5. Reproduzir o mesmo vídeo novamente (Hot RAM Hit)
    print("\n[4] Usuário reproduz o Vídeo ID 0 novamente (Hot RAM Hit)...")
    play2 = delivery_brain.play_video(video_id=0, num_prefetch=3)
    assert play2["cache_source"] == "HOT_RAM", "Erro: Deveria ter carregado do Hot RAM Cache!"
    
    # 6. Testar reprodução de um vídeo prefetado (Carregamento instantâneo de thumbnail + preview)
    target_pref_id = prefetched_ids[0]
    print(f"\n[5] Usuário muda para o próximo vídeo '{target_pref_id}' (já prefetado)...")
    play_pref = delivery_brain.play_video(video_id=target_pref_id, num_prefetch=3)
    # Como foi prefetado, HNeRV e Thumbnail já estavam em RAM, e AV1 é carregado de forma sob demanda
    print(f"      - Vídeo prefetado '{target_pref_id}' carregado com sucesso via prefetch!")
    
    # 7. Testar Evicção LRU (Limitação do Hot Cache de 3 elementos)
    print("\n[6] Testando Política de Despejo LRU da RAM (limite de 3 vídeos)...")
    print(f"      - Vídeos atualmente na RAM: {list(cache_manager.hot_cache.keys())}")
    
    # Reproduzir vídeos 7, 8, 9 para forçar rotação de cache
    print("      - Forçando visualizações de outros vídeos para estourar o Hot Cache (IDs 7, 8, 9)...")
    delivery_brain.play_video(video_id=7, num_prefetch=1)
    delivery_brain.play_video(video_id=8, num_prefetch=1)
    delivery_brain.play_video(video_id=9, num_prefetch=1)
    
    # Verifica tamanho do cache
    current_cache_size = len(cache_manager.hot_cache)
    print(f"      - Tamanho atual do Hot Cache RAM: {current_cache_size} (limite máximo é 3)")
    assert current_cache_size == 3, f"Erro: O cache excedeu o limite máximo! Tamanho: {current_cache_size}"
    
    print(f"      - Vídeos finais na RAM: {list(cache_manager.hot_cache.keys())}")
    
    # 7. Testar Extrator Temporal SNN/RWKV no Cadastro
    print("\n[7] Testando Extrator Temporal Alternativo SNN/RWKV no Cadastro...")
    # Criar uma sequência de 15 quadros temporais, onde cada quadro possui embedding de 256-D
    T_frames = 15
    d_in = 256
    video_seq_frames = np.random.randn(T_frames, d_in).astype(np.float32)
    
    # Cadastrar o vídeo ID 99 usando a sequência de quadros como embedding (entrada 2D)
    delivery_brain.register_video(
        video_id=99,
        embedding=video_seq_frames,
        av1_segment=os.urandom(50 * 1024),
        hnerv_weights=np.random.normal(0, 0.1, 1024).astype(np.float32),
        thumbnail=np.random.uniform(0, 1, (64, 64)).astype(np.float32)
    )
    
    # Ler o embedding gerado na HiveStore para verificar se possui dimensão 256
    meta_99 = metadata_store.read_cell_meta(99)
    vec_99 = metadata_store.read_vector(meta_99["v_off"])
    
    print(f"      - Vídeo temporal ID 99 registrado.")
    print(f"      - Dimensão do embedding temporal extraído via SNN/RWKV: {vec_99.shape[0]} (esperado: 256)")
    
    assert vec_99.shape[0] == 256, f"Erro: Esperava dimensão 256 para o embedding temporal, obteve {vec_99.shape[0]}!"
    print("      - Extrator Temporal SNN/RWKV integrado e validado com sucesso!")
    
    # 8. Testar Codificador/Decodificador Neural HNeRV (Previews de Vídeo)
    print("\n[8] Testando Codificador/Decodificador Neural HNeRV (Previews de Vídeo)...")
    # Gerar um vídeo mock com 8 quadros onde um quadrado branco se move diagonalmente
    T_frames = 8
    real_frames = np.zeros((T_frames, 64, 64), dtype=np.float32)
    for t in range(T_frames):
        # Quadrado de 12x12 em movimento diagonal
        pos = t * 6
        real_frames[t, pos:pos+12, pos:pos+12] = 1.0
        
    # Inicializar HNeRV Codec
    hnerv_codec = HNeRVCodec(latent_dim=64, frame_dim=64)
    
    # Codificar o vídeo para obter a representação em pesos (Treinamento rápido)
    print("      - Treinando rede implícita HNeRV (codificando vídeo de 8 frames)...")
    t_start = time.time()
    weights = hnerv_codec.encode(real_frames, epochs=300, lr=0.08)
    duration = time.time() - t_start
    print(f"      - Vídeo codificado em pesos HNeRV em {duration*1000:.2f} ms.")
    print(f"      - Tamanho do vetor de pesos HNeRV gerado: {weights.shape[0]} floats (~{weights.shape[0]*4/1024:.2f} KB)")
    
    assert weights.shape[0] == 2656, f"Erro: Esperava vetor de pesos com 2656 floats (cerca de 10 KB), obteve {weights.shape[0]}!"
    
    # Decodificar o vídeo para reconstruir os frames originais
    print("      - Decodificando pesos neurais (reconstruindo frames)...")
    reconstructed_frames = hnerv_codec.decode(weights, T=T_frames)
    
    # Avaliar perda média quadrática de reconstrução (MSE)
    mse = np.mean((real_frames - reconstructed_frames) ** 2)
    print(f"      - Erro de Reconstrução Neural Médio (MSE): {mse:.6f}")
    
    assert mse < 0.05, f"Erro: O erro de reconstrução HNeRV é excessivo ({mse:.4f} > 0.05)!"
    print("      - Codec Neural HNeRV integrado e validado com sucesso com 100% de convergência!")
    
    # Fechar recursos
    video_storage.close()
    metadata_store.close()
    cleanup_files()
    
    print("\n=== PIPELINE DE VÍDEO CONCLUÍDO COM 100% DE SUCESSO! ===")

if __name__ == "__main__":
    test_video_tiered_pipeline()
