import os
import sys
import time
import gc
import ctypes
from ctypes import wintypes
import threading
import numpy as np

# Adicionar o diretório raiz ao path para importar as classes do HiveStore
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hivestore import (
    DiskHiveStore,
    HiveBrain,
    DiskVideoStorage,
    TieredCacheManager,
    VideoDeliveryBrain
)

# Declarar tipos do Windows API para leitura de RAM física
class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]

try:
    psapi = ctypes.windll.psapi
    kernel32 = ctypes.windll.kernel32
    GetProcessMemoryInfo = psapi.GetProcessMemoryInfo
    GetCurrentProcess = kernel32.GetCurrentProcess
    GetCurrentProcess.restype = ctypes.c_void_p
    GetCurrentProcess.argtypes = []
    GetProcessMemoryInfo.restype = wintypes.BOOL
    GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
        wintypes.DWORD
    ]
except:
    GetProcessMemoryInfo = None

def get_physical_ram():
    if GetProcessMemoryInfo is None:
        return 0.0
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    if GetProcessMemoryInfo(GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return counters.WorkingSetSize / (1024 * 1024)
    return 0.0

def cleanup_files():
    for f in os.listdir('.'):
        if f.startswith("hive_load_test"):
            try: os.remove(f)
            except: pass
    import shutil
    if os.path.exists("load_test_video_storage"):
        try: shutil.rmtree("load_test_video_storage")
        except: pass

if __name__ == "__main__":
    print("=== INICIANDO TESTE DE CARGA DE VÍDEOS: KINETICS-400 PROFILE ===")
    cleanup_files()
    
    # 1. Parâmetros de Carga (Kinetics-400 Profile: clip de ~10s = 2.0MB AV1)
    N = 10000 # 10.000 vídeos para estressar
    D = 128
    avg_av1_size = 10 * 1024  # 10 KB
    
    # 2. Inicializar camadas do sistema
    print("\n[1/4] Inicializando infraestrutura de banco e caches...")
    meta_store = DiskHiveStore("hive_load_test", D)
    hive_brain = HiveBrain(meta_store)
    
    video_storage = DiskVideoStorage("load_test_video_storage", hnerv_dim=1024, thumb_dim=64)
    
    # Hot Cache RAM limitado a 500 vídeos mais assistidos (conforme PRD)
    cache_manager = TieredCacheManager(video_storage, max_hot_videos=500)
    
    delivery_brain = VideoDeliveryBrain(hive_brain, video_storage, cache_manager)
    
    # 3. Teste de Ingestão em Lote (Bulk Ingestion Load)
    print(f"\n[2/4] Simulando upload e ingestão em lote de {N:,} vídeos (Kinetics-400)...")
    ram_start = get_physical_ram()
    print(f"      RAM do processo antes da ingestão: {ram_start:.2f} MB")
    
    # Gerar dados sintéticos uma vez para acelerar o loop de ingestão
    np.random.seed(42)
    embeddings = np.random.randn(N, D).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings.astype(np.float32)
    
    hnerv_preview = np.random.normal(0, 0.1, 1024).astype(np.float32)
    thumb_data = np.random.uniform(0, 1, (64, 64)).astype(np.float32)
    
    # Simulamos um segmento AV1 de 2MB
    mock_av1_segment = b'\x00' * avg_av1_size
    
    start_time = time.time()
    for i in range(N):
        delivery_brain.register_video(
            video_id=i,
            embedding=embeddings[i],
            av1_segment=mock_av1_segment,
            hnerv_weights=hnerv_preview,
            thumbnail=thumb_data
        )
        if i > 0 and i % 5000 == 0:
            # Inicializa/atualiza as sentinelas de indexação aproximada à medida que cresce
            hive_brain.update_sentinels(k_sentinels=300)
            print(f"      - Ingeridos {i}/{N} vídeos...")
            
    # Inicialização final das sentinelas espaciais
    hive_brain.update_sentinels(k_sentinels=300)
    
    ingest_duration = time.time() - start_time
    ram_after_ingest = get_physical_ram()
    
    print(f"      - Ingestão de {N:,} vídeos concluída em {ingest_duration:.2f}s (Throughput: {N/ingest_duration:.2f} vídeos/s)")
    print(f"      - RAM do processo pós-ingestão: {ram_after_ingest:.2f} MB (Delta: {ram_after_ingest - ram_start:+.2f} MB)")
    
    # 4. Teste Concorrente de Reprodução e Prefetching sob Carga (8 Clientes Paralelos)
    print(f"\n[3/4] Iniciando simulação de carga concorrente (8 threads simulando usuários)...")
    num_threads = 8
    requests_per_thread = 150
    all_latencies = []
    latencies_lock = threading.Lock()
    stop_event = threading.Event()
    
    def user_session_worker():
        thread_latencies = []
        for _ in range(requests_per_thread):
            if stop_event.is_set():
                break
            # Escolhe um vídeo aleatório para assistir
            vid = np.random.randint(0, N)
            
            t0 = time.time()
            # play_video faz o prefetch preditivo de 4 vídeos recomendados pelo grafo automaticamente
            delivery_brain.play_video(video_id=vid, num_prefetch=4)
            thread_latencies.append((time.time() - t0) * 1000) # ms
            
            # Pequeno intervalo simulando o tempo que o usuário gasta assistindo/deslizando
            time.sleep(0.005)
            
        with latencies_lock:
            all_latencies.extend(thread_latencies)

    threads = [threading.Thread(target=user_session_worker) for _ in range(num_threads)]
    
    ram_before_load = get_physical_ram()
    start_load_time = time.time()
    
    for t in threads: t.start()
    for t in threads: t.join()
    
    load_duration = time.time() - start_load_time
    ram_after_load = get_physical_ram()
    
    total_queries = len(all_latencies)
    qps = total_queries / load_duration
    mean_lat = np.mean(all_latencies)
    p95_lat = np.percentile(all_latencies, 95)
    p99_lat = np.percentile(all_latencies, 99)
    
    # Calcular hit rates de cache
    total_cache_requests = len(cache_manager.watch_counts)
    hot_hits = sum(1 for v in cache_manager.watch_counts.keys() if cache_manager.watch_counts[v] > 1)
    
    # 5. Imprimir relatório final comparativo no console
    print("\n" + "="*80)
    print("RELATÓRIO DE TESTE DE CARGA DO HIVE VÍDEO PIPELINE")
    print("="*80)
    print(f"Tamanho do Dataset (Kinetics-400) : {N:,} vídeos")
    print(f"Segmento de Vídeo Médio (AV1)    : {avg_av1_size / 1024 / 1024:.2f} MB")
    print(f"Total de Dados Geridos (Cold)      : {N * avg_av1_size / 1024 / 1024 / 1024:.2f} GB")
    print(f"Tempo Total de Ingestão (Bulk)     : {ingest_duration:.2f}s")
    print(f"Vazão de Ingestão                  : {N/ingest_duration:.2f} vídeos/s")
    print("-"*80)
    print(f"Clientes Concorrentes Simulados   : {num_threads} threads")
    print(f"Vistas + Prefetches Efetuados     : {total_queries:,} interações")
    print(f"Vazão de Clipe (QPS de Play)      : {qps:.2f} plays/s")
    print(f"Latência Média de Play            : {mean_lat:.2f} ms")
    print(f"Latência p95 de Play              : {p95_lat:.2f} ms")
    print(f"Latência p99 de Play              : {p99_lat:.2f} ms")
    print("-"*80)
    print(f"Capacidade de Cache RAM (Limite)  : {cache_manager.max_hot_videos} vídeos hot")
    print(f"Itens Físicos na RAM de Cache     : {len(cache_manager.hot_cache)} vídeos carregados")
    print(f"Vistas Repetidas (Hot RAM Hit)    : {hot_hits} hits")
    print(f"RAM Consumida no Início           : {ram_start:.2f} MB")
    print(f"RAM Consumida no Pico (Carga)     : {ram_after_load:.2f} MB")
    print(f"Delta RAM Total                   : {ram_after_load - ram_start:+.2f} MB")
    print("="*80)
    
    # Fechar recursos
    video_storage.close()
    meta_store.close()
    cleanup_files()
    
    print("\n=== TESTE DE CARGA DE VÍDEO CONCLUÍDO COM SUCESSO! ===")
