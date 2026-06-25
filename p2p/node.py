import os
import numpy as np
from hivestore.infrastructure.mmap_store import DiskHiveStore
from hivestore.usecases.brain import HiveBrain

class PeerNode:
    """
    Representa um nó descentralizado na rede P2P (PeerHive).
    Cada nó possui:
    - Capacidade de upload/download de rede.
    - Seu próprio banco de dados local (DiskHiveStore) representando seu fragmento do grafo.
    - Um cache local de vídeos físicos (AV1 e pesos HNeRV).
    - Tabela local de feromônios para busca semântica local.
    """
    def __init__(self, node_id, download_kbps, upload_kbps, dimension, db_dir="sim_p2p_dbs"):
        self.node_id = node_id
        self.download_kbps = download_kbps
        self.upload_kbps = upload_kbps
        self.dimension = dimension
        self.db_dir = db_dir
        
        # Garantir diretório dos bancos
        os.makedirs(db_dir, exist_ok=True)
        self.db_name = f"peer_db_{node_id}"
        self.db_path = os.path.join(self.db_dir, self.db_name)
        
        # Limpar base antiga se houver
        self._cleanup_db_files()
        
        # Inicializar armazenamento vetorial local
        self.store = DiskHiveStore(self.db_path, dimension)
        self.brain = HiveBrain(self.store, max_cache_size=1000)
        
        # Vizinhos diretos no overlay P2P
        self.peers = []
        
        # Vídeos físicos hospedados por este nó (IDs globais)
        self.hosted_videos = set()
        
        # Tabelas de mapeamento de IDs globais para IDs locais do grafo
        self.global_to_local = {}
        self.local_to_global = {}
        self.local_count = 0
        
        # Feromônios locais recebidos/acumulados
        self.pheromones = {}

    def _cleanup_db_files(self):
        """Limpa arquivos antigos de banco de dados deste nó."""
        if os.path.exists(self.db_dir):
            for f in os.listdir(self.db_dir):
                if f.startswith(self.db_name):
                    try:
                        os.remove(os.path.join(self.db_dir, f))
                    except Exception:
                        pass

    def connect_peer(self, other):
        """Adiciona conexão bidirecional no overlay P2P."""
        if other not in self.peers and other.node_id != self.node_id:
            self.peers.append(other)
        if self not in other.peers:
            other.peers.append(self)

    def host_video_payload(self, video_id):
        """Salva fisicamente o payload do vídeo neste nó (simulando cache/seeding)."""
        self.hosted_videos.add(video_id)

    def insert_index_vector(self, vec, global_id):
        """
        Insere o vetor de busca semântica do vídeo no grafo local do nó.
        Usa mapeamento global/local para manter consistência.
        """
        local_id = self.local_count
        self.global_to_local[global_id] = local_id
        self.local_to_global[local_id] = global_id
        self.local_count += 1
        
        self.brain.insert_vector(vec, local_id, k_neighbors=8)

    def search_local_knn(self, query_vec, k=3):
        """Executa a busca semântica local no seu subgrafo k-NN."""
        total = self.store.c_buf._tail // self.store.c_stride
        if total == 0:
            return []
            
        # Atualizar sentinelas locais
        self.brain.update_sentinels(k_sentinels=max(1, min(20, total)))
        
        # Montar dict de feromônios locais mapeados para IDs locais
        local_pheromones = {}
        for glob_id, phero_val in self.pheromones.items():
            if glob_id in self.global_to_local:
                local_pheromones[self.global_to_local[glob_id]] = phero_val
                
        # Buscar vizinhos
        local_hits = self.brain.find_neighbors(
            query_vec, 
            k=k, 
            beam_width=5, 
            n_entry_points=2, 
            pheromones=local_pheromones,
            pheromone_weight=0.08
        )
        
        # Converter resultados locais para globais
        global_results = []
        for loc_id in local_hits:
            if loc_id in self.local_to_global:
                glob_id = self.local_to_global[loc_id]
                vec = self.brain.get_vector(loc_id)
                global_results.append((glob_id, vec))
                
        return global_results

    def update_pheromone(self, video_id, value):
        """Atualiza a intensidade do feromônio local de um vídeo."""
        self.pheromones[video_id] = float(value)

    def close(self):
        """Fecha conexões e apaga bases locais."""
        self.store.close()
        self._cleanup_db_files()
