import os
import time
import numpy as np
from hivestore.infrastructure.mmap_store import DiskHiveStore
from hivestore.usecases.brain import HiveBrain

try:
    from hivestore.infrastructure.cython_ops import hive_ops
    USE_CYTHON = True
except ImportError:
    try:
        import hive_ops
        USE_CYTHON = True
    except ImportError:
        USE_CYTHON = False

def cleanup_files(pattern, db_dir="."):
    """Clean up temporary shard database files matching a pattern."""
    for f in os.listdir(db_dir):
        if f.startswith(pattern):
            try:
                os.remove(os.path.join(db_dir, f))
            except Exception:
                pass

class ShardNode:
    """Represents a physical shard node with simulated network latency and failover flags."""
    def __init__(self, shard_id, dimension, latency_ms=10, db_dir="."):
        self.shard_id = shard_id
        self.db_dir = db_dir
        self.db_name = f"hive_dist_shard_{shard_id}"
        cleanup_files(self.db_name, self.db_dir)
        
        db_path = os.path.join(self.db_dir, self.db_name)
        self.store = DiskHiveStore(db_path, dimension)
        self.brain = HiveBrain(self.store)
        
        self.latency_sec = latency_ms / 1000.0
        self.failed = False
        
        # Local-to-global indexing mappings
        self.global_to_local = {}
        self.local_to_global = {}
        self.local_vectors = []
        self.local_count = 0

    def simulate_network_delay(self):
        """Simulate network latency using sleep."""
        if self.latency_sec > 0:
            time.sleep(self.latency_sec)

    def insert(self, vec, video_id):
        """Insert a vector into this shard with network simulation and failover check."""
        self.simulate_network_delay()
        if self.failed:
            raise ConnectionError(f"Shard {self.shard_id} is offline.")
            
        local_id = self.local_count
        self.global_to_local[video_id] = local_id
        self.local_to_global[local_id] = video_id
        self.local_vectors.append((vec, video_id))
        self.local_count += 1
        
        self.brain.insert_vector(vec, local_id, k_neighbors=12)

    def search_local(self, query, k=3):
        """Perform a local vector search in this shard."""
        self.simulate_network_delay()
        if self.failed:
            raise ConnectionError(f"Shard {self.shard_id} is offline.")
            
        total = self.store.c_buf._tail // self.store.c_stride
        if total == 0:
            return []
        self.brain.update_sentinels(k_sentinels=max(1, min(50, total)))
        return self.brain.find_neighbors(query, k=k)

    def clear_data(self):
        """Clear local database files and reset metrics for rebalancing/split."""
        self.store.close()
        cleanup_files(self.db_name, self.db_dir)
        db_path = os.path.join(self.db_dir, self.db_name)
        self.store = DiskHiveStore(db_path, self.store.D)
        self.brain = HiveBrain(self.store)
        self.global_to_local.clear()
        self.local_to_global.clear()
        self.local_vectors.clear()
        self.local_count = 0

    def close(self):
        """Close connection and clean up files."""
        self.store.close()
        cleanup_files(self.db_name, self.db_dir)


class GatewayCoordinator:
    """Gateway Coordinator node that handles Voronoi spatial routing, dynamic splits, pruning, and failover."""
    def __init__(self, dimension, num_shards=5, latency_ms=10, balance_threshold=0.40, db_dir="."):
        self.D = dimension
        self.num_shards = num_shards
        self.latency_ms = latency_ms
        self.balance_threshold = balance_threshold
        self.db_dir = db_dir
        
        # Dynamic Shard allocation
        self.shards = [ShardNode(i, dimension, latency_ms, db_dir) for i in range(num_shards)]
        
        # Global Voronoi spatial centroids
        np.random.seed(42)
        centroids = np.random.randn(num_shards, dimension).astype(np.float32)
        self.centroids = [centroids[i] / np.linalg.norm(centroids[i]) for i in range(num_shards)]
        
        self.total_indexed = 0

    def route_write(self, vec, video_id, trigger_rebalance=True):
        """Route write queries to the spatially nearest shard based on Voronoi centroids."""
        sims = [np.dot(c, vec) for c in self.centroids]
        target_shard_id = int(np.argmax(sims))
        
        # Insert with fallback failover routing
        try:
            self.shards[target_shard_id].insert(vec, video_id)
        except ConnectionError:
            # Fallback to the second closest shard if primary is offline
            sorted_shards = np.argsort(-sims)
            fallback_shard_id = int(sorted_shards[1])
            self.shards[fallback_shard_id].insert(vec, video_id)
            target_shard_id = fallback_shard_id
            
        self.total_indexed += 1
        
        # Monitor for dynamic rebalancing (only after a minimal warm-up period)
        if trigger_rebalance and self.total_indexed > 100:
            self.check_and_trigger_split()
            
        return target_shard_id

    def check_and_trigger_split(self):
        """Check if any shard load exceeds the balance threshold."""
        for shard_id, shard in enumerate(self.shards):
            load = shard.local_count / self.total_indexed
            if load > self.balance_threshold:
                print(f"\n[Alerta de Sobrecarga] Shard {shard_id} ultrapassou o limite com {load:.1%} da carga ({shard.local_count} vetores)!")
                self.split_shard(shard_id)
                break

    def split_shard(self, shard_id):
        """Split an overloaded shard's Voronoi cell into two separate nodes."""
        print(f"[Rebalanceamento Dinâmico] Iniciando Split do Shard {shard_id}...")
        
        # Create new shard
        new_shard_id = len(self.shards)
        new_shard = ShardNode(new_shard_id, self.D, self.latency_ms, self.db_dir)
        self.shards.append(new_shard)
        
        # Mathematical perturbation to split the centroid into two sub-regions
        overloaded_centroid = self.centroids[shard_id]
        perturbation = np.random.normal(0, 0.05, self.D).astype(np.float32)
        
        c1 = overloaded_centroid + perturbation
        c2 = overloaded_centroid - perturbation
        
        c1 /= np.linalg.norm(c1)
        c2 /= np.linalg.norm(c2)
        
        # Update centroids list
        self.centroids[shard_id] = c1
        self.centroids.append(c2)
        
        # Migrate vectors
        migrated_vectors = list(self.shards[shard_id].local_vectors)
        self.shards[shard_id].clear_data()
        
        print(f"      - Re-roteando {len(migrated_vectors)} vetores entre Shard {shard_id} e Shard {new_shard_id}...")
        
        for vec, video_id in migrated_vectors:
            sims = [np.dot(c, vec) for c in self.centroids]
            # Route vector between the two split target shards
            target = shard_id if sims[shard_id] > sims[new_shard_id] else new_shard_id
            self.shards[target].insert(vec, video_id)
            
        print(f"      - Split concluído: Shard {shard_id} ({self.shards[shard_id].local_count} itens), Shard {new_shard_id} ({self.shards[new_shard_id].local_count} itens).")

    def route_search(self, query_vec, top_shards_to_query=2, k=3):
        """Route vector search queries, querying only the nearest active shards (pruning)."""
        sims = [np.dot(c, query_vec) for c in self.centroids]
        sorted_shards = np.argsort(-np.array(sims))
        
        shard_candidates = []
        shards_hit = 0
        
        for shard_id in sorted_shards:
            if shards_hit >= top_shards_to_query:
                break
            shard = self.shards[shard_id]
            try:
                # Query shard
                local_hits = shard.search_local(query_vec, k=k)
                shards_hit += 1
                for local_idx in local_hits:
                    global_idx = shard.local_to_global[local_idx]
                    v = shard.brain.get_vector(local_idx)
                    shard_candidates.append((global_idx, v, shard_id))
            except ConnectionError:
                print(f"[Failover] Shard {shard_id} caiu! Redirecionando requisição para próximo Shard...")
                continue
                
        if USE_CYTHON:
            aggregated_results = hive_ops.merge_results_cython(shard_candidates, query_vec, k)
        else:
            aggregated_results = []
            for global_idx, v, shard_id in shard_candidates:
                sim = np.dot(query_vec, v)
                aggregated_results.append((global_idx, sim, shard_id))
            aggregated_results.sort(key=lambda x: x[1], reverse=True)
            aggregated_results = aggregated_results[:k]
            
        return aggregated_results, shards_hit

    def close(self):
        """Clean up all allocated shards."""
        for shard in self.shards:
            shard.close()
