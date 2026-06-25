import numpy as np
import time

try:
    from hivestore.infrastructure.cython_ops import hive_ops
    USE_CYTHON = True
except ImportError:
    try:
        import hive_ops
        USE_CYTHON = True
    except ImportError:
        USE_CYTHON = False

class HiveBrain:
    def __init__(self, store, max_cache_size=10000):
        self.store = store
        self.sentinels_ids = []
        self.sentinels_matrix = None
        self.vector_cache = {}
        self.meta_cache = {}
        self.neighbor_cache = {}
        self.max_cache_size = max_cache_size

    def _check_cache_limits(self):
        if len(self.vector_cache) > self.max_cache_size:
            self.vector_cache.clear()
        if len(self.meta_cache) > self.max_cache_size:
            self.meta_cache.clear()
        if len(self.neighbor_cache) > self.max_cache_size:
            self.neighbor_cache.clear()

    def get_vector(self, idx):
        if idx in self.vector_cache:
            return self.vector_cache[idx]
        v = self.store.read_vector(idx)
        v = np.asarray(v, dtype=np.float32)
        self._check_cache_limits()
        self.vector_cache[idx] = v
        return v

    def get_cell_meta(self, idx):
        if idx in self.meta_cache:
            return self.meta_cache[idx]
        meta = self.store.read_cell_meta(idx)
        self._check_cache_limits()
        self.meta_cache[idx] = meta
        return meta

    def get_neighbors(self, idx):
        if idx in self.neighbor_cache:
            return self.neighbor_cache[idx]
        meta = self.get_cell_meta(idx)
        neighs = self.store.read_neighbors(meta)
        self._check_cache_limits()
        self.neighbor_cache[idx] = neighs
        return neighs

    def update_sentinels(self, k_sentinels=300):
        total = self.store.c_buf._tail // self.store.c_stride
        if total == 0: return
        
        # 1. Collect candidates for sentinels
        n_candidates = min(total, k_sentinels * 5)
        candidates = np.random.choice(total, n_candidates, replace=False)
        
        # 2. Retrieve candidate vectors
        cand_vectors = np.array([self.get_vector(c) for c in candidates], dtype=np.float32)
        
        # 3. FPS Algorithm
        selected_ids = [candidates[0]]
        selected_vecs = [cand_vectors[0]]
        
        max_sims = np.dot(cand_vectors, cand_vectors[0])
        max_sims[0] = 9999.0
        
        for _ in range(min(k_sentinels - 1, total - 1)):
            farthest_cand_idx = np.argmin(max_sims)
            farthest_idx = candidates[farthest_cand_idx]
            
            if farthest_idx in selected_ids:
                remaining_candidates = [i for i, c in enumerate(candidates) if c not in selected_ids]
                if not remaining_candidates: break
                farthest_cand_idx = np.random.choice(remaining_candidates)
                farthest_idx = candidates[farthest_cand_idx]
                
            selected_ids.append(farthest_idx)
            selected_vecs.append(cand_vectors[farthest_cand_idx])
            
            new_sims = np.dot(cand_vectors, cand_vectors[farthest_cand_idx])
            max_sims = np.maximum(max_sims, new_sims)
            max_sims[farthest_cand_idx] = 9999.0
            
        self.sentinels_ids = selected_ids
        self.sentinels_matrix = np.array(selected_vecs, dtype=np.float32)

    def search(self, query, beam_width=5, n_entry_points=3, pheromones=None, pheromone_weight=0.0):
        if self.sentinels_matrix is None or len(self.sentinels_ids) == 0:
            total_elements = self.store.c_buf._tail // self.store.c_stride
            self.update_sentinels(k_sentinels=max(1, min(100, total_elements)))
            if self.sentinels_matrix is None or len(self.sentinels_ids) == 0:
                raise ValueError("O banco de dados está vazio. Adicione vetores antes de realizar buscas.")
        
        q_norm = np.linalg.norm(query)
        q_vec = query / q_norm if q_norm > 0 else query
        q_vec = np.asarray(q_vec, dtype=np.float32)
        
        sims = np.dot(self.sentinels_matrix, q_vec)
        top_sentinels = np.argsort(-sims)[:n_entry_points]
        candidates = [self.sentinels_ids[idx] for idx in top_sentinels]
        
        if USE_CYTHON:
            return hive_ops.local_search_cython(
                q_vec,
                candidates,
                beam_width,
                20,
                self.neighbor_cache,
                self.vector_cache,
                self.get_neighbors,
                self.get_vector,
                pheromones,
                pheromone_weight
            )
            
        visited = set(candidates)
        best_node = candidates[0]
        best_sim = np.dot(q_vec, self.get_vector(best_node))
        best_score = best_sim
        p_val = 0.0
        if pheromones is not None and best_node in pheromones:
            p_val = float(pheromones[best_node])
        best_score += pheromone_weight * p_val

        for _ in range(20):
            next_candidates = []
            for curr in candidates:
                neighs = self.get_neighbors(curr)
                for n in neighs:
                    if n not in visited:
                        visited.add(n)
                        sim = np.dot(q_vec, self.get_vector(n))
                        p_val = 0.0
                        if pheromones is not None and n in pheromones:
                            p_val = float(pheromones[n])
                        score = sim + pheromone_weight * p_val
                        next_candidates.append((n, score))
                        if score > best_score:
                            best_score = score
                            best_node = n

            if not next_candidates: break
            next_candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = [n for n, s in next_candidates[:beam_width]]

        return best_node

    def search_batch(self, queries, beam_width=5, n_entry_points=3, num_workers=None, pheromones=None, pheromone_weight=0.0):
        from concurrent.futures import ThreadPoolExecutor
        
        if self.sentinels_matrix is None or len(self.sentinels_ids) == 0:
            total_elements = self.store.c_buf._tail // self.store.c_stride
            self.update_sentinels(k_sentinels=max(1, min(100, total_elements)))
            if self.sentinels_matrix is None or len(self.sentinels_ids) == 0:
                raise ValueError("O banco de dados está vazio. Adicione vetores antes de realizar buscas.")
        
        q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
        q_vecs = np.where(q_norms > 0, queries / q_norms, queries)
        q_vecs = np.asarray(q_vecs, dtype=np.float32)
        
        sims = np.dot(q_vecs, self.sentinels_matrix.T)
        top_sentinels_batch = np.argsort(-sims, axis=1)[:, :n_entry_points]
        
        def search_single(args):
            q_vec, top_sentinels_idxs = args
            candidates = [self.sentinels_ids[idx] for idx in top_sentinels_idxs]
            
            if USE_CYTHON:
                return hive_ops.local_search_cython(
                    q_vec,
                    candidates,
                    beam_width,
                    20,
                    self.neighbor_cache,
                    self.vector_cache,
                    self.get_neighbors,
                    self.get_vector,
                    pheromones,
                    pheromone_weight
                )
                
            visited = set(candidates)
            best_node = candidates[0]
            best_sim = np.dot(q_vec, self.get_vector(best_node))
            best_score = best_sim
            p_val = 0.0
            if pheromones is not None and best_node in pheromones:
                p_val = float(pheromones[best_node])
            best_score += pheromone_weight * p_val

            for _ in range(20):
                next_candidates = []
                for curr in candidates:
                    neighs = self.get_neighbors(curr)
                    for n in neighs:
                        if n not in visited:
                            visited.add(n)
                            sim = np.dot(q_vec, self.get_vector(n))
                            p_val = 0.0
                            if pheromones is not None and n in pheromones:
                                p_val = float(pheromones[n])
                            score = sim + pheromone_weight * p_val
                            next_candidates.append((n, score))
                            if score > best_score:
                                best_score = score
                                best_node = n

                if not next_candidates: break
                next_candidates.sort(key=lambda x: x[1], reverse=True)
                candidates = [n for n, s in next_candidates[:beam_width]]

            return best_node

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(search_single, zip(q_vecs, top_sentinels_batch)))
            
        return results

    def find_neighbors(self, query, k=12, beam_width=5, n_entry_points=3, pheromones=None, pheromone_weight=0.0):
        total = self.store.c_buf._tail // self.store.c_stride
        if total == 0: return []
        
        if self.sentinels_matrix is None or len(self.sentinels_ids) == 0:
            self.update_sentinels(k_sentinels=max(1, min(100, total)))
            if self.sentinels_matrix is None or len(self.sentinels_ids) == 0:
                return []
        
        q_norm = np.linalg.norm(query)
        q_vec = query / q_norm if q_norm > 0 else query
        q_vec = np.asarray(q_vec, dtype=np.float32)
        
        sims = np.dot(self.sentinels_matrix, q_vec)
        top_sentinels = np.argsort(-sims)[:min(len(self.sentinels_ids), n_entry_points)]
        
        candidates = [self.sentinels_ids[idx] for idx in top_sentinels]
        
        if USE_CYTHON:
            return hive_ops.find_neighbors_cython(
                q_vec,
                candidates,
                k,
                beam_width,
                20,
                self.neighbor_cache,
                self.vector_cache,
                self.get_neighbors,
                self.get_vector,
                pheromones,
                pheromone_weight
            )
            
        visited = {}
        for c in candidates:
            sim = np.dot(q_vec, self.get_vector(c))
            p_val = 0.0
            if pheromones is not None and c in pheromones:
                p_val = float(pheromones[c])
            visited[c] = sim + pheromone_weight * p_val
            
        for _ in range(20):
            next_candidates = []
            for curr in candidates:
                neighs = self.get_neighbors(curr)
                for n in neighs:
                    if n not in visited:
                        sim = np.dot(q_vec, self.get_vector(n))
                        p_val = 0.0
                        if pheromones is not None and n in pheromones:
                            p_val = float(pheromones[n])
                        score = sim + pheromone_weight * p_val
                        visited[n] = score
                        next_candidates.append((n, score))
            
            if not next_candidates: break
            next_candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = [n for n, s in next_candidates[:beam_width]]
            
        sorted_visited = sorted(visited.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, s_val in sorted_visited[:k]]

    def search_hamming(self, query, beam_width=5, n_entry_points=3, pheromones=None, pheromone_weight=0.0):
        if self.sentinels_matrix is None or len(self.sentinels_ids) == 0:
            total_elements = self.store.c_buf._tail // self.store.c_stride
            self.update_sentinels(k_sentinels=max(1, min(100, total_elements)))
            if self.sentinels_matrix is None or len(self.sentinels_ids) == 0:
                raise ValueError("O banco de dados está vazio. Adicione vetores antes de realizar buscas.")
        
        q_vec = np.asarray(query, dtype=np.float32)
        
        if USE_CYTHON:
            sims = np.array([-float(hive_ops.fast_hamming(q_vec, s)) for s in self.sentinels_matrix], dtype=np.float32)
        else:
            sims = np.array([-float(np.sum((q_vec >= 0.0) != (s >= 0.0))) for s in self.sentinels_matrix], dtype=np.float32)
            
        top_sentinels = np.argsort(-sims)[:n_entry_points]
        candidates = [self.sentinels_ids[idx] for idx in top_sentinels]
        
        if USE_CYTHON:
            return hive_ops.local_search_hamming_cython(
                q_vec,
                candidates,
                beam_width,
                20,
                self.neighbor_cache,
                self.vector_cache,
                self.get_neighbors,
                self.get_vector,
                pheromones,
                pheromone_weight
            )
            
        visited = set(candidates)
        best_node = candidates[0]
        best_score = -float(np.sum((q_vec >= 0.0) != (self.get_vector(best_node) >= 0.0)))
        p_val = 0.0
        if pheromones is not None and best_node in pheromones:
            p_val = float(pheromones[best_node])
        best_score += pheromone_weight * p_val

        for _ in range(20):
            next_candidates = []
            for curr in candidates:
                neighs = self.get_neighbors(curr)
                for n in neighs:
                    if n not in visited:
                        visited.add(n)
                        score = -float(np.sum((q_vec >= 0.0) != (self.get_vector(n) >= 0.0)))
                        p_val = 0.0
                        if pheromones is not None and n in pheromones:
                            p_val = float(pheromones[n])
                        score += pheromone_weight * p_val
                        next_candidates.append((n, score))
                        if score > best_score:
                            best_score = score
                            best_node = n

            if not next_candidates: break
            next_candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = [n for n, s in next_candidates[:beam_width]]

        return best_node

    def find_neighbors_hamming(self, query, k=12, beam_width=5, n_entry_points=3, pheromones=None, pheromone_weight=0.0):
        total = self.store.c_buf._tail // self.store.c_stride
        if total == 0: return []
        
        if self.sentinels_matrix is None or len(self.sentinels_ids) == 0:
            self.update_sentinels(k_sentinels=max(1, min(100, total)))
            if self.sentinels_matrix is None or len(self.sentinels_ids) == 0:
                return []
        
        q_vec = np.asarray(query, dtype=np.float32)
        
        if USE_CYTHON:
            sims = np.array([-float(hive_ops.fast_hamming(q_vec, s)) for s in self.sentinels_matrix], dtype=np.float32)
        else:
            sims = np.array([-float(np.sum((q_vec >= 0.0) != (s >= 0.0))) for s in self.sentinels_matrix], dtype=np.float32)
            
        top_sentinels = np.argsort(-sims)[:min(len(self.sentinels_ids), n_entry_points)]
        candidates = [self.sentinels_ids[idx] for idx in top_sentinels]
        
        if USE_CYTHON:
            return hive_ops.find_neighbors_hamming_cython(
                q_vec,
                candidates,
                k,
                beam_width,
                20,
                self.neighbor_cache,
                self.vector_cache,
                self.get_neighbors,
                self.get_vector,
                pheromones,
                pheromone_weight
            )
            
        visited = {}
        for c in candidates:
            score = -float(np.sum((q_vec >= 0.0) != (self.get_vector(c) >= 0.0)))
            p_val = 0.0
            if pheromones is not None and c in pheromones:
                p_val = float(pheromones[c])
            visited[c] = score + pheromone_weight * p_val
            
        for _ in range(20):
            next_candidates = []
            for curr in candidates:
                neighs = self.get_neighbors(curr)
                for n in neighs:
                    if n not in visited:
                        score = -float(np.sum((q_vec >= 0.0) != (self.get_vector(n) >= 0.0)))
                        p_val = 0.0
                        if pheromones is not None and n in pheromones:
                            p_val = float(pheromones[n])
                        score = score + pheromone_weight * p_val
                        visited[n] = score
                        next_candidates.append((n, score))
            
            if not next_candidates: break
            next_candidates.sort(key=lambda x: x[1], reverse=True)
            candidates = [n for n, s in next_candidates[:beam_width]]
            
        sorted_visited = sorted(visited.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, s_val in sorted_visited[:k]]

    def insert_vector(self, vec, new_id, k_neighbors=12, beam_width=5):
        vec = np.asarray(vec, dtype=np.float32)
        
        self._check_cache_limits()
        self.vector_cache[new_id] = vec
        
        neighbors = self.find_neighbors(vec, k=k_neighbors, beam_width=beam_width)
        
        v_off = self.store.append_vector(vec)
        n_off = self.store.append_graph_edges(neighbors)
        self.store.write_cell_meta(new_id, v_off, n_off, len(neighbors))
        
        self._check_cache_limits()
        self.meta_cache[new_id] = {"v_off": v_off, "n_off": n_off, "n_count": len(neighbors)}
        self.neighbor_cache[new_id] = np.array(neighbors, dtype=np.int32)
        
        for n in neighbors:
            n_meta = self.get_cell_meta(n)
            curr_neighs = list(self.get_neighbors(n))
            if new_id not in curr_neighs:
                curr_neighs.append(new_id)
                if len(curr_neighs) > 20:
                    curr_neighs = curr_neighs[-20:]
                    
                new_n_off = self.store.append_graph_edges(curr_neighs)
                self.store.write_cell_meta(n, n_meta["v_off"], new_n_off, len(curr_neighs))
                
                self._check_cache_limits()
                self.meta_cache[n] = {"v_off": n_meta["v_off"], "n_off": new_n_off, "n_count": len(curr_neighs)}
                self.neighbor_cache[n] = np.array(curr_neighs, dtype=np.int32)
        
        return new_id
