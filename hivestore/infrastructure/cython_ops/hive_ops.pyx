# cython: language_level=3
import numpy as np
cimport numpy as cn
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_dot(float[:] a, float[:] b):
    """Calculates cosine similarity dot product in optimized C."""
    cdef int i
    cdef int n = a.shape[0]
    cdef float val = 0.0
    for i in range(n):
        val += a[i] * b[i]
    return val

@cython.boundscheck(False)
@cython.wraparound(False)
def local_search_cython(
    float[:] q_vec, 
    list initial_candidates, 
    int beam_width, 
    int max_hops,
    dict neighbor_cache,
    dict vector_cache,
    object get_neighbors_fallback_func, 
    object get_vector_fallback_func,
    dict pheromones=None,
    float pheromone_weight=0.0
):
    """
    Optimized beam search on the graph starting from multiple entry points (coarse quantization)
    guided by a heuristic combining cosine similarity and pheromone scores.
    """
    cdef list candidates = list(initial_candidates)
    cdef set visited = set(initial_candidates)
    
    cdef int best_node = candidates[0]
    cdef float[:] best_vec
    if best_node in vector_cache:
        best_vec = vector_cache[best_node]
    else:
        best_vec = get_vector_fallback_func(best_node)
        
    cdef float best_sim = fast_dot(q_vec, best_vec)
    cdef float best_score = best_sim
    cdef float p_val = 0.0
    if pheromones is not None and best_node in pheromones:
        p_val = <float>pheromones[best_node]
    best_score += pheromone_weight * p_val
    
    cdef int hop, curr, n
    cdef float sim, score
    cdef list next_candidates
    cdef object neighs
    cdef float[:] vec_n
    
    for hop in range(max_hops):
        next_candidates = []
        for curr in candidates:
            if curr in neighbor_cache:
                neighs = neighbor_cache[curr]
            else:
                neighs = get_neighbors_fallback_func(curr)
                
            for n in neighs:
                if n not in visited:
                    visited.add(n)
                    if n in vector_cache:
                        vec_n = vector_cache[n]
                    else:
                        vec_n = get_vector_fallback_func(n)
                        
                    sim = fast_dot(q_vec, vec_n)
                    p_val = 0.0
                    if pheromones is not None and n in pheromones:
                        p_val = <float>pheromones[n]
                    score = sim + pheromone_weight * p_val
                    next_candidates.append((n, score))
                    
                    if score > best_score:
                        best_score = score
                        best_node = n
                        
        if not next_candidates:
            break
            
        next_candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = [n for n, s in next_candidates[:beam_width]]
        
    return best_node

@cython.boundscheck(False)
@cython.wraparound(False)
def find_neighbors_cython(
    float[:] q_vec,
    list initial_candidates,
    int k,
    int beam_width,
    int max_hops,
    dict neighbor_cache,
    dict vector_cache,
    object get_neighbors_fallback_func,
    object get_vector_fallback_func,
    dict pheromones=None,
    float pheromone_weight=0.0
):
    """
    Finds the top-k nearest neighbors on the graph starting from initial candidates (coarse quantization)
    guided by a heuristic combining cosine similarity and pheromone scores.
    """
    cdef list candidates = list(initial_candidates)
    cdef dict visited = {}
    cdef int c, n
    cdef float sim, score, p_val
    cdef list next_candidates
    cdef object neighs
    cdef float[:] vec_n
    cdef float[:] vec_c
    
    for c in candidates:
        if c in vector_cache:
            vec_c = vector_cache[c]
        else:
            vec_c = get_vector_fallback_func(c)
        sim = fast_dot(q_vec, vec_c)
        p_val = 0.0
        if pheromones is not None and c in pheromones:
            p_val = <float>pheromones[c]
        visited[c] = sim + pheromone_weight * p_val
        
    for _ in range(max_hops):
        next_candidates = []
        for curr in candidates:
            if curr in neighbor_cache:
                neighs = neighbor_cache[curr]
            else:
                neighs = get_neighbors_fallback_func(curr)
                
            for n in neighs:
                if n not in visited:
                    if n in vector_cache:
                        vec_n = vector_cache[n]
                    else:
                        vec_n = get_vector_fallback_func(n)
                    sim = fast_dot(q_vec, vec_n)
                    p_val = 0.0
                    if pheromones is not None and n in pheromones:
                        p_val = <float>pheromones[n]
                    score = sim + pheromone_weight * p_val
                    visited[n] = score
                    next_candidates.append((n, score))
                    
        if not next_candidates:
            break
            
        next_candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = [n for n, s in next_candidates[:beam_width]]
        
    cdef list sorted_visited = sorted(visited.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, s_val in sorted_visited[:k]]

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_hamming(float[:] a, float[:] b):
    """Calculates Hamming distance between two float vectors based on sign difference."""
    cdef int i
    cdef int n = a.shape[0]
    cdef int dist = 0
    for i in range(n):
        if (a[i] >= 0.0) != (b[i] >= 0.0):
            dist += 1
    return dist

@cython.boundscheck(False)
@cython.wraparound(False)
def merge_results_cython(list shard_results, float[:] query_vec, int k):
    """
    Aggregates results from multiple shards, calculates dot product similarity using fast_dot,
    and returns top-k sorted items.
    
    shard_results: list of tuples (global_id, float[:] vector, int shard_id)
    Returns: list of tuples (global_id, score, shard_id) sorted by score descending.
    """
    cdef int i
    cdef int num_results = len(shard_results)
    cdef list scored_results = []
    cdef object item
    cdef int global_id, shard_id
    cdef float[:] vec
    cdef float sim
    
    for i in range(num_results):
        item = shard_results[i]
        global_id = item[0]
        vec = item[1]
        shard_id = item[2]
        
        sim = fast_dot(query_vec, vec)
        scored_results.append((global_id, sim, shard_id))
        
    scored_results.sort(key=lambda x: x[1], reverse=True)
    return scored_results[:k]

@cython.boundscheck(False)
@cython.wraparound(False)
def local_search_hamming_cython(
    float[:] q_vec, 
    list initial_candidates, 
    int beam_width, 
    int max_hops,
    dict neighbor_cache,
    dict vector_cache,
    object get_neighbors_fallback_func, 
    object get_vector_fallback_func,
    dict pheromones=None,
    float pheromone_weight=0.0
):
    """
    Optimized beam search on the graph starting from multiple entry points
    guided by a heuristic combining negative Hamming distance and pheromone scores.
    """
    cdef list candidates = list(initial_candidates)
    cdef set visited = set(initial_candidates)
    
    cdef int best_node = candidates[0]
    cdef float[:] best_vec
    if best_node in vector_cache:
        best_vec = vector_cache[best_node]
    else:
        best_vec = get_vector_fallback_func(best_node)
        
    cdef float best_score = -<float>fast_hamming(q_vec, best_vec)
    cdef float p_val = 0.0
    if pheromones is not None and best_node in pheromones:
        p_val = <float>pheromones[best_node]
    best_score += pheromone_weight * p_val
    
    cdef int hop, curr, n
    cdef float score
    cdef list next_candidates
    cdef object neighs
    cdef float[:] vec_n
    
    for hop in range(max_hops):
        next_candidates = []
        for curr in candidates:
            if curr in neighbor_cache:
                neighs = neighbor_cache[curr]
            else:
                neighs = get_neighbors_fallback_func(curr)
                
            for n in neighs:
                if n not in visited:
                    visited.add(n)
                    if n in vector_cache:
                        vec_n = vector_cache[n]
                    else:
                        vec_n = get_vector_fallback_func(n)
                        
                    score = -<float>fast_hamming(q_vec, vec_n)
                    p_val = 0.0
                    if pheromones is not None and n in pheromones:
                        p_val = <float>pheromones[n]
                    score += pheromone_weight * p_val
                    next_candidates.append((n, score))
                    
                    if score > best_score:
                        best_score = score
                        best_node = n
                        
        if not next_candidates:
            break
            
        next_candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = [n for n, s in next_candidates[:beam_width]]
        
    return best_node

@cython.boundscheck(False)
@cython.wraparound(False)
def find_neighbors_hamming_cython(
    float[:] q_vec,
    list initial_candidates,
    int k,
    int beam_width,
    int max_hops,
    dict neighbor_cache,
    dict vector_cache,
    object get_neighbors_fallback_func,
    object get_vector_fallback_func,
    dict pheromones=None,
    float pheromone_weight=0.0
):
    """
    Finds the top-k nearest neighbors on the graph starting from initial candidates
    guided by a heuristic combining negative Hamming distance and pheromone scores.
    """
    cdef list candidates = list(initial_candidates)
    cdef dict visited = {}
    cdef int c, n
    cdef float score, p_val
    cdef list next_candidates
    cdef object neighs
    cdef float[:] vec_n
    cdef float[:] vec_c
    
    for c in candidates:
        if c in vector_cache:
            vec_c = vector_cache[c]
        else:
            vec_c = get_vector_fallback_func(c)
        score = -<float>fast_hamming(q_vec, vec_c)
        p_val = 0.0
        if pheromones is not None and c in pheromones:
            p_val = <float>pheromones[c]
        visited[c] = score + pheromone_weight * p_val
        
    for _ in range(max_hops):
        next_candidates = []
        for curr in candidates:
            if curr in neighbor_cache:
                neighs = neighbor_cache[curr]
            else:
                neighs = get_neighbors_fallback_func(curr)
                
            for n in neighs:
                if n not in visited:
                    if n in vector_cache:
                        vec_n = vector_cache[n]
                    else:
                        vec_n = get_vector_fallback_func(n)
                    score = -<float>fast_hamming(q_vec, vec_n)
                    p_val = 0.0
                    if pheromones is not None and n in pheromones:
                        p_val = <float>pheromones[n]
                    score += pheromone_weight * p_val
                    visited[n] = score
                    next_candidates.append((n, score))
                    
        if not next_candidates:
            break
            
        next_candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = [n for n, s in next_candidates[:beam_width]]
        
    cdef list sorted_visited = sorted(visited.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, s_val in sorted_visited[:k]]

