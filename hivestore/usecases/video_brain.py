import numpy as np
import time

class VideoDeliveryBrain:
    def __init__(self, hive_brain, video_storage, cache_manager):
        self.hive_brain = hive_brain
        self.storage = video_storage
        self.cache_manager = cache_manager
        self.temporal_extractor = None

    def register_video(self, video_id, embedding, av1_segment, hnerv_weights, thumbnail):
        """
        Coordinates the registration of a new video in the system:
        1. Indexes its BNN semantic embedding in the HiveStore graph database.
           If the input embedding is 2D (sequence of frame embeddings), it uses Spiking RWKV 
           to compile them into a neuromorphic temporal embedding.
        2. Writes its physical components (AV1 cold bytes, HNeRV warm weights, thumbnail) to the storage.
        """
        emb_np = np.asarray(embedding, dtype=np.float32)
        
        if emb_np.ndim == 2:
            # Sequential/temporal frames input -> extract neuromorphic temporal embedding via Spiking RWKV
            if self.temporal_extractor is None:
                from hivestore.adapters.spiking_rwkv import SpikingRWKVTemporalExtractor
                self.temporal_extractor = SpikingRWKVTemporalExtractor(d_in=emb_np.shape[1], d_model=256)
            emb_f32 = self.temporal_extractor.extract_temporal_embedding(emb_np)
        else:
            # Static embedding input
            emb_f32 = emb_np
            
        # 1. Index in HiveStore
        self.hive_brain.insert_vector(emb_f32, video_id, k_neighbors=12)
        
        # 2. Save physical video segments
        self.storage.store_video(video_id, av1_segment, hnerv_weights, thumbnail)

    def play_video(self, video_id, num_prefetch=4, previous_video_id=None, previous_completion_ratio=None):
        """
        Simulates playing a video:
        1. Accesses the video from the Tiered Cache (instantly loads to Hot RAM cache).
        2. Logs the watch completion feedback from the previous video, if available.
        3. Queries the HiveStore graph for the nearest neighbor video IDs (recommendations) guided by quality pheromones.
        4. Prefetches the neural previews and thumbnails of the top neighbor videos (predictive loading).
        """
        if previous_video_id is not None and previous_completion_ratio is not None:
            self.cache_manager.record_watch_completion(previous_video_id, previous_completion_ratio)

        print(f"\n[Video Delivery Brain] Iniciando reprodução do vídeo ID: {video_id}...")
        
        # 1. Load active video
        start_time = time.time()
        play_payload = self.cache_manager.access_video(video_id)
        load_duration = (time.time() - start_time) * 1000
        
        print(f"      - Vídeo carregado via cache [{play_payload['cache_hit']}] em {load_duration:.2f} ms")
        print(f"      - Estatísticas de visualização do vídeo {video_id}: {self.cache_manager.watch_counts[video_id]} views")
        
        # 2. Query recommendations based on Waggle Dance graph neighbors
        embedding = self.hive_brain.get_vector(video_id)
        
        # Calculate normalized and temporally decayed pheromone scores based on quality (completion rate)
        now = time.time()
        base_decay_lambda = 0.00005
        
        pheromones_dict = {}
        for vid, views in self.cache_manager.watch_counts.items():
            last_time = self.cache_manager.watch_timestamps.get(vid, now)
            dt = now - last_time
            
            # Calculate average watch completion rate as a quality metric
            completions = self.cache_manager.watch_completions[vid]
            avg_completion = np.mean(completions) if completions else 0.5
            
            # Cubing completion rate aggressively penalizes clickbaits:
            # 31% completion → 3% effective views (vs 9.6% with squaring)
            # This prevents high-volume low-quality videos from dominating pheromone rankings
            quality_weight = avg_completion ** 3
            effective_views = views * quality_weight
            
            # Dynamic decay lambda: videos with more high-quality views decay logarithmically slower
            dynamic_lambda = base_decay_lambda / np.log(np.e + effective_views)
            
            # Exponential time decay with dynamic lambda
            decayed_views = effective_views * np.exp(-dynamic_lambda * dt)
            
            # Logarithmic scaling log(1 + x) prevents large numbers from drowning out cosine similarity
            pheromones_dict[vid] = float(np.log1p(decayed_views))
            
        # Calibrate query weight based on recommendation preferences (default weight = 0.05)
        pheromone_weight = 0.05
        neighbor_ids = self.hive_brain.find_neighbors(
            embedding, 
            k=num_prefetch, 
            pheromones=pheromones_dict, 
            pheromone_weight=pheromone_weight
        )
        
        # Remove self if returned
        if video_id in neighbor_ids:
            neighbor_ids = [n for n in neighbor_ids if n != video_id]
            
        neighbor_ids = neighbor_ids[:num_prefetch]
        print(f"      - Recomendações 'For You' baseadas no grafo (Waggle Dance): {neighbor_ids}")
        
        # 3. Predictive Prefetching: Load previews and thumbnails of neighbors into Hot RAM Cache
        start_prefetch = time.time()
        prefetched = self.cache_manager.prefetch_videos(neighbor_ids)
        prefetch_duration = (time.time() - start_prefetch) * 1000
        
        print(f"      - Prefetching Preditivo: {prefetched} previews neurais/thumbnails carregados na RAM em {prefetch_duration:.2f} ms")
        
        return {
            "video_id": video_id,
            "av1_segment_size": len(play_payload["av1_segment"]),
            "hnerv_preview_size": play_payload["hnerv_weights"].shape[0],
            "thumbnail_shape": play_payload["thumbnail"].shape,
            "cache_source": play_payload["cache_hit"],
            "prefetched_neighbors": neighbor_ids,
            "hot_cache_size": len(self.cache_manager.hot_cache)
        }
