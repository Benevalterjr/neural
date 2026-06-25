import collections
from hivestore.core.video_interfaces import ITieredCacheManager

class TieredCacheManager(ITieredCacheManager):
    def __init__(self, video_storage, max_hot_videos=500):
        self.storage = video_storage
        self.max_hot_videos = max_hot_videos
        
        # Hot Cache in RAM: maps video_id -> { 'av1': bytes, 'hnerv': np.ndarray, 'thumbnail': np.ndarray }
        # Uses OrderedDict to implement Least Recently Used (LRU) eviction
        self.hot_cache = collections.OrderedDict()
        
        # Watch statistics tracking with timestamps and retention rates
        self.watch_counts = collections.defaultdict(int)
        self.watch_timestamps = {}
        self.watch_completions = collections.defaultdict(list)

    def record_watch_completion(self, video_id: int, completion_ratio: float):
        """Records the watch completion ratio (0.0 to 1.0) of a video view."""
        self.watch_completions[video_id].append(float(completion_ratio))
        # Keep only the last 100 entries to prevent memory growth
        if len(self.watch_completions[video_id]) > 100:
            self.watch_completions[video_id].pop(0)

    def access_video(self, video_id: int) -> dict:
        """
        Loads a video's segments. Serves from Hot RAM cache if available,
        otherwise falls back to Warm Cache (mmap) for preview and Cold Storage (disk) for AV1.
        Updates LRU cache state and watch counts.
        """
        import time
        self.watch_counts[video_id] += 1
        self.watch_timestamps[video_id] = time.time()
        
        # Check Hot RAM Cache (Hits instantly)
        if video_id in self.hot_cache:
            # Move to end to mark as most recently used
            data = self.hot_cache.pop(video_id)
            self.hot_cache[video_id] = data
            return {
                "video_id": video_id,
                "av1_segment": data["av1"],
                "hnerv_weights": data["hnerv"],
                "thumbnail": data["thumbnail"],
                "cache_hit": "HOT_RAM"
            }
            
        # Cache Miss on Hot RAM -> Fetch from Warm Cache (mmap) and Cold Storage (disk)
        # HNeRV and Thumbnail are loaded instantly from Warm Cache (mmap)
        hnerv = self.storage.read_hnerv_weights(video_id)
        thumb = self.storage.read_thumbnail(video_id)
        
        # AV1 segment loaded sequentially from Cold Storage
        av1 = self.storage.read_av1_segment(video_id)
        
        # Put into Hot RAM Cache
        self.hot_cache[video_id] = {
            "av1": av1,
            "hnerv": hnerv,
            "thumbnail": thumb
        }
        
        # Handle Eviction: Keep Hot Cache size <= 500
        if len(self.hot_cache) > self.max_hot_videos:
            # Pop the oldest (first) item in the OrderedDict (LRU eviction)
            evicted_id, _ = self.hot_cache.popitem(last=False)
            # Evicted video returns to Warm Cache (mmap) / Cold Storage (disk) status
            
        return {
            "video_id": video_id,
            "av1_segment": av1,
            "hnerv_weights": hnerv,
            "thumbnail": thumb,
            "cache_hit": "COLD_DISK_WARM_MMAP"
        }

    def prefetch_videos(self, video_ids: list):
        """
        Prefetches HNeRV previews and thumbnails of the specified neighbor videos
        directly into the Hot RAM Cache to enable instant loading for the user.
        """
        prefetched_count = 0
        for vid in video_ids:
            if vid in self.hot_cache:
                continue  # Already in Hot RAM
                
            # Prefetch only HNeRV (neural preview) and Thumbnail into Hot RAM Cache
            hnerv = self.storage.read_hnerv_weights(vid)
            thumb = self.storage.read_thumbnail(vid)
            
            # Since AV1 is cold and large, we do not fetch AV1 to save RAM/bandwidth,
            # but we keep neural preview and thumbnail loaded in Hot RAM for instant playback startup.
            self.hot_cache[vid] = {
                "av1": b"",  # Lazy-loaded when actual play starts
                "hnerv": hnerv,
                "thumbnail": thumb
            }
            
            prefetched_count += 1
            
            # Evict if capacity exceeded
            if len(self.hot_cache) > self.max_hot_videos:
                self.hot_cache.popitem(last=False)
                
        return prefetched_count
