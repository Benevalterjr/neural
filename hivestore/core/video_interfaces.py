import abc
import numpy as np

class IVideoStorage(abc.ABC):
    @abc.abstractmethod
    def store_video(self, video_id: int, av1_segment: bytes, hnerv_weights: np.ndarray, thumbnail: np.ndarray) -> None:
        """Stores a video's AV1 segment, HNeRV weights, and thumbnail."""
        pass

    @abc.abstractmethod
    def read_av1_segment(self, video_id: int) -> bytes:
        """Reads the full AV1 segment for a video from cold storage."""
        pass

    @abc.abstractmethod
    def read_hnerv_weights(self, video_id: int) -> np.ndarray:
        """Reads HNeRV weights (neural preview) for a video."""
        pass

    @abc.abstractmethod
    def read_thumbnail(self, video_id: int) -> np.ndarray:
        """Reads the thumbnail for a video."""
        pass


class ITieredCacheManager(abc.ABC):
    @abc.abstractmethod
    def access_video(self, video_id: int) -> dict:
        """Accesses a video, updating watch statistics and loading parts to Hot RAM Cache."""
        pass

    @abc.abstractmethod
    def prefetch_videos(self, video_ids: list) -> None:
        """Prefetches thumbnails and HNeRV previews of the specified videos into the cache."""
        pass
