import os
import mmap
import numpy as np
from hivestore.core.video_interfaces import IVideoStorage

class DiskVideoStorage(IVideoStorage):
    def __init__(self, base_dir="video_storage", hnerv_dim=1024, thumb_dim=64):
        self.base_dir = base_dir
        self.hnerv_dim = hnerv_dim
        self.thumb_dim = thumb_dim
        
        self.hnerv_stride = hnerv_dim * 4  # float32 = 4 bytes
        self.thumb_stride = thumb_dim * thumb_dim * 4  # 64x64 float32
        
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(os.path.join(base_dir, "cold_av1"), exist_ok=True)
        
        self.hnerv_path = os.path.join(base_dir, "warm_hnerv.dat")
        self.thumb_path = os.path.join(base_dir, "warm_thumbs.dat")
        
        # Initialize flat binary files for HNeRV and Thumbnails if not exists
        for path in [self.hnerv_path, self.thumb_path]:
            if not os.path.exists(path):
                with open(path, "wb") as f:
                    f.write(b'\x00' * (1024 * 1024))  # Preallocate 1MB
                    
        self.hnerv_file = open(self.hnerv_path, "r+b")
        self.hnerv_mm = mmap.mmap(self.hnerv_file.fileno(), 0)
        
        self.thumb_file = open(self.thumb_path, "r+b")
        self.thumb_mm = mmap.mmap(self.thumb_file.fileno(), 0)

    def _ensure_file_capacity(self, mm_attr, file_obj, needed_size):
        mm = getattr(self, mm_attr)
        if needed_size <= mm.size():
            return mm
        new_size = max(int(mm.size() * 1.5), needed_size)
        mm.close()
        file_obj.truncate(new_size)
        new_mm = mmap.mmap(file_obj.fileno(), new_size)
        setattr(self, mm_attr, new_mm)
        return new_mm

    def store_video(self, video_id, av1_segment, hnerv_weights, thumbnail):
        # 1. Cold Storage: Store AV1 segment as sequential file
        av1_path = os.path.join(self.base_dir, "cold_av1", f"video_{video_id}.av1")
        with open(av1_path, "wb") as f:
            f.write(av1_segment)
            
        # 2. Warm Storage (mmap HNeRV weights)
        hnerv_offset = video_id * self.hnerv_stride
        self._ensure_file_capacity("hnerv_mm", self.hnerv_file, hnerv_offset + self.hnerv_stride)
        weights_f32 = np.asarray(hnerv_weights, dtype=np.float32)
        self.hnerv_mm[hnerv_offset:hnerv_offset + self.hnerv_stride] = weights_f32.tobytes()
        
        # 3. Warm Storage (mmap Thumbnail)
        thumb_offset = video_id * self.thumb_stride
        self._ensure_file_capacity("thumb_mm", self.thumb_file, thumb_offset + self.thumb_stride)
        thumb_f32 = np.asarray(thumbnail, dtype=np.float32)
        self.thumb_mm[thumb_offset:thumb_offset + self.thumb_stride] = thumb_f32.tobytes()

    def read_av1_segment(self, video_id):
        av1_path = os.path.join(self.base_dir, "cold_av1", f"video_{video_id}.av1")
        if not os.path.exists(av1_path):
            raise FileNotFoundError(f"Video {video_id} AV1 segment not found in Cold Storage.")
        with open(av1_path, "rb") as f:
            return f.read()

    def read_hnerv_weights(self, video_id):
        hnerv_offset = video_id * self.hnerv_stride
        if hnerv_offset + self.hnerv_stride > self.hnerv_mm.size():
            return np.zeros(self.hnerv_dim, dtype=np.float32)
        buf = self.hnerv_mm[hnerv_offset:hnerv_offset + self.hnerv_stride]
        return np.frombuffer(buf, dtype=np.float32).copy()

    def read_thumbnail(self, video_id):
        thumb_offset = video_id * self.thumb_stride
        if thumb_offset + self.thumb_stride > self.thumb_mm.size():
            return np.zeros((self.thumb_dim, self.thumb_dim), dtype=np.float32)
        buf = self.thumb_mm[thumb_offset:thumb_offset + self.thumb_stride]
        return np.frombuffer(buf, dtype=np.float32).reshape(self.thumb_dim, self.thumb_dim).copy()

    def close(self):
        if hasattr(self, 'hnerv_mm') and self.hnerv_mm is not None:
            try: self.hnerv_mm.close()
            except: pass
            self.hnerv_mm = None
        if hasattr(self, 'hnerv_file') and self.hnerv_file is not None:
            try: self.hnerv_file.close()
            except: pass
            self.hnerv_file = None
            
        if hasattr(self, 'thumb_mm') and self.thumb_mm is not None:
            try: self.thumb_mm.close()
            except: pass
            self.thumb_mm = None
        if hasattr(self, 'thumb_file') and self.thumb_file is not None:
            try: self.thumb_file.close()
            except: pass
            self.thumb_file = None

    def __del__(self):
        try: self.close()
        except: pass
