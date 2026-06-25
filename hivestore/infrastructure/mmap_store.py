import os
import mmap
import struct
import json
import numpy as np

from hivestore.core.interfaces import IVectorStore
from hivestore.infrastructure.lock import RWLock, ReadLockContext, WriteLockContext

class GrowableBuffer:
    def __init__(self, path, stride):
        self.path = path
        self.stride = stride
        initial_capacity = 256 * 1024 # 256 KB (small to force multiple resizes)
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(b'\x00' * initial_capacity)
        self.f = open(self.path, "r+b")
        self.mm = mmap.mmap(self.f.fileno(), 0)
        self.capacity = self.mm.size()
        self._tail = 0

    def _ensure_capacity(self, needed):
        if needed <= self.capacity: return
        new_cap = max(int(self.capacity * 1.5), needed)
        self.mm.close()
        self.f.truncate(new_cap)
        self.mm = mmap.mmap(self.f.fileno(), new_cap)
        self.capacity = new_cap

    def append(self, data_bytes):
        off = self._tail
        self._ensure_capacity(off + len(data_bytes))
        self.mm[off:off+len(data_bytes)] = data_bytes
        self._tail += len(data_bytes)
        return off

    def close(self):
        if hasattr(self, 'mm') and self.mm is not None:
            try:
                self.mm.close()
            except:
                pass
            self.mm = None
        if hasattr(self, 'f') and self.f is not None:
            try:
                self.f.close()
            except:
                pass
            self.f = None

    def __del__(self):
        try:
            self.close()
        except:
            pass

class DiskHiveStore(IVectorStore):
    def __init__(self, base_path, dimension):
        self.base_path = base_path
        self.D = dimension
        self.v_stride = dimension * 4
        self.c_stride = 32
        
        self.v_buf = GrowableBuffer(f"{base_path}_v.dat", self.v_stride)
        self.c_buf = GrowableBuffer(f"{base_path}_c.dat", self.c_stride)
        self.g_buf = GrowableBuffer(f"{base_path}_g.dat", 4)
        
        self._rwlock = RWLock()
        
        self.meta_path = f"{base_path}_meta.json"
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r") as f:
                meta = json.load(f)
                self.v_buf._tail = meta.get("v_tail", 0)
                self.c_buf._tail = meta.get("c_tail", 0)
                self.g_buf._tail = meta.get("g_tail", 0)
        else:
            self.save_meta()

    def read_lock(self):
        return ReadLockContext(self._rwlock)

    def write_lock(self):
        return WriteLockContext(self._rwlock)

    def save_meta(self):
        try:
            with open(self.meta_path, "w") as f:
                json.dump({
                    "v_tail": self.v_buf._tail,
                    "c_tail": self.c_buf._tail,
                    "g_tail": self.g_buf._tail
                }, f)
        except:
            pass

    def append_vector(self, vec):
        with self.write_lock():
            # Force conversion to float32 to ensure consistency with Cython expectations
            vec_f32 = vec.astype(np.float32)
            off = self.v_buf.append(vec_f32.tobytes())
            return off // self.v_stride

    def append_graph_edges(self, neighbors):
        with self.write_lock():
            off = self.g_buf.append(np.array(neighbors, dtype=np.int32).tobytes())
            return off

    def write_cell_meta(self, cell_id, v_off, n_off, n_count):
        with self.write_lock():
            data = struct.pack("<qiiiqi", v_off, 1, n_off, n_count, 0, 1)
            off = cell_id * self.c_stride
            self.c_buf._ensure_capacity(off + self.c_stride)
            self.c_buf.mm[off:off+32] = data
            if off + 32 > self.c_buf._tail: 
                self.c_buf._tail = off + 32

    def read_cell_meta(self, cell_id):
        with self.read_lock():
            off = cell_id * self.c_stride
            unpacked = struct.unpack("<qiiiqi", self.c_buf.mm[off:off+32])
            return {"v_off": unpacked[0], "n_off": unpacked[2], "n_count": unpacked[3]}

    def read_vector(self, idx):
        with self.read_lock():
            return np.frombuffer(self.v_buf.mm, dtype=np.float32, count=self.D, offset=idx * self.v_stride).copy()

    def read_neighbors(self, meta):
        with self.read_lock():
            if meta["n_count"] <= 0: return np.array([], dtype=np.int32)
            return np.frombuffer(self.g_buf.mm, dtype=np.int32, count=meta["n_count"], offset=meta["n_off"]).copy()

    def close(self):
        with self.write_lock():
            try:
                self.save_meta()
            except:
                pass
            self.v_buf.close()
            self.c_buf.close()
            self.g_buf.close()

    def __del__(self):
        try:
            self.close()
        except:
            pass
