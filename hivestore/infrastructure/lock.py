import threading
import time

class RWLock:
    def __init__(self):
        self.lock = threading.Lock()
        self.read_ready = threading.Condition(self.lock)
        self.readers = 0
        self.writers_waiting = 0
        self.writer_active = False
        self.total_read_wait_time = 0.0
        self.total_write_wait_time = 0.0
        self.read_acquires = 0
        self.write_acquires = 0

    def reset_stats(self):
        with self.lock:
            self.total_read_wait_time = 0.0
            self.total_write_wait_time = 0.0
            self.read_acquires = 0
            self.write_acquires = 0

    def get_stats(self):
        with self.lock:
            return {
                "total_read_wait": self.total_read_wait_time,
                "total_write_wait": self.total_write_wait_time,
                "read_acquires": self.read_acquires,
                "write_acquires": self.write_acquires
            }

    def acquire_read(self):
        t0 = time.time()
        with self.lock:
            while self.writer_active or self.writers_waiting > 0:
                self.read_ready.wait()
            self.readers += 1
            self.total_read_wait_time += (time.time() - t0)
            self.read_acquires += 1

    def release_read(self):
        with self.lock:
            self.readers -= 1
            if self.readers == 0:
                self.read_ready.notify_all()

    def acquire_write(self):
        t0 = time.time()
        with self.lock:
            self.writers_waiting += 1
            while self.writer_active or self.readers > 0:
                self.read_ready.wait()
            self.writers_waiting -= 1
            self.writer_active = True
            self.total_write_wait_time += (time.time() - t0)
            self.write_acquires += 1

    def release_write(self):
        with self.lock:
            self.writer_active = False
            self.read_ready.notify_all()

class ReadLockContext:
    def __init__(self, rwlock):
        self.rwlock = rwlock
    def __enter__(self):
        self.rwlock.acquire_read()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.rwlock.release_read()

class WriteLockContext:
    def __init__(self, rwlock):
        self.rwlock = rwlock
    def __enter__(self):
        self.rwlock.acquire_write()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.rwlock.release_write()
