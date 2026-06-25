import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import numpy as np

# Create matrices
A = np.random.randn(256, 28).astype(np.float32)
B = np.random.randn(28, 256).astype(np.float32)

start = time.time()
for _ in range(1000):
    C = A.dot(B)
end = time.time()
print(f"Time for 1000 dot products of shape (256, 28) * (28, 256): {end - start:.6f} seconds")
