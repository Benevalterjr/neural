import time
import numpy as np
from spiking_rwkv import SpikingRWKVMNIST

X = np.random.randn(256, 784).astype(np.float32)
y = np.eye(10)[np.random.randint(0, 10, size=256)]

model = SpikingRWKVMNIST(d_model=256, sparsity=0.15)

print("Running one training step...")
start = time.time()
loss = model.train_step(X, y)
end = time.time()
print(f"Time for one step: {end - start:.4f} seconds")
