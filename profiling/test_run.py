import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import numpy as np
from tensorflow.keras.datasets import mnist
from spiking_rwkv import SpikingRWKVMNIST

print("Loading data...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = (X_train.reshape(-1, 784).astype(np.float32) / 255.0 - 0.5) * 2
X_test  = (X_test.reshape(-1, 784).astype(np.float32) / 255.0 - 0.5) * 2
y_train = np.eye(10)[y_train]
y_test  = np.eye(10)[y_test]

print("Initializing model...")
model = SpikingRWKVMNIST(d_model=256, sparsity=0.15)

print("Running 1 train step...")
start = time.time()
model.train_step(X_train[0:256], y_train[0:256], lr=0.005)
print(f"Train step completed in {time.time() - start:.4f}s")

print("Evaluating...")
start = time.time()
test_acc = model.evaluate(X_test, y_test)
print(f"Evaluation completed in {time.time() - start:.4f}s")
print(f"Test Acc: {test_acc:.2%}")
