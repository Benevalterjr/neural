import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import numpy as np
from spiking_rwkv import SpikingRWKVMNIST

X = np.random.randn(256, 784).astype(np.float32)
model = SpikingRWKVMNIST(d_model=256, sparsity=0.15)

batch_size = X.shape[0]
X_seq = X.reshape(-1, 28, 28).transpose(1, 2, 0)
Wb = {name: model.get_binary_weight(name) for name in model.latent_weights.keys()}
wd = np.clip(model.wd, 0.01, 0.99)
wf = np.clip(model.wf, 0.01, 0.99)

e_prev = np.zeros((28, batch_size), dtype=np.float32)
H_prev = np.zeros((model.d_model, batch_size), dtype=np.float32)
V_prev = np.zeros((model.d_model, batch_size), dtype=np.float32)

times = {
    'emb': 0.0,
    'shift': 0.0,
    'rkv': 0.0,
    'state': 0.0,
    'ffn': 0.0,
    'lif': 0.0
}

for t in range(28):
    x_t = X_seq[t]
    
    t0 = time.time()
    e_t = np.where(x_t >= 0.0, 1.0, 0.0)
    times['emb'] += time.time() - t0
    
    t0 = time.time()
    e_shift_t = 0.5 * e_t + 0.5 * e_prev
    e_prev = e_t
    times['shift'] += time.time() - t0
    
    t0 = time.time()
    R_t = model.stable_sign(Wb['r'].dot(e_shift_t))
    K_t = model.stable_sign(Wb['k'].dot(e_shift_t))
    V_val_t = model.stable_sign(Wb['v'].dot(e_shift_t))
    times['rkv'] += time.time() - t0
    
    t0 = time.time()
    H_t = wd * H_prev + K_t * V_val_t
    H_prev = H_t
    O_t = R_t * H_t
    times['state'] += time.time() - t0
    
    t0 = time.time()
    g_t = Wb['g'].dot(O_t)
    u_t = Wb['u'].dot(O_t)
    O_ffn_t = model.stable_sign(g_t) * model.stable_sign(u_t)
    times['ffn'] += time.time() - t0
    
    t0 = time.time()
    V_t = wf * V_prev + O_ffn_t
    s_t = np.where(V_t >= 1.0, 1.0, 0.0)
    V_prev = np.where(s_t > 0, 0.0, V_t)
    times['lif'] += time.time() - t0

for k, v in times.items():
    print(f"{k}: {v:.6f}s")
