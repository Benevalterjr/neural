import time
import numpy as np
from spiking_rwkv import SpikingRWKVMNIST

X = np.random.randn(256, 784).astype(np.float32)
y = np.eye(10)[np.random.randint(0, 10, size=256)]

model = SpikingRWKVMNIST(d_model=256, sparsity=0.15)

# Profile forward pass
start = time.time()
logits, cache = model.forward(X, training=True)
fwd_time = time.time() - start

# Profile backward pass setup
start = time.time()
probs = model.softmax(logits)
loss = -np.mean(np.sum(y * np.log(probs + 1e-8), axis=1))
batch_size = X.shape[0]
delta = (probs - y).T / batch_size

e_list, e_shift_list, R_list, K_list, V_val_list, H_list, O_list, g_list, u_list, O_ffn_list, V_list, s_list, s_avg, Wb, wd, wf = cache
dWb = {k: np.zeros_like(v) for k, v in Wb.items()}
dwd = np.zeros_like(model.wd)
dwf = np.zeros_like(model.wf)
dWb['out'] = delta.dot(s_avg.T)
ds_avg = Wb['out'].T.dot(delta)
ds_t = ds_avg / 28.0
dV_next = np.zeros((model.d_model, batch_size), dtype=np.float32)
dH_next = np.zeros((model.d_model, batch_size), dtype=np.float32)
setup_time = time.time() - start

# Profile BPTT loop
start = time.time()
for t in range(27, -1, -1):
    s_t = s_list[t]
    V_t = V_list[t]
    O_ffn_t = O_ffn_list[t]
    g_t = g_list[t]
    u_t = u_list[t]
    O_t = O_list[t]
    R_t = R_list[t]
    K_t = K_list[t]
    V_val_t = V_val_list[t]
    H_t = H_list[t]
    H_prev = H_list[t-1] if t > 0 else np.zeros_like(H_t)
    V_prev = V_list[t-1] if t > 0 else np.zeros_like(V_t)
    e_shift_t = e_shift_list[t]
    
    ste_lif = model.surrogate_gradient(V_t - 1.0)
    dV_t = ds_t * ste_lif + dV_next * wf * (1.0 - s_t)
    dwf += np.sum(dV_t * V_prev, axis=1, keepdims=True)
    dO_ffn_t = dV_t
    dV_next = dV_t
    
    ste_g = model.surrogate_gradient(g_t)
    ste_u = model.surrogate_gradient(u_t)
    dg_t = dO_ffn_t * model.stable_sign(u_t) * ste_g
    du_t = dO_ffn_t * model.stable_sign(g_t) * ste_u
    
    dWb['g'] += dg_t.dot(O_t.T)
    dWb['u'] += du_t.dot(O_t.T)
    dO_t = Wb['g'].T.dot(dg_t) + Wb['u'].T.dot(du_t)
    
    dR_t_raw = dO_t * H_t
    dH_t_raw = dO_t * R_t
    
    ste_R = model.surrogate_gradient(Wb['r'].dot(e_shift_t))
    dR_t = dR_t_raw * ste_R
    dWb['r'] += dR_t.dot(e_shift_t.T)
    
    dH_t_total = dH_t_raw + dH_next
    dwd += np.sum(dH_t_total * H_prev, axis=1, keepdims=True)
    
    dK_t_raw = dH_t_total * V_val_t
    dV_val_t_raw = dH_t_total * K_t
    
    dH_next = dH_t_total * wd
    
    ste_K = model.surrogate_gradient(Wb['k'].dot(e_shift_t))
    dK_t = dK_t_raw * ste_K
    dWb['k'] += dK_t.dot(e_shift_t.T)
    
    ste_V = model.surrogate_gradient(Wb['v'].dot(e_shift_t))
    dV_val_t = dV_val_t_raw * ste_V
    dWb['v'] += dV_val_t.dot(e_shift_t.T)
bptt_time = time.time() - start

# Profile weights update
start = time.time()
for name in model.latent_weights.keys():
    dW = dWb[name]
    model.m_w[name] = beta1 = 0.9 * model.m_w[name] + 0.1 * dW
    model.v_w[name] = beta2 = 0.999 * model.v_w[name] + 0.001 * (dW ** 2)
    m_hat = model.m_w[name]
    v_hat = model.v_w[name]
    step = 0.005 * m_hat / (np.sqrt(v_hat) + 1e-8)
    model.latent_weights[name] -= step * model.masks[name]
    model.latent_weights[name] = np.clip(model.latent_weights[name], -1.5, 1.5)
update_time = time.time() - start

print(f"Forward pass time: {fwd_time:.4f}s")
print(f"Setup time: {setup_time:.4f}s")
print(f"BPTT loop time: {bptt_time:.4f}s")
print(f"Update time: {update_time:.4f}s")
