import os
import numpy as np
from scipy import sparse

class SpikingRWKVMNIST:
    def __init__(self, d_model=256, sparsity=1.0, seed=42):
        np.random.seed(seed)
        self.d_model = d_model
        self.sparsity = sparsity
        
        # Dimensões das camadas (Entrada é 28, Saída é 10 classes)
        self.w_shapes = {
            'r': (d_model, 28),
            'k': (d_model, 28),
            'v': (d_model, 28),
            'g': (d_model, d_model),
            'u': (d_model, d_model),
            'out': (10, d_model)
        }
        
        self.latent_weights = {}
        self.masks = {}
        
        for name, shape in self.w_shapes.items():
            mask = sparse.random(*shape, density=sparsity,
                                 data_rvs=np.ones,
                                 random_state=seed + hash(name) % 1000).toarray().astype(np.float32)
            self.masks[name] = mask
            w = np.random.normal(0, 0.05, size=shape).astype(np.float32)
            self.latent_weights[name] = w
            
        # Parâmetros de decaimento temporal contínuos (SRWKV e LIF)
        self.wd = np.random.uniform(0.1, 0.9, size=(d_model, 1)).astype(np.float32)
        self.wf = np.random.uniform(0.1, 0.9, size=(d_model, 1)).astype(np.float32)
        
        # Estados do otimizador Adam
        self.m_w = {k: np.zeros_like(v) for k, v in self.latent_weights.items()}
        self.v_w = {k: np.zeros_like(v) for k, v in self.latent_weights.items()}
        
        self.m_wd = np.zeros_like(self.wd)
        self.v_wd = np.zeros_like(self.wd)
        self.m_wf = np.zeros_like(self.wf)
        self.v_wf = np.zeros_like(self.wf)
        
        self.t = 0

    def stable_sign(self, x):
        return np.where(x >= 0, 1.0, -1.0)

    def get_binary_weight(self, name):
        return self.stable_sign(self.latent_weights[name]) * self.masks[name]

    def forward(self, X, training=False):
        batch_size = X.shape[0]
        # Reshape para (batch_size, 28, 28) e depois transpõe para (28, 28, batch_size) para processamento sequencial
        X_seq = X.reshape(-1, 28, 28).transpose(1, 2, 0)
        
        Wb = {name: self.get_binary_weight(name) for name in self.latent_weights.keys()}
        
        wd = np.clip(self.wd, 0.01, 0.99)
        wf = np.clip(self.wf, 0.01, 0.99)
        
        e_prev = np.zeros((28, batch_size), dtype=np.float32)
        H_prev = np.zeros((self.d_model, batch_size), dtype=np.float32)
        V_prev = np.zeros((self.d_model, batch_size), dtype=np.float32)
        
        e_list = []
        e_shift_list = []
        R_raw_list = []
        K_raw_list = []
        V_val_raw_list = []
        R_list = []
        K_list = []
        V_val_list = []
        H_list = []
        O_list = []
        g_list = []
        u_list = []
        O_ffn_list = []
        V_list = []
        s_list = []
        
        # Loop Temporal - Linha por linha (Passo 2)
        for t in range(28):
            x_t = X_seq[t]
            
            # Passo 1: Binary Embedding
            e_t = np.where(x_t >= 0.0, 1.0, 0.0)
            e_list.append(e_t)
            
            # Token Shift
            e_shift_t = 0.5 * e_t + 0.5 * e_prev
            e_shift_list.append(e_shift_t)
            e_prev = e_t
            
            # Passo 3: Mistura de Tokens com Spiking RWKV (SRWKV)
            R_raw = Wb['r'].dot(e_shift_t)
            K_raw = Wb['k'].dot(e_shift_t)
            V_val_raw = Wb['v'].dot(e_shift_t)
            
            R_raw_list.append(R_raw)
            K_raw_list.append(K_raw)
            V_val_raw_list.append(V_val_raw)
            
            R_t = self.stable_sign(R_raw)
            K_t = self.stable_sign(K_raw)
            V_val_t = self.stable_sign(V_val_raw)
            
            R_list.append(R_t)
            K_list.append(K_t)
            V_val_list.append(V_val_t)
            
            # Atualização do Estado do RWKV
            H_t = wd * H_prev + K_t * V_val_t
            H_list.append(H_t)
            H_prev = H_t
            
            O_t = R_t * H_t
            O_list.append(O_t)
            
            # Passo 4: Filtragem com Feed-Forward (SRFFN)
            g_t = Wb['g'].dot(O_t)
            u_t = Wb['u'].dot(O_t)
            g_list.append(g_t)
            u_list.append(u_t)
            
            O_ffn_t = self.stable_sign(g_t) * self.stable_sign(u_t)
            O_ffn_list.append(O_ffn_t)
            
            # Passo 5: Dinâmica de Disparo do Neurônio LIF (Leaky Integrate-and-Fire)
            V_t = wf * V_prev + O_ffn_t
            s_t = np.where(V_t >= 1.0, 1.0, 0.0)
            
            V_prev = np.where(s_t > 0, 0.0, V_t)
            
            V_list.append(V_t)
            s_list.append(s_t)
            
        s_avg = np.mean(s_list, axis=0)
        logits = Wb['out'].dot(s_avg)
        
        if training:
            cache = (e_list, e_shift_list, R_raw_list, K_raw_list, V_val_raw_list, R_list, K_list, V_val_list, H_list, O_list, g_list, u_list, O_ffn_list, V_list, s_list, s_avg, Wb, wd, wf)
            return logits.T, cache
        else:
            return logits.T

    def surrogate_gradient(self, z, alpha=2.0):
        val = (np.pi / 2.0) * alpha * z
        return alpha / (2.0 * (1.0 + val * val))

    def train_step(self, X, y, lr=0.005):
        logits, cache = self.forward(X, training=True)
        probs = self.softmax(logits)
        loss = -np.mean(np.sum(y * np.log(probs + 1e-8), axis=1))

        batch_size = X.shape[0]
        delta = (probs - y).T / batch_size

        e_list, e_shift_list, R_raw_list, K_raw_list, V_val_raw_list, R_list, K_list, V_val_list, H_list, O_list, g_list, u_list, O_ffn_list, V_list, s_list, s_avg, Wb, wd, wf = cache
        
        self.t += 1
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        
        dWb = {k: np.zeros_like(v) for k, v in Wb.items()}
        dwd = np.zeros_like(self.wd)
        dwf = np.zeros_like(self.wf)
        
        dWb['out'] = delta.dot(s_avg.T)
        ds_avg = Wb['out'].T.dot(delta)
        ds_t = ds_avg / 28.0
        
        dV_next = np.zeros((self.d_model, batch_size), dtype=np.float32)
        dH_next = np.zeros((self.d_model, batch_size), dtype=np.float32)
        
        for t in range(27, -1, -1):
            s_t = s_list[t]
            V_t = V_list[t]
            O_ffn_t = O_ffn_list[t]
            g_t = g_list[t]
            u_t = u_list[t]
            O_t = O_list[t]
            R_t = R_list[t]
            H_t = H_list[t]
            K_t = K_list[t]
            V_val_t = V_val_list[t]
            e_shift_t = e_shift_list[t]
            
            # Gradiente LIF
            # V_t = wf * V_prev + O_ffn_t; s_t = (V_t >= 1.0)
            # Como resetamos V se s_t disparou, a dinâmica é:
            # V_prev = V_prev_raw se s_t_prev = 0 senão 0
            dV_t = dV_next + ds_t * self.surrogate_gradient(V_t - 1.0)
            dwf += np.sum(dV_t * (V_list[t-1] if t > 0 else 0.0), axis=1, keepdims=True)
            dV_next = dV_t * np.clip(wf, 0.01, 0.99) * (1.0 - s_t)
            
            dO_ffn = dV_t
            
            # FFN gradiente
            # O_ffn = sign(g) * sign(u)
            dg_sign = dO_ffn * self.stable_sign(u_t)
            du_sign = dO_ffn * self.stable_sign(g_t)
            
            dg = dg_sign * self.surrogate_gradient(g_t)
            du = du_sign * self.surrogate_gradient(u_t)
            
            dWb['g'] += dg.dot(O_t.T)
            dWb['u'] += du.dot(O_t.T)
            
            dO_t = Wb['g'].T.dot(dg) + Wb['u'].T.dot(du)
            
            # RWKV Mistura de Canais Gradiente
            # O_t = R_t * H_t
            dR_t = dO_t * H_t
            dH_t = dO_t * R_t + dH_next
            
            dR_raw = dR_t * self.surrogate_gradient(R_raw_list[t])
            dWb['r'] += dR_raw.dot(e_shift_t.T)
            
            # H_t = wd * H_prev + K_t * V_val_t
            dwd += np.sum(dH_t * (H_list[t-1] if t > 0 else 0.0), axis=1, keepdims=True)
            dH_next = dH_t * np.clip(wd, 0.01, 0.99)
            
            dK_t = dH_t * V_val_t
            dV_val_t = dH_t * K_t
            
            dK_raw = dK_t * self.surrogate_gradient(K_raw_list[t])
            dV_val_raw = dV_val_t * self.surrogate_gradient(V_val_raw_list[t])
            
            dWb['k'] += dK_raw.dot(e_shift_t.T)
            dWb['v'] += dV_val_raw.dot(e_shift_t.T)
            
        # Atualização dos parâmetros com Adam
        for k in self.latent_weights.keys():
            self.m_w[k] = beta1 * self.m_w[k] + (1.0 - beta1) * dWb[k]
            self.v_w[k] = beta2 * self.v_w[k] + (1.0 - beta2) * (dWb[k] ** 2)
            
            m_hat = self.m_w[k] / (1.0 - beta1 ** self.t)
            v_hat = self.v_w[k] / (1.0 - beta2 ** self.t)
            self.latent_weights[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)
            self.latent_weights[k] = np.clip(self.latent_weights[k], -1.5, 1.5)
            
        self.m_wd = beta1 * self.m_wd + (1.0 - beta1) * dwd
        self.v_wd = beta2 * self.v_wd + (1.0 - beta2) * (dwd ** 2)
        m_hat_d = self.m_wd / (1.0 - beta1 ** self.t)
        v_hat_d = self.v_wd / (1.0 - beta2 ** self.t)
        self.wd -= lr * m_hat_d / (np.sqrt(v_hat_d) + eps)
        
        self.m_wf = beta1 * self.m_wf + (1.0 - beta1) * dwf
        self.v_wf = beta2 * self.v_wf + (1.0 - beta2) * (dwf ** 2)
        m_hat_f = self.m_wf / (1.0 - beta1 ** self.t)
        v_hat_f = self.v_wf / (1.0 - beta2 ** self.t)
        self.wf -= lr * m_hat_f / (np.sqrt(v_hat_f) + eps)
        
        return loss

    def softmax(self, x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def evaluate(self, X, y):
        logits = self.forward(X)
        preds = np.argmax(logits, axis=1)
        labels = np.argmax(y, axis=1)
        return np.mean(preds == labels)


class SpikingRWKVTemporalExtractor:
    """Uses Spiking RWKV and LIF spike dynamics to compile video frame sequences into temporal embeddings."""
    def __init__(self, d_in=128, d_model=256, sparsity=1.0, seed=42):
        np.random.seed(seed)
        self.d_in = d_in
        self.d_model = d_model
        self.sparsity = sparsity
        
        self.w_shapes = {
            'r': (d_model, d_in),
            'k': (d_model, d_in),
            'v': (d_model, d_in),
            'g': (d_model, d_model),
            'u': (d_model, d_model)
        }
        
        self.latent_weights = {}
        self.masks = {}
        
        for name, shape in self.w_shapes.items():
            mask = sparse.random(*shape, density=sparsity,
                                 data_rvs=np.ones,
                                 random_state=seed + hash(name) % 1000).toarray().astype(np.float32)
            self.masks[name] = mask
            w = np.random.normal(0, 0.05, size=shape).astype(np.float32)
            self.latent_weights[name] = w
            
        self.wd = np.random.uniform(0.1, 0.9, size=(d_model, 1)).astype(np.float32)
        self.wf = np.random.uniform(0.1, 0.9, size=(d_model, 1)).astype(np.float32)

    def stable_sign(self, x):
        return np.where(x >= 0, 1.0, -1.0)

    def get_binary_weight(self, name):
        return self.stable_sign(self.latent_weights[name]) * self.masks[name]

    def extract_temporal_embedding(self, X_seq):
        """
        X_seq: numpy array of shape (T, d_in) representing the sequence of T frames.
        Returns:
            s_avg: 256-D float32 vector representing the temporally accumulated spiking features.
        """
        T = X_seq.shape[0]
        # Transpose to (d_in, T) for quick matrix multiplication
        X_seq_t = X_seq.T
        
        Wb = {name: self.get_binary_weight(name) for name in self.latent_weights.keys()}
        wd = np.clip(self.wd, 0.01, 0.99)
        wf = np.clip(self.wf, 0.01, 0.99)
        
        e_prev = np.zeros((self.d_in,), dtype=np.float32)
        H_prev = np.zeros((self.d_model, 1), dtype=np.float32)
        V_prev = np.zeros((self.d_model, 1), dtype=np.float32)
        
        s_list = []
        
        for t in range(T):
            x_t = X_seq_t[:, t:t+1]
            
            # Binary Embedding
            e_t = np.where(x_t >= 0.0, 1.0, 0.0)
            
            # Token Shift
            e_shift_t = 0.5 * e_t + 0.5 * e_prev.reshape(-1, 1)
            e_prev = e_t.flatten()
            
            # SRWKV Token Mix
            R_raw = Wb['r'].dot(e_shift_t)
            K_raw = Wb['k'].dot(e_shift_t)
            V_val_raw = Wb['v'].dot(e_shift_t)
            
            R_t = self.stable_sign(R_raw)
            K_t = self.stable_sign(K_raw)
            V_val_t = self.stable_sign(V_val_raw)
            
            # RWKV State Update
            H_t = wd * H_prev + K_t * V_val_t
            H_prev = H_t
            
            O_t = R_t * H_t
            
            # SRFFN filter
            g_t = Wb['g'].dot(O_t)
            u_t = Wb['u'].dot(O_t)
            O_ffn_t = self.stable_sign(g_t) * self.stable_sign(u_t)
            
            # LIF Neuron Spike
            V_t = wf * V_prev + O_ffn_t
            s_t = np.where(V_t >= 1.0, 1.0, 0.0)
            V_prev = np.where(s_t > 0, 0.0, V_t)
            
            s_list.append(s_t.flatten())
            
        s_avg = np.mean(s_list, axis=0)
        return s_avg.astype(np.float32)

