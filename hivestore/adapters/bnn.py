import numpy as np
from scipy import sparse
from tensorflow.keras.datasets import mnist

class StableSparseBNN:
    def __init__(self, layer_sizes, sparsity=0.15, seed=42):
        np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.sparsity = sparsity

        self.latent_weights = []
        self.masks = []

        for i in range(len(layer_sizes)-1):
            shape = (layer_sizes[i+1], layer_sizes[i])
            mask = sparse.random(*shape, density=sparsity,
                                 data_rvs=np.ones,
                                 random_state=seed+i).toarray().astype(np.float32)
            self.masks.append(mask)
            # Inicialização pequena para facilitar flips iniciais controlados
            w = np.random.normal(0, 0.05, size=shape).astype(np.float32)
            self.latent_weights.append(w)
            
        # Adam optimizer state para pesos latentes
        self.m = [np.zeros_like(w) for w in self.latent_weights]
        self.v = [np.zeros_like(w) for w in self.latent_weights]
        self.t = 0

    def stable_sign(self, x):
        return np.where(x >= 0, 1.0, -1.0)

    def get_binary_weights(self, i):
        return self.stable_sign(self.latent_weights[i]) * self.masks[i]

    def forward(self, X, training=False):
        a = X.T.astype(np.float32)
        activations = [a]
        z_values = []
        bn_cache = []
        Wb_list = []

        for i in range(len(self.latent_weights)):
            Wb = self.get_binary_weights(i)
            Wb_list.append(Wb)
            z_raw = Wb.dot(a)

            # Normalização (crucial para BNN)
            if i < len(self.latent_weights) - 1:
                mean = z_raw.mean(axis=1, keepdims=True)
                std = z_raw.std(axis=1, keepdims=True) + 1e-6
                z = (z_raw - mean) / std
                if training:
                    bn_cache.append(std)
            else:
                z = z_raw

            z_values.append(z)

            if i == 0:
                a = z # Camada de entrada contínua
            elif i < len(self.latent_weights) - 1:
                a = self.stable_sign(z)
            else:
                a = z # Logits

            activations.append(a)

        return (a.T, activations, z_values, bn_cache, Wb_list) if training else a.T

    def softmax(self, x):
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def train_step(self, X, y, lr=0.005):
        logits, activations, z_values, bn_cache, Wb_list = self.forward(X, training=True)
        probs = self.softmax(logits)
        loss = -np.mean(np.sum(y * np.log(probs + 1e-8), axis=1))

        # Divisão por batch_size para gradiente médio correto
        delta = (probs - y).T / X.shape[0]

        bn_idx = len(bn_cache) - 1
        self.t += 1
        
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8

        for i in range(len(self.latent_weights)-1, -1, -1):
            z = z_values[i]
            prev_a = activations[i]

            if i == len(self.latent_weights) - 1:
                grad = delta
            else:
                # Gradiente substituto (surrogate gradient) de arco tangente exato de SpikeGPT:
                # sigma'(z) = alpha / (2 * (1 + (pi/2 * alpha * z)^2))
                alpha_ste = 2.0
                val = (np.pi / 2.0) * alpha_ste * z
                ste = alpha_ste / (2.0 * (1.0 + val * val))
                dy = delta * ste
                
                # Backprop through batch norm
                std = bn_cache[bn_idx]
                bn_idx -= 1
                N = dy.shape[1]
                grad = (1.0 / (N * std)) * (N * dy - np.sum(dy, axis=1, keepdims=True) - z * np.sum(dy * z, axis=1, keepdims=True))

            dW = grad.dot(prev_a.T)

            # Adam update para pesos latentes
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * dW
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * (dW ** 2)
            
            m_hat = self.m[i] / (1.0 - beta1 ** self.t)
            v_hat = self.v[i] / (1.0 - beta2 ** self.t)
            
            step = lr * m_hat / (np.sqrt(v_hat) + eps)

            # Atualização + Clipping (Estabilização Crítica)
            self.latent_weights[i] -= step * self.masks[i]
            self.latent_weights[i] = np.clip(self.latent_weights[i], -1.5, 1.5)

            # Reutiliza o Wb do forward pass para backpropagation correto e rápido
            Wb = Wb_list[i]
            delta = Wb.T.dot(grad)

        return loss

    def evaluate(self, X, y):
        logits = self.forward(X)
        preds = np.argmax(logits, axis=1)
        labels = np.argmax(y, axis=1)
        return np.mean(preds == labels)
