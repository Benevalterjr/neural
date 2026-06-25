import numpy as np

class HNeRVCodec:
    """
    Hybrid Neural Representation for Videos (HNeRV) Codec.
    Maps temporal frames into a lightweight factorized neural network weight vector (~10KB).
    """
    def __init__(self, n_frequencies=8, latent_dim=64, frame_dim=64, seed=42):
        self.n_frequencies = n_frequencies  # Generates 2 * n_frequencies positional features
        self.pos_dim = 2 * n_frequencies
        self.latent_dim = latent_dim
        self.frame_dim = frame_dim
        self.flat_img_dim = frame_dim * frame_dim
        self.seed = seed
        
        # Initialize a deterministic global spatial projection matrix shared across all videos (He initialized std ~ 0.15)
        np.random.seed(seed)
        self.W_global = np.random.normal(0, 0.15, size=(latent_dim, self.flat_img_dim)).astype(np.float32)
        self.b_global = -3.0  # Deterministic global negative bias for preview sparsity

    def _positional_encoding(self, tau):
        """Map temporal coordinate tau (0.0 to 1.0) into sinusoidal high-frequency features."""
        pe = []
        for i in range(self.n_frequencies):
            freq = (2 ** i) * np.pi
            pe.append(np.sin(freq * tau))
            pe.append(np.cos(freq * tau))
        return np.array(pe, dtype=np.float32)

    def encode(self, video_frames, epochs=60, lr=0.01):
        """
        Train a video-specific MLP to map temporal coordinates to spatial frame details.
        video_frames: numpy array of shape (T, 64, 64) with pixel values between 0.0 and 1.0.
        Returns:
            flat_weights: numpy array of shape (2656,) containing the trained weights.
        """
        T = video_frames.shape[0]
        Y = video_frames.reshape(T, self.flat_img_dim).astype(np.float32)
        
        # Precompute positional encodings for all frame indices
        taus = np.linspace(0.0, 1.0, T) if T > 1 else np.array([0.0])
        X_pe = np.array([self._positional_encoding(t) for t in taus], dtype=np.float32)
        
        # He initialization: std = sqrt(2 / fan_in)
        np.random.seed(999)  # Fixed initialization seed
        W1 = np.random.normal(0, 0.25, size=(32, self.pos_dim)).astype(np.float32)
        b1 = np.zeros((32, 1), dtype=np.float32)
        W2 = np.random.normal(0, 0.20, size=(self.latent_dim, 32)).astype(np.float32)
        b2 = np.zeros((self.latent_dim, 1), dtype=np.float32)
        
        # Adam optimizer state variables
        m_W1, v_W1 = np.zeros_like(W1), np.zeros_like(W1)
        m_b1, v_b1 = np.zeros_like(b1), np.zeros_like(b1)
        m_W2, v_W2 = np.zeros_like(W2), np.zeros_like(W2)
        m_b2, v_b2 = np.zeros_like(b2), np.zeros_like(b2)
        
        beta1, beta2 = 0.9, 0.999
        eps = 1e-8
        t_step = 0
        
        # Fast optimization loop
        for epoch in range(epochs):
            # Forward pass
            h0 = X_pe.T
            
            # Layer 1 (LeakyReLU activation)
            z1 = W1.dot(h0) + b1
            a1 = np.where(z1 > 0.0, z1, 0.01 * z1)
            
            # Layer 2 (LeakyReLU activation)
            z2 = W2.dot(a1) + b2
            a2 = np.where(z2 > 0.0, z2, 0.01 * z2)
            
            # Shared spatial projection and Sigmoid mapping
            logits = self.W_global.T.dot(a2) + self.b_global
            preds = 1.0 / (1.0 + np.exp(-np.clip(logits, -15.0, 15.0)))
            
            # Loss gradients
            diff = preds - Y.T
            d_logits = diff * preds * (1.0 - preds)
            
            # Backpropagation using LeakyReLU derivative
            da2 = self.W_global.dot(d_logits)
            dz2 = da2 * np.where(z2 > 0.0, 1.0, 0.01)
            
            dW2 = dz2.dot(a1.T) / T
            db2 = np.mean(dz2, axis=1, keepdims=True)
            
            da1 = W2.T.dot(dz2)
            dz1 = da1 * np.where(z1 > 0.0, 1.0, 0.01)
            
            dW1 = dz1.dot(h0.T) / T
            db1 = np.mean(dz1, axis=1, keepdims=True)
            
            # Update parameters using Adam
            t_step += 1
            
            # W1 & b1 updates
            m_W1 = beta1 * m_W1 + (1.0 - beta1) * dW1
            v_W1 = beta2 * v_W1 + (1.0 - beta2) * (dW1 ** 2)
            W1 -= lr * (m_W1 / (1.0 - beta1 ** t_step)) / (np.sqrt(v_W1 / (1.0 - beta2 ** t_step)) + eps)
            
            m_b1 = beta1 * m_b1 + (1.0 - beta1) * db1
            v_b1 = beta2 * v_b1 + (1.0 - beta2) * (db1 ** 2)
            b1 -= lr * (m_b1 / (1.0 - beta1 ** t_step)) / (np.sqrt(v_b1 / (1.0 - beta2 ** t_step)) + eps)
            
            # W2 & b2 updates
            m_W2 = beta1 * m_W2 + (1.0 - beta1) * dW2
            v_W2 = beta2 * v_W2 + (1.0 - beta2) * (dW2 ** 2)
            W2 -= lr * (m_W2 / (1.0 - beta1 ** t_step)) / (np.sqrt(v_W2 / (1.0 - beta2 ** t_step)) + eps)
            
            m_b2 = beta1 * m_b2 + (1.0 - beta1) * db2
            v_b2 = beta2 * v_b2 + (1.0 - beta2) * (db2 ** 2)
            b2 -= lr * (m_b2 / (1.0 - beta1 ** t_step)) / (np.sqrt(v_b2 / (1.0 - beta2 ** t_step)) + eps)
            
        # Concatenate weights into a single vector of shape (2656,)
        flat_weights = np.concatenate([
            W1.flatten(),
            b1.flatten(),
            W2.flatten(),
            b2.flatten()
        ])
        return flat_weights

    def decode(self, flat_weights, T):
        """
        Reconstruct the T frames of the video from the HNeRV neural weight vector.
        flat_weights: numpy array of shape (2656,).
        T: number of frames to decode.
        Returns:
            reconstructed_frames: numpy array of shape (T, 64, 64) with pixel values between 0.0 and 1.0.
        """
        # Slice flat weight vector
        idx = 0
        w1_size = 32 * self.pos_dim
        W1 = flat_weights[idx:idx + w1_size].reshape(32, self.pos_dim)
        idx += w1_size
        
        b1_size = 32
        b1 = flat_weights[idx:idx + b1_size].reshape(32, 1)
        idx += b1_size
        
        w2_size = self.latent_dim * 32
        W2 = flat_weights[idx:idx + w2_size].reshape(self.latent_dim, 32)
        idx += w2_size
        
        b2_size = self.latent_dim
        b2 = flat_weights[idx:idx + b2_size].reshape(self.latent_dim, 1)
        
        # Decoding pass
        taus = np.linspace(0.0, 1.0, T) if T > 1 else np.array([0.0])
        X_pe = np.array([self._positional_encoding(t) for t in taus], dtype=np.float32)
        
        h0 = X_pe.T
        z1 = W1.dot(h0) + b1
        a1 = np.where(z1 > 0.0, z1, 0.01 * z1)
        z2 = W2.dot(a1) + b2
        a2 = np.where(z2 > 0.0, z2, 0.01 * z2)
        
        logits = self.W_global.T.dot(a2) + self.b_global
        preds = 1.0 / (1.0 + np.exp(-np.clip(logits, -15.0, 15.0)))
        
        # Reshape flat predictions back to frame sequence shape (T, 64, 64)
        reconstructed_frames = preds.T.reshape(T, self.frame_dim, self.frame_dim)
        return reconstructed_frames
