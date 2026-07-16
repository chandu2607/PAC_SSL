"""
pac_ssl_model.py — PAC-SSL Self-Supervised Pretraining Model Architecture
Implements:
  1. Fast GPU-based Hilbert Transform Feature Extraction (Theta Phase 4-8 Hz & Gamma Amplitude 30-80 Hz).
  2. Encoder Stage 1: Lightweight Per-Channel CNN Frequency-Decomposition Front-End with Dual Kernel Sizes.
  3. Encoder Stage 2: Learnable-Adjacency Graph Convolution across Channels (C=18).
  4. Encoder Stage 3: Temporal Transformer Sequence Model.
  5. Pretext Task Head: Binary classification of Genuine (label=1) vs Swapped (label=0) phase-amplitude couplings.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PACFeatureExtractor(nn.Module):
    """
    Extracts Hilbert-based Phase-Amplitude Coupling (PAC) representations directly on GPU using FFT.
    Given raw/filtered EEG window `x` of shape (B, C, T) where T=1024, fs=256 Hz:
      - Theta Phase (4-8 Hz): analytic signal via frequency masking [4, 8] Hz -> angle -> cos & sin.
      - Gamma Amplitude (30-80 Hz): analytic signal via frequency masking [30, 80] Hz -> magnitude.
    Returns stacked representation of shape (B, C, 4, T) = [x, cos_phi_theta, sin_phi_theta, amp_gamma].
    """
    def __init__(self, fs=256, n_samples=1024):
        super().__init__()
        self.fs = fs
        self.n_samples = n_samples
        
        # Precompute FFT frequency bins (0 to fs/2, length n_samples//2 + 1 = 513 bins)
        freqs = torch.fft.rfftfreq(n_samples, d=1.0/fs)
        
        # Masks for Theta (4-8 Hz) and Gamma (30-80 Hz)
        theta_mask = (freqs >= 4.0) & (freqs <= 8.0)
        gamma_mask = (freqs >= 30.0) & (freqs <= 80.0)
        
        self.register_buffer("theta_mask", theta_mask.float().unsqueeze(0).unsqueeze(0))
        self.register_buffer("gamma_mask", gamma_mask.float().unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        # x shape: (B, C, T)
        X_fft = torch.fft.rfft(x, dim=-1) # (B, C, 513)
        
        # Analytic signal for Theta
        # For real signal, analytic signal in freq domain has positive freqs doubled, negative zeroed.
        # rfft only stores non-negative frequencies, so we multiply non-DC/non-Nyquist by 2.
        X_theta = X_fft * self.theta_mask * 2.0
        z_theta = torch.fft.irfft(X_theta, n=self.n_samples, dim=-1) # complex not required since irfft returns real projection if we don't use ifft
        # To get true complex analytic signal in time domain, use torch.fft.ifft on full spectrum or construct complex:
        # Using full FFT:
        X_full = torch.fft.fft(x, dim=-1) # (B, C, T)
        freqs_full = torch.fft.fftfreq(self.n_samples, d=1.0/self.fs).to(x.device)
        
        theta_full_mask = ((freqs_full >= 4.0) & (freqs_full <= 8.0)).float().unsqueeze(0).unsqueeze(0)
        gamma_full_mask = ((freqs_full >= 30.0) & (freqs_full <= 80.0)).float().unsqueeze(0).unsqueeze(0)
        
        # Analytic signal z(t) = 2 * IFFT(X(f) * u(f)) where u(f) is step function for positive freqs
        z_theta_complex = torch.fft.ifft(X_full * theta_full_mask * 2.0, dim=-1)
        z_gamma_complex = torch.fft.ifft(X_full * gamma_full_mask * 2.0, dim=-1)
        
        # Instantaneous Phase of Theta: angle of z_theta_complex
        phi_theta = torch.angle(z_theta_complex)
        cos_phi = torch.cos(phi_theta)
        sin_phi = torch.sin(phi_theta)
        
        # Instantaneous Envelope of Gamma: magnitude of z_gamma_complex
        amp_gamma = torch.abs(z_gamma_complex)
        
        # Stack features: (B, C, 4, T)
        features = torch.stack([x, cos_phi, sin_phi, amp_gamma], dim=2)
        return features


class DualKernelCNNFrontEnd(nn.Module):
    """
    Encoder Stage 1: Lightweight per-channel CNN frequency-decomposition front-end with dual kernel sizes.
    Processes each channel independently.
    Dual kernel sizes:
      - K1 = 33 (~130 ms at 256Hz) for capturing slow theta dynamics & phase modulations.
      - K2 = 7 (~27 ms at 256Hz) for capturing fast gamma bursts & local envelope changes.
    """
    def __init__(self, in_features=4, out_channels=32):
        super().__init__()
        # Branch 1: Large kernel (K=33)
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_features, out_channels // 2, kernel_size=33, stride=2, padding=16, bias=False),
            nn.BatchNorm1d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4) # Downsamples by 4 -> total downsample = 8
        )
        # Branch 2: Small kernel (K=7)
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_features, out_channels // 2, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )
        # Further downsampling / temporal feature extraction
        self.fusion = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Output temporal sequence length from T=1024 is downsampled by 2 * 4 * 2 = 16 -> T'=64.

    def forward(self, x):
        # x: (B, C, 4, T) -> reshape to process channels independently: (B*C, 4, T)
        B, C, F_in, T = x.shape
        x_flat = x.view(B * C, F_in, T)
        
        out1 = self.branch1(x_flat) # (B*C, out//2, T//8 = 128)
        out2 = self.branch2(x_flat) # (B*C, out//2, T//8 = 128)
        out_cat = torch.cat([out1, out2], dim=1) # (B*C, out, 128)
        
        out = self.fusion(out_cat) # (B*C, out_channels, T//16 = 64)
        _, D, T_out = out.shape
        out = out.view(B, C, D, T_out) # (B, C, D, T')
        return out


class LearnableAdjacencyGCN(nn.Module):
    """
    Encoder Stage 2: Learnable-adjacency Graph Convolution across Channels (C=18).
    Captures cross-channel functional connectivity and synchronization across scalp electrodes.
    Adjacency matrix A is learned dynamically via electrode node embeddings.
    """
    def __init__(self, num_channels=18, embed_dim=16, feature_dim=32):
        super().__init__()
        self.num_channels = num_channels
        self.node_embeddings = nn.Parameter(torch.randn(num_channels, embed_dim) * 0.1)
        self.weight_gcn = nn.Linear(feature_dim, feature_dim)
        self.weight_self = nn.Linear(feature_dim, feature_dim)
        self.bn = nn.BatchNorm2d(feature_dim)

    def get_adjacency_matrix(self):
        # A = softmax(ReLU(E * E^T) / sqrt(d))
        scores = torch.matmul(self.node_embeddings, self.node_embeddings.T) / math.sqrt(self.node_embeddings.shape[1])
        adj = F.softmax(F.relu(scores), dim=-1)
        return adj

    def forward(self, x):
        # x: (B, C, D, T')
        B, C, D, T_out = x.shape
        adj = self.get_adjacency_matrix() # (C, C)
        
        # Permute to apply linear projection: (B, T', C, D)
        x_perm = x.permute(0, 3, 1, 2)
        
        # Spatial graph aggregation across C: A * X * W_gcn + X * W_self
        # adj shape (C, C), x_perm shape (B, T', C, D) -> matmul along C
        gcn_out = torch.matmul(adj, x_perm) # (B, T', C, D)
        gcn_out = self.weight_gcn(gcn_out) + self.weight_self(x_perm)
        
        # Permute back: (B, D, C, T') -> BatchNorm2d -> ReLU
        gcn_out = gcn_out.permute(0, 3, 2, 1) # (B, D, C, T')
        gcn_out = F.relu(self.bn(gcn_out), inplace=True)
        # Return in (B, C, D, T') shape
        return gcn_out.permute(0, 2, 1, 3)


class TemporalTransformerEncoder(nn.Module):
    """
    Encoder Stage 3: Temporal Transformer or equivalent sequence model.
    Pools/fuses spatial channel information and models long-range temporal dependencies over the 4-second window.
    """
    def __init__(self, num_channels=18, in_dim=32, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        # Spatial channel projection: aggregate across 18 channels to form global temporal tokens
        self.spatial_proj = nn.Sequential(
            nn.Linear(num_channels * in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Positional encoding for sequence length T' = 64
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, C, D, T') -> permute/reshape to (B, T', C*D)
        B, C, D, T_out = x.shape
        x_seq = x.permute(0, 3, 1, 2).contiguous().view(B, T_out, C * D)
        
        tokens = self.spatial_proj(x_seq) # (B, T_out, d_model)
        tokens = tokens + self.pos_embed[:, :T_out, :]
        
        out_seq = self.transformer(tokens) # (B, T_out, d_model)
        out_seq = self.norm(out_seq)
        
        # Temporal mean pool across the sequence to get final representation vector z (B, d_model)
        z = out_seq.mean(dim=1)
        return z


class PACSSLEncoder(nn.Module):
    """
    Full PAC-SSL Self-Supervised Encoder (Stages 1 -> 2 -> 3 exactly in order).
    """
    def __init__(self, fs=256, n_samples=1024, num_channels=18, cnn_out=32, d_model=128):
        super().__init__()
        self.feature_extractor = PACFeatureExtractor(fs=fs, n_samples=n_samples)
        self.stage1_cnn = DualKernelCNNFrontEnd(in_features=4, out_channels=cnn_out)
        self.stage2_gcn = LearnableAdjacencyGCN(num_channels=num_channels, embed_dim=16, feature_dim=cnn_out)
        self.stage3_transformer = TemporalTransformerEncoder(num_channels=num_channels, in_dim=cnn_out, d_model=d_model)

    def forward_from_features(self, features):
        """Forward pass starting from pre-extracted 4-channel PAC features (B, C, 4, T)."""
        h_cnn = self.stage1_cnn(features)       # Stage 1: (B, C, D, T')
        h_gcn = self.stage2_gcn(h_cnn)          # Stage 2: (B, C, D, T')
        z = self.stage3_transformer(h_gcn)      # Stage 3: (B, d_model)
        return z

    def forward(self, x):
        """Forward pass starting from raw/filtered EEG window (B, C, T)."""
        features = self.feature_extractor(x)
        return self.forward_from_features(features)


class PACSSLPretextModel(nn.Module):
    """
    Complete model for pretext task training:
    Takes pre-extracted PAC features, constructs Genuine (label=1) vs Swapped (label=0) couplings,
    passes through PACSSLEncoder, and outputs binary classification logits.
    """
    def __init__(self, encoder: PACSSLEncoder, d_model=128):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, features):
        # features: (B, C, 4, T) = [x, cos_phi, sin_phi, amp_gamma]
        z = self.encoder.forward_from_features(features) # (B, d_model)
        logits = self.classifier(z).squeeze(-1)          # (B,)
        return logits
