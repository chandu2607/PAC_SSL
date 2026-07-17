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
      - Theta Phase (4-8 Hz): analytic signal via frequency masking [4, 8] Hz -> angle -> theta_phase.
      - Gamma Amplitude (30-80 Hz): analytic signal via frequency masking [30, 80] Hz -> magnitude -> gamma_amplitude.
    Returns stacked 2-channel representation z-scored per segment: shape (B, C, 2, T) = [theta_phase_z, amp_gamma_z].
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
        X_full = torch.fft.fft(x, dim=-1) # (B, C, T)
        freqs_full = torch.fft.fftfreq(self.n_samples, d=1.0/self.fs).to(x.device)
        
        theta_full_mask = ((freqs_full >= 4.0) & (freqs_full <= 8.0)).float().unsqueeze(0).unsqueeze(0)
        gamma_full_mask = ((freqs_full >= 30.0) & (freqs_full <= 80.0)).float().unsqueeze(0).unsqueeze(0)
        
        # Analytic signal z(t) = 2 * IFFT(X(f) * u(f)) where u(f) is step function for positive freqs
        z_theta_complex = torch.fft.ifft(X_full * theta_full_mask * 2.0, dim=-1)
        z_gamma_complex = torch.fft.ifft(X_full * gamma_full_mask * 2.0, dim=-1)
        
        # Instantaneous Phase of Theta: angle of z_theta_complex
        phi_theta = torch.angle(z_theta_complex)
        
        # Instantaneous Envelope of Gamma: magnitude of z_gamma_complex
        amp_gamma = torch.abs(z_gamma_complex)
        
        # Z-score normalize each feature per segment along time dimension (dim=-1)
        phi_theta_z = (phi_theta - phi_theta.mean(dim=-1, keepdim=True)) / (phi_theta.std(dim=-1, keepdim=True) + 1e-6)
        amp_gamma_z = (amp_gamma - amp_gamma.mean(dim=-1, keepdim=True)) / (amp_gamma.std(dim=-1, keepdim=True) + 1e-6)
        
        # Stack features: (B, C, 2, T) = [phi_theta_z, amp_gamma_z]
        features = torch.stack([phi_theta_z, amp_gamma_z], dim=2)
        return features


class DualKernelCNNFrontEnd(nn.Module):
    """
    Encoder Stage 1: Lightweight per-channel CNN frequency-decomposition front-end with dual kernel sizes.
    Processes each channel independently.
    Dual kernel sizes:
      - K1 = 33 (~130 ms at 256Hz) for capturing slow theta dynamics & phase modulations.
      - K2 = 7 (~27 ms at 256Hz) for capturing fast gamma bursts & local envelope changes.
    """
    def __init__(self, in_features=3, out_channels=32):
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
    Reverted simpler, previously validated 3-layer 1D CNN encoder.
    Takes the standard 2-channel input (theta_phase, gamma_amplitude concatenated across the channel dimension, z-scored per segment).
    3-layer 1D CNN: Conv1d -> BatchNorm1d -> ReLU -> stride-2 downsampling -> AdaptiveAvgPool1d at the end.
    """
    def __init__(self, fs=256, n_samples=1024, num_channels=18, cnn_out=64, d_model=128, **kwargs):
        super().__init__()
        self.fs = fs
        self.n_samples = n_samples
        self.num_channels = num_channels
        self.d_model = d_model
        self.feature_extractor = PACFeatureExtractor(fs=fs, n_samples=n_samples)
        
        in_channels = num_channels * 2 # 18 * 2 = 36
        
        # 3-layer 1D CNN with stride-2 downsampling
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, d_model, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward_from_features(self, features):
        # features shape: (B, C, 2, T) e.g. (B, 18, 2, 1024)
        B, C, F_in, T = features.shape
        x_flat = features.view(B, C * F_in, T) # (B, 36, T)
        
        h1 = self.conv1(x_flat) # (B, 64, T//2)
        h2 = self.conv2(h1)     # (B, 128, T//4)
        h3 = self.conv3(h2)     # (B, d_model=128, T//8)
        
        z = self.pool(h3).squeeze(-1) # (B, d_model=128)
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
        # features: (B, C, 2, T) = [phi_theta_z, amp_gamma_z]
        z = self.encoder.forward_from_features(features) # (B, d_model)
        logits = self.classifier(z).squeeze(-1)          # (B,)
        return logits
