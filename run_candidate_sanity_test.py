"""
run_candidate_sanity_test.py — Small-Scale Sanity Run for Candidate Architectures (Rule 5)
Tests:
  - Candidate 1: Base CNN + Training Protocol Fixes (Cosine LR warmup/decay, AdamW, balanced batches)
  - Candidate 2: Learnable-Adjacency GCN across channels (C=18) + Protocol Fixes
  - Candidate 3: Full Dual-Kernel Frequency Decomposition CNN + GCN + Temporal Transformer + Protocol Fixes
Runs 2 epochs of pretext pretraining on subsample (chb01, chb02, chb19, chb20), checks VRAM/time per epoch,
and verifies feature extraction and personal normalization + velocity evaluation.
"""
import os
import sys
import time
import math
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

from pac_ssl_model import (
    PACFeatureExtractor, DualKernelCNNFrontEnd, LearnableAdjacencyGCN,
    TemporalTransformerEncoder, PACSSLEncoder
)
from train_pretext_stage2 import MmapPreprocessedEEGDataset, BlockRandomSampler, generate_pac_couplings_on_gpu
from lopo_evaluation import compute_roc_auc_numpy, train_classifier_head_on_fold
from run_stage4_calibration import get_patient_block_ids
from run_personal_norm_velocity_experiment import compute_smoothed_velocity_features
from lopo_v2 import smart_calibration_block

DATA_ROOT = Path("data/preprocessed")

# ==========================================
# Candidate 2: GCN Encoder
# ==========================================
class PACSSLEncoderGCN(nn.Module):
    """
    Candidate 2: Learnable-Adjacency Graph Convolution across 18 channels.
    Processes each channel (2 features: theta_phase_z, amp_gamma_z) via 1D CNN,
    then applies GCN across C=18 electrodes, then pools/projects down to d_model=128.
    """
    def __init__(self, fs=256, n_samples=1024, num_channels=18, d_model=128, **kwargs):
        super().__init__()
        self.num_channels = num_channels
        self.d_model = d_model
        self.feature_extractor = PACFeatureExtractor(fs=fs, n_samples=n_samples)
        
        # Per-channel front-end: (B*C, 2, 1024) -> (B*C, 32, 64)
        self.channel_cnn = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 32, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        ) # T' = 1024 / (2*4*2) = 64
        
        # Graph convolution across C=18 channels
        self.gcn = LearnableAdjacencyGCN(num_channels=num_channels, embed_dim=16, feature_dim=32)
        
        # Temporal pooling / projection across C and T' to d_model vector
        self.proj = nn.Sequential(
            nn.Conv1d(num_channels * 32, 256, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, d_model, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward_from_features(self, features):
        # features: (B, C, 2, T)
        B, C, F_in, T = features.shape
        x_flat = features.view(B * C, F_in, T)
        h_cnn = self.channel_cnn(x_flat) # (B*C, 32, T'=64)
        _, D, T_out = h_cnn.shape
        h_spatial = h_cnn.reshape(B, C, D, T_out) # (B, C, 32, 64)
        
        # GCN across channels
        h_gcn = self.gcn(h_spatial) # (B, C, 32, 64)
        
        # Flatten C and D for temporal projection using .reshape() to handle permuted layouts safely
        h_seq = h_gcn.reshape(B, C * D, T_out) # (B, 18*32, 64)
        h_proj = self.proj(h_seq) # (B, d_model=128, 16)
        z = self.pool(h_proj).squeeze(-1) # (B, 128)
        return z

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.forward_from_features(features)


# ==========================================
# Candidate 3: Dual-Kernel CNN + GCN + Transformer Encoder
# ==========================================
class PACSSLEncoderFull(nn.Module):
    """
    Candidate 3: Full state-of-the-art architecture.
    1. DualKernelCNNFrontEnd (K1=33 for slow theta, K2=7 for fast gamma amplitude bursts -> T'=64).
    2. LearnableAdjacencyGCN across C=18 spatial channels.
    3. TemporalTransformerEncoder (nhead=4, layers=2) to fuse spatial channels and model long sequences.
    """
    def __init__(self, fs=256, n_samples=1024, num_channels=18, d_model=128, **kwargs):
        super().__init__()
        self.num_channels = num_channels
        self.d_model = d_model
        self.feature_extractor = PACFeatureExtractor(fs=fs, n_samples=n_samples)
        
        self.front_end = DualKernelCNNFrontEnd(in_features=2, out_channels=32)
        self.gcn = LearnableAdjacencyGCN(num_channels=num_channels, embed_dim=16, feature_dim=32)
        self.transformer = TemporalTransformerEncoder(
            num_channels=num_channels, in_dim=32, d_model=d_model, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1
        )

    def forward_from_features(self, features):
        # features: (B, C, 2, T)
        h_fe = self.front_end(features) # (B, C, 32, T'=64)
        h_gcn = self.gcn(h_fe)          # (B, C, 32, 64)
        z = self.transformer(h_gcn)     # (B, d_model=128)
        return z

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.forward_from_features(features)


# ==========================================
# Common Pretext Classification Model Wrapper
# ==========================================
class PACSSLPretextModelWrapper(nn.Module):
    def __init__(self, encoder, d_model=128):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, features):
        z = self.encoder.forward_from_features(features)
        return self.classifier(z).squeeze(-1)


def run_sanity_for_candidate(name, encoder_cls, train_loader, val_loader, device, epochs=2):
    print(f"\n=====================================================================================", flush=True)
    print(f"=== Sanity Test: {name} ===", flush=True)
    print(f"=====================================================================================", flush=True)
    
    fe = PACFeatureExtractor(fs=256, n_samples=1024).to(device)
    encoder = encoder_cls(fs=256, n_samples=1024, num_channels=18, d_model=128).to(device)
    model = PACSSLPretextModelWrapper(encoder=encoder, d_model=128).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable Parameters: {n_params:,}", flush=True)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Protocol D fix: Cosine Annealing learning rate with warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        
    t0_start = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        total_loss, correct, total_samples = 0.0, 0, 0
        t_ep = time.time()
        
        for batch_idx, raw_x in enumerate(train_loader, 1):
            raw_x = raw_x.to(device, non_blocking=True)
            with torch.no_grad():
                feats = fe(raw_x)
                xc, yc = generate_pac_couplings_on_gpu(feats, device)
                
            optimizer.zero_grad()
            logits = model(xc)
            loss = criterion(logits, yc)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Protocol D: gradient clipping
            optimizer.step()
            
            total_loss += loss.item() * len(yc)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == yc).sum().item()
            total_samples += len(yc)
            
            if batch_idx % 50 == 0 or batch_idx == len(train_loader):
                elapsed = time.time() - t_ep
                print(f"  [{name}] Epoch {ep}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {total_loss/total_samples:.4f} | Acc: {correct/total_samples*100:.1f}% | Time: {elapsed:.1f}s", flush=True)
                
        scheduler.step()
        
    ep_time = (time.time() - t0_start) / epochs
    vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if torch.cuda.is_available() else 0.0
    print(f"\n[Sanity Result - {name}] Avg Time/Epoch: {ep_time:.1f}s | Max VRAM: {vram_mb:.1f} MB | Final Acc: {correct/total_samples*100:.2f}%", flush=True)
    
    # Estimate full training time across all 21 patients (Full dataset is ~383,819 windows vs Subsample ~80,000 windows -> ~4.8x batches)
    est_full_ep_min = (ep_time * 4.8) / 60.0
    print(f"[Cost Estimate - {name}] Full dataset (21 patients) time per epoch: ~{est_full_ep_min:.1f} minutes (~{est_full_ep_min*10:.1f} min for 10 epochs)", flush=True)
    
    return encoder, ep_time, vram_mb


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Candidate Sanity Runs starting on {device} ===", flush=True)
    
    sanity_patients = ["chb01", "chb02", "chb19", "chb20"]
    print(f"Loading subsample dataset across {sanity_patients}...", flush=True)
    dataset = MmapPreprocessedEEGDataset(patients=sanity_patients, max_samples_per_patient=16000)
    
    n_total = len(dataset)
    n_val = int(n_total * 0.1)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    
    train_sampler = BlockRandomSampler(train_ds, block_size=4096)
    train_loader = DataLoader(train_ds, batch_size=256, sampler=train_sampler, num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
    
    print(f"Subsample dataset ready: {n_train:,} training windows ({len(train_loader)} batches)\n", flush=True)
    
    # Run Candidate 1 (Base CNN + Protocol D)
    enc1, t1, vram1 = run_sanity_for_candidate("Candidate 1 (Base CNN + Protocol D)", PACSSLEncoder, train_loader, val_loader, device, epochs=2)
    
    # Run Candidate 2 (Learnable-Adjacency GCN + Protocol D)
    enc2, t2, vram2 = run_sanity_for_candidate("Candidate 2 (GCN + Protocol D)", PACSSLEncoderGCN, train_loader, val_loader, device, epochs=2)
    
    # Run Candidate 3 (Full DualKernelCNN + GCN + Transformer + Protocol D)
    enc3, t3, vram3 = run_sanity_for_candidate("Candidate 3 (Full DualKernelCNN+GCN+Transformer + Protocol D)", PACSSLEncoderFull, train_loader, val_loader, device, epochs=2)
    
    print("\n=====================================================================================", flush=True)
    print("=== Summary of Candidate Sanity Checks & Full Cost Estimates ===", flush=True)
    print("=====================================================================================", flush=True)
    print(f"{'Candidate':<52} | {'Time/Epoch (Sub)':<16} | {'Est. Time/Epoch (Full)':<24} | {'Peak VRAM':<12}", flush=True)
    print("-" * 110, flush=True)
    print(f"{'Candidate 1 (Base CNN + Protocol D)':<52} | {t1:<16.1f}s | ~{t1*4.8/60:.1f} min/ep (~{(t1*4.8/60)*10:.1f}m for 10ep) | {vram1:.1f} MB", flush=True)
    print(f"{'Candidate 2 (Learnable GCN + Protocol D)':<52} | {t2:<16.1f}s | ~{t2*4.8/60:.1f} min/ep (~{(t2*4.8/60)*10:.1f}m for 10ep) | {vram2:.1f} MB", flush=True)
    print(f"{'Candidate 3 (Full DualKernelCNN+GCN+Transformer)':<52} | {t3:<16.1f}s | ~{t3*4.8/60:.1f} min/ep (~{(t3*4.8/60)*10:.1f}m for 10ep) | {vram3:.1f} MB", flush=True)
    print("=====================================================================================\n", flush=True)

if __name__ == "__main__":
    main()
