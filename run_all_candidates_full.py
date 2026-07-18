"""
run_all_candidates_full.py — Full Pretraining & Evaluation for Candidate Architectures (Rules 1-5)
Selection Criterion (Stated Before Results): Mean Pos+Vel AUC across all valid evaluated patients (N=21).
Candidate Approaches:
  - Candidate 1: Base CNN + Protocol D fixes (Cosine LR warmup/decay, AdamW, balanced batches)
  - Candidate 2: Learnable-Adjacency GCN across C=18 channels + Protocol D fixes
  - Candidate 3 Lite: Dual-Kernel Frequency Decomposition CNN + GCN + Protocol D fixes
For each candidate:
  1. Trains across all 23 subjects for 10 epochs using memory-mapped .npy dataset.
  2. Extracts 128-dim representations across all subjects into cache.
  3. Evaluates with personal baseline normalization (unique_inter[0]) + smoothed velocity features (window=4) + smart calibration block.
  4. Executes 20-shuffle surrogate control for EVERY evaluated patient to verify empirical p < 0.05.
  5. Outputs complete summary tables to data/preprocessed/all_candidates_comparison.txt and markdown artifact.
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
    PACSSLEncoder
)
from train_pretext_stage2 import MmapPreprocessedEEGDataset, BlockRandomSampler, generate_pac_couplings_on_gpu, PATIENTS_ALL
from lopo_evaluation import compute_roc_auc_numpy, train_classifier_head_on_fold
from run_stage4_calibration import get_patient_block_ids
from run_personal_norm_velocity_experiment import compute_smoothed_velocity_features
from lopo_v2 import smart_calibration_block, fmt_time
from run_prestage4_item_b import load_features_dict

DATA_ROOT = Path("data/preprocessed")

# ==========================================
# Candidate 2: GCN Encoder
# ==========================================
class PACSSLEncoderGCN(nn.Module):
    def __init__(self, fs=256, n_samples=1024, num_channels=18, d_model=128, **kwargs):
        super().__init__()
        self.num_channels = num_channels
        self.d_model = d_model
        self.feature_extractor = PACFeatureExtractor(fs=fs, n_samples=n_samples)
        
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
        )
        self.gcn = LearnableAdjacencyGCN(num_channels=num_channels, embed_dim=16, feature_dim=32)
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
        B, C, F_in, T = features.shape
        x_flat = features.view(B * C, F_in, T)
        h_cnn = self.channel_cnn(x_flat)
        _, D, T_out = h_cnn.shape
        h_spatial = h_cnn.reshape(B, C, D, T_out)
        h_gcn = self.gcn(h_spatial)
        h_seq = h_gcn.reshape(B, C * D, T_out)
        h_proj = self.proj(h_seq)
        z = self.pool(h_proj).squeeze(-1)
        return z

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.forward_from_features(features)


# ==========================================
# Candidate 3 Lite: Dual-Kernel CNN + GCN
# ==========================================
class PACSSLEncoderDualKernelGCN(nn.Module):
    def __init__(self, fs=256, n_samples=1024, num_channels=18, d_model=128, **kwargs):
        super().__init__()
        self.num_channels = num_channels
        self.d_model = d_model
        self.feature_extractor = PACFeatureExtractor(fs=fs, n_samples=n_samples)
        
        self.front_end = DualKernelCNNFrontEnd(in_features=2, out_channels=32) # out (B, C, 32, T'=64)
        self.gcn = LearnableAdjacencyGCN(num_channels=num_channels, embed_dim=16, feature_dim=32)
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
        h_fe = self.front_end(features) # (B, C, 32, T'=64)
        B, C, D, T_out = h_fe.shape
        h_gcn = self.gcn(h_fe)          # (B, C, 32, 64)
        h_seq = h_gcn.reshape(B, C * D, T_out)
        h_proj = self.proj(h_seq)
        z = self.pool(h_proj).squeeze(-1)
        return z

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.forward_from_features(features)


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


def train_and_extract(name, encoder_cls, train_loader, val_loader, device, epochs=10, cache_h5=Path("test.h5")):
    if cache_h5.exists():
        print(f"\n[{name}] Cache already exists at {cache_h5}. Loading extracted representations directly...", flush=True)
        return load_features_dict(cache_h5)
        
    print(f"\n=====================================================================================", flush=True)
    print(f"=== Training & Extracting: {name} ({epochs} Epochs) ===", flush=True)
    print(f"=====================================================================================", flush=True)
    
    fe = PACFeatureExtractor(fs=256, n_samples=1024).to(device)
    encoder = encoder_cls(fs=256, n_samples=1024, num_channels=18, d_model=128).to(device)
    model = PACSSLPretextModelWrapper(encoder=encoder, d_model=128).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    t0_start = time.time()
    best_val_acc = 0.0
    best_state_dict = None
    
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * len(yc)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == yc).sum().item()
            total_samples += len(yc)
            
        train_acc = correct / total_samples * 100.0
        
        # Validation
        model.eval()
        val_loss, val_correct, val_samples = 0.0, 0, 0
        with torch.no_grad():
            for raw_x in val_loader:
                raw_x = raw_x.to(device, non_blocking=True)
                feats = fe(raw_x)
                xc, yc = generate_pac_couplings_on_gpu(feats, device)
                logits = model(xc)
                loss = criterion(logits, yc)
                val_loss += loss.item() * len(yc)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == yc).sum().item()
                val_samples += len(yc)
                
        val_acc = val_correct / val_samples * 100.0
        elapsed = time.time() - t_ep
        print(f"  [{name}] Epoch {ep:02d}/{epochs:02d} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Time: {elapsed:.1f}s", flush=True)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = {k: v.cpu() for k, v in encoder.state_dict().items()}
            
        scheduler.step()
        
    print(f"\n[{name}] Training done in {fmt_time(time.time()-t0_start)}. Peak Val Acc: {best_val_acc:.2f}%. Extracting representations...", flush=True)
    
    encoder.load_state_dict({k: v.to(device) for k, v in best_state_dict.items()})
    encoder.eval()
    fe.eval()
    
    from lopo_evaluation import NPY_ROOT
    features_dict = {}
    with h5py.File(cache_h5, "w") as out_f:
        for p in PATIENTS_ALL:
            pre_path = NPY_ROOT / f"{p}_preictal.npy"
            inter_path = NPY_ROOT / f"{p}_interictal.npy"
            pre_data = np.load(pre_path, mmap_mode='r') if pre_path.exists() else np.zeros((0,18,1024), dtype=np.float32)
            inter_data = np.load(inter_path, mmap_mode='r') if inter_path.exists() else np.zeros((0,18,1024), dtype=np.float32)

            def encode(data):
                zs = []
                for idx in range(0, len(data), 256):
                    batch = torch.from_numpy(data[idx:idx+256].copy()).to(device)
                    with torch.no_grad():
                        zs.append(encoder.forward_from_features(fe(batch)).cpu())
                return torch.cat(zs, dim=0) if zs else torch.zeros((0,128))

            z_pre = encode(pre_data)
            z_inter = encode(inter_data)
            grp = out_f.create_group(p)
            grp.create_dataset("preictal", data=z_pre.numpy(), compression="gzip")
            grp.create_dataset("interictal", data=z_inter.numpy(), compression="gzip")
            features_dict[p] = {"preictal": z_pre, "interictal": z_inter}
            
    print(f"[{name}] Saved representations to {cache_h5}", flush=True)
    return features_dict


def evaluate_candidate(name, features_dict, device, n_shuffles=20):
    print(f"\n=====================================================================================", flush=True)
    print(f"=== Evaluating: {name} (20 Surrogate Shuffles per Patient) ===", flush=True)
    print(f"=====================================================================================", flush=True)
    
    valid = [p for p in PATIENTS_ALL if p not in ["chb06", "chb08"] and p in features_dict]
    
    all_pre_pos = []
    patient_data = {}
    
    for p in valid:
        z_pre = features_dict[p]["preictal"]
        z_inter = features_dict[p]["interictal"]
        try:
            pre_blocks, inter_blocks = get_patient_block_ids(p)
        except FileNotFoundError:
            continue
        pre_arr = np.array(pre_blocks)
        inter_arr = np.array(inter_blocks)
        if len(pre_arr) == 0 or len(inter_arr) == 0:
            continue
            
        unique_inter = sorted(set(inter_arr))
        cal_inter = unique_inter[0]
        mu = z_inter[inter_arr == cal_inter].mean(dim=0)
        sigma = z_inter[inter_arr == cal_inter].std(dim=0).clamp(min=1e-6)
        
        z_pre_norm = (z_pre - mu) / sigma
        z_inter_norm = (z_inter - mu) / sigma
        
        s_pre, v_pre, _ = compute_smoothed_velocity_features(z_pre_norm, pre_arr, window=4)
        s_inter, v_inter, _ = compute_smoothed_velocity_features(z_inter_norm, inter_arr, window=4)
        
        all_pre_pos.append(s_pre)
        patient_data[p] = {
            "posvel_pre": torch.cat([s_pre, v_pre], dim=1),
            "posvel_inter": torch.cat([s_inter, v_inter], dim=1),
            "pos_pre": s_pre,
            "pre_arr": pre_arr,
            "inter_arr": inter_arr,
            "cal_inter": cal_inter
        }
        
    pop_centroid_pos = torch.cat(all_pre_pos, dim=0).mean(dim=0)
    
    results = []
    for p, d in patient_data.items():
        posvel_pre = d["posvel_pre"]
        posvel_inter = d["posvel_inter"]
        pre_arr = d["pre_arr"]
        inter_arr = d["inter_arr"]
        cal_inter = d["cal_inter"]
        
        cal_pre = smart_calibration_block(d["pos_pre"], pre_arr, pop_centroid_pos)
        
        z_pre_cal = posvel_pre[pre_arr == cal_pre]
        z_inter_cal = posvel_inter[inter_arr == cal_inter]
        z_pre_test = posvel_pre[pre_arr != cal_pre]
        z_inter_test = posvel_inter[inter_arr != cal_inter]
        
        if len(z_pre_cal) == 0 or len(z_inter_cal) == 0 or len(z_pre_test) == 0 or len(z_inter_test) == 0:
            continue
            
        n_cal = min(len(z_pre_cal), len(z_inter_cal))
        X_cal = torch.cat([z_pre_cal[:n_cal], z_inter_cal[:n_cal]], dim=0)
        y_cal = torch.cat([torch.ones(n_cal), torch.zeros(n_cal)], dim=0)
        
        torch.manual_seed(42 + int(p[3:]))
        head = train_classifier_head_on_fold(X_cal, y_cal, device, epochs=15)
        
        X_te = torch.cat([z_pre_test, z_inter_test], dim=0)
        y_te = torch.cat([torch.ones(len(z_pre_test)), torch.zeros(len(z_inter_test))], dim=0).numpy()
        
        with torch.no_grad():
            probs = torch.sigmoid(head(X_te.to(device))).cpu().numpy()
        real_auc = compute_roc_auc_numpy(y_te, probs)
        
        # Surrogate 20 shuffles
        surr_aucs = []
        for s in range(n_shuffles):
            torch.manual_seed(1000 + s*100 + int(p[3:]))
            y_perm = y_cal[torch.randperm(len(y_cal))]
            head_perm = train_classifier_head_on_fold(X_cal, y_perm, device, epochs=15)
            with torch.no_grad():
                probs_perm = torch.sigmoid(head_perm(X_te.to(device))).cpu().numpy()
            surr_aucs.append(compute_roc_auc_numpy(y_te, probs_perm))
            
        p_val = (sum(1 for sa in surr_aucs if sa >= real_auc) + 1.0) / (n_shuffles + 1.0)
        passed = p_val <= 0.05
        results.append({
            "patient": p, "auc": real_auc, "surr_mean": np.mean(surr_aucs),
            "surr_std": np.std(surr_aucs), "p_val": p_val, "pass": passed
        })
        print(f"  [{p:<8}] AUC: {real_auc:.4f} | Surr: {np.mean(surr_aucs):.4f} ± {np.std(surr_aucs):.4f} | p-val: {p_val:.4f} -> {'[PASS]' if passed else '[FAIL]'}", flush=True)
        
    mean_auc = np.mean([r["auc"] for r in results])
    n_pass = sum(1 for r in results if r["pass"])
    print(f"\n--- [{name} Summary] Mean Pos+Vel AUC across {len(results)} patients: {mean_auc:.4f} | Passing Patients (p<=0.05): {n_pass}/{len(results)} ---", flush=True)
    return mean_auc, n_pass, results


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Full Architecture Search & Evaluation starting on {device} ===", flush=True)
    
    dataset = MmapPreprocessedEEGDataset(patients=PATIENTS_ALL)
    n_total = len(dataset)
    n_val = int(n_total * 0.1)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    
    train_sampler = BlockRandomSampler(train_ds, block_size=8192)
    train_loader = DataLoader(train_ds, batch_size=256, sampler=train_sampler, num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
    
    summary_all = {}
    
    # 1. Candidate 1: Base CNN + Protocol D (already extracted to encoder_features_z_cand1.h5, will load instantly)
    fd1 = train_and_extract("Candidate 1 (Base CNN + Protocol D)", PACSSLEncoder, train_loader, val_loader, device, epochs=10, cache_h5=DATA_ROOT / "encoder_features_z_cand1.h5")
    m1, p1, r1 = evaluate_candidate("Candidate 1 (Base CNN + Protocol D)", fd1, device, n_shuffles=20)
    summary_all["Candidate 1"] = {"mean_auc": m1, "n_pass": p1, "results": r1}
    
    # 2. Candidate 3 Lite: Dual-Kernel CNN + GCN + Protocol D
    fd3 = train_and_extract("Candidate 3 Lite (DualKernelCNN + GCN + Protocol D)", PACSSLEncoderDualKernelGCN, train_loader, val_loader, device, epochs=10, cache_h5=DATA_ROOT / "encoder_features_z_cand3.h5")
    m3, p3, r3 = evaluate_candidate("Candidate 3 Lite (DualKernelCNN + GCN + Protocol D)", fd3, device, n_shuffles=20)
    summary_all["Candidate 3 Lite"] = {"mean_auc": m3, "n_pass": p3, "results": r3}
    
    # 3. Candidate 2: Learnable-Adjacency GCN + Protocol D
    fd2 = train_and_extract("Candidate 2 (Learnable GCN + Protocol D)", PACSSLEncoderGCN, train_loader, val_loader, device, epochs=10, cache_h5=DATA_ROOT / "encoder_features_z_cand2.h5")
    m2, p2, r2 = evaluate_candidate("Candidate 2 (Learnable GCN + Protocol D)", fd2, device, n_shuffles=20)
    summary_all["Candidate 2"] = {"mean_auc": m2, "n_pass": p2, "results": r2}
    
    print("\n=====================================================================================", flush=True)
    print("=== FINAL ARCHITECTURE SELECTION SUMMARY (Selection Criterion: Mean AUC across all valid patients) ===", flush=True)
    print("=====================================================================================", flush=True)
    for cname, s in summary_all.items():
        print(f"  {cname:<24} -> Mean AUC: {s['mean_auc']:.4f} | Passing Subjects (p<=0.05): {s['n_pass']} / {len(s['results'])}", flush=True)
    print("=====================================================================================\n", flush=True)
    
    out_txt = DATA_ROOT / "all_candidates_comparison.txt"
    with open(out_txt, "w") as f:
        for cname, s in summary_all.items():
            f.write(f"=== {cname} ===\nMean AUC: {s['mean_auc']:.4f} | Passing: {s['n_pass']}\n")
            f.write("Patient\tAUC\tSurrMean\tSurrStd\tp_val\tPass\n")
            for r in s["results"]:
                f.write(f"{r['patient']}\t{r['auc']:.4f}\t{r['surr_mean']:.4f}\t{r['surr_std']:.4f}\t{r['p_val']:.4f}\t{r['pass']}\n")
            f.write("\n")
    print(f"Saved full comparison table to {out_txt}", flush=True)

if __name__ == "__main__":
    main()
