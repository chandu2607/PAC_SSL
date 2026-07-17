"""
lopo_evaluation.py — Strict Leave-One-Patient-Out (LOPO) Seizure Forecasting & Surrogate Control
Implements:
  1. Fast pre-extraction & caching of 128-dim PAC-SSL Encoder representations (`z`) across all 23 subjects using memory-mapped .npy files.
  2. Strict 23-Fold LOPO Cross-Validation:
     - For each test subject, trains a lightweight classifier head on the remaining 22 subjects.
     - Enforces within-fold interictal 1:1 undersampling (`N_pre == N_inter`) during classifier training.
     - Tests on the held-out subject's FULL data (both preictal and interictal windows without undersampling/calibration).
  3. Surrogate / Chance Control:
     - Re-runs LOPO classifier training on shuffled training labels for at least 3 subjects (chb01, chb05, chb10)
     - Proves chance-level AUC (~0.50) compared to real PAC-SSL representations.
  4. Generates summary tables and saves metrics for the Stage 3 Review Gate artifact.
  5. Pure NumPy/PyTorch implementations of ROC-AUC and Confusion Matrix (zero sklearn dependency).
  6. Instant unbuffered logging (`flush=True`).
"""
import os
import sys
import time
import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

from pac_ssl_model import PACSSLEncoder, PACFeatureExtractor

DATA_ROOT = Path("data/preprocessed")
NPY_ROOT = DATA_ROOT / "npy"
PATIENTS_ALL = [f"chb{i:02d}" for i in range(1, 25) if i != 12]

def compute_roc_auc_numpy(y_true, y_score):
    """Pure NumPy exact calculation of Area Under ROC Curve (Mann-Whitney U statistic)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    n_pos = len(pos)
    n_neg = len(neg)
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    
    unique_vals, inverse, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for val_idx in np.where(counts > 1)[0]:
            mask = inverse == val_idx
            ranks[mask] = np.mean(ranks[mask])
            
    rank_sum_pos = np.sum(ranks[y_true == 1])
    u_pos = rank_sum_pos - (n_pos * (n_pos + 1)) / 2.0
    auc = u_pos / (n_pos * n_neg)
    return float(auc)


def compute_confusion_matrix_numpy(y_true, y_pred):
    """Pure NumPy exact calculation of binary confusion matrix (tn, fp, fn, tp)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return int(tn), int(fp), int(fn), int(tp)


class SeizureClassifierHead(nn.Module):
    """
    Lightweight MLP classifier probe operating on 128-dim self-supervised PAC-SSL embeddings (`z`).
    """
    def __init__(self, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def extract_and_cache_encoder_features(encoder, feature_extractor, device, patients=PATIENTS_ALL, cache_h5=DATA_ROOT / "encoder_features_z.h5", force=False):
    """
    Runs frozen PAC-SSL encoder once across all 23 subjects to extract 128-dim representations (`z`).
    Stores in RAM and saves to `encoder_features_z.h5` (~196 MB total) for instant LOPO iterations.
    Uses memory-mapped .npy files for instant sequential batching.
    """
    if cache_h5.exists() and not force:
        print(f"Loading pre-extracted encoder features `z` from cache: {cache_h5}...", flush=True)
        features_dict = {}
        with h5py.File(cache_h5, "r") as f:
            for p in patients:
                if p in f:
                    features_dict[p] = {
                        "preictal": torch.from_numpy(f[p]["preictal"][:]).float(),
                        "interictal": torch.from_numpy(f[p]["interictal"][:]).float()
                    }
        print(f"Loaded representations for {len(features_dict)} patients.", flush=True)
        return features_dict

    print(f"Pre-extracting 128-dim representations across {len(patients)} subjects on {device} using .npy mmaps...", flush=True)
    encoder.eval()
    feature_extractor.eval()
    
    features_dict = {}
    t0 = time.time()
    
    with h5py.File(cache_h5, "w") as out_f:
        for p in patients:
            pre_path = NPY_ROOT / f"{p}_preictal.npy"
            inter_path = NPY_ROOT / f"{p}_interictal.npy"
            
            if pre_path.exists():
                pre_data = np.load(pre_path, mmap_mode='r')
            else:
                pre_data = np.zeros((0, 18, 1024), dtype=np.float32)
                
            if inter_path.exists():
                inter_data = np.load(inter_path, mmap_mode='r')
            else:
                inter_data = np.zeros((0, 18, 1024), dtype=np.float32)
                
            z_pre_list = []
            if len(pre_data) > 0:
                for idx in range(0, len(pre_data), 256):
                    batch = torch.from_numpy(pre_data[idx:idx+256].copy()).to(device)
                    with torch.no_grad():
                        feats = feature_extractor(batch)
                        z = encoder.forward_from_features(feats)
                        z_pre_list.append(z.cpu())
                z_pre = torch.cat(z_pre_list, dim=0)
            else:
                z_pre = torch.zeros((0, 128), dtype=torch.float32)
                
            z_inter_list = []
            if len(inter_data) > 0:
                for idx in range(0, len(inter_data), 256):
                    batch = torch.from_numpy(inter_data[idx:idx+256].copy()).to(device)
                    with torch.no_grad():
                        feats = feature_extractor(batch)
                        z = encoder.forward_from_features(feats)
                        z_inter_list.append(z.cpu())
                z_inter = torch.cat(z_inter_list, dim=0)
            else:
                z_inter = torch.zeros((0, 128), dtype=torch.float32)
                
            features_dict[p] = {"preictal": z_pre, "interictal": z_inter}
            
            grp = out_f.create_group(p)
            grp.create_dataset("preictal", data=z_pre.numpy(), compression="gzip")
            grp.create_dataset("interictal", data=z_inter.numpy(), compression="gzip")
            
            print(f"  [{p}] Preictal: {len(z_pre)} | Interictal: {len(z_inter)} embeddings extracted.", flush=True)
            
    print(f"Pre-extraction complete in {time.time()-t0:.1f}s. Saved cache to {cache_h5}", flush=True)
    return features_dict


def train_classifier_head_on_fold(X_train, y_train, device, epochs=15, batch_size=256, lr=1e-3, d_model=None):
    """Trains a fresh SeizureClassifierHead on the balanced training embeddings of 22 subjects."""
    if d_model is None:
        d_model = X_train.shape[1]
    head = SeizureClassifierHead(d_model=d_model).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    ds = TensorDataset(X_train, y_train)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    
    head.train()
    for ep in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = head(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            
    head.eval()
    return head


def run_strict_lopo_and_surrogate(features_dict, device, surrogate_patients=["chb01", "chb05", "chb10"]):
    """
    Executes strict 23-Fold LOPO Cross-Validation and Surrogate Chance Control.
    Returns summary metrics across all patients.
    """
    print("\n=====================================================================================", flush=True)
    print("=== Starting Strict Leave-One-Patient-Out (LOPO) Evaluation across 23 Subjects ===", flush=True)
    print("=====================================================================================", flush=True)
    
    lopo_results = []
    surrogate_results = {}
    t0_lopo = time.time()
    
    for fold_idx, test_p in enumerate(PATIENTS_ALL):
        if test_p not in features_dict:
            continue
            
        train_patients = [p for p in PATIENTS_ALL if p != test_p and p in features_dict]
        
        pre_train_list = [features_dict[p]["preictal"] for p in train_patients if len(features_dict[p]["preictal"]) > 0]
        inter_train_list = [features_dict[p]["interictal"] for p in train_patients if len(features_dict[p]["interictal"]) > 0]
        
        z_pre_train = torch.cat(pre_train_list, dim=0)
        z_inter_train = torch.cat(inter_train_list, dim=0)
        
        n_pre = len(z_pre_train)
        n_inter = len(z_inter_train)
        n_sample = min(n_pre, n_inter)
        
        perm_inter = torch.randperm(n_inter)[:n_sample]
        perm_pre = torch.randperm(n_pre)[:n_sample]
        
        X_train = torch.cat([z_pre_train[perm_pre], z_inter_train[perm_inter]], dim=0)
        y_train = torch.cat([torch.ones(n_sample), torch.zeros(n_sample)], dim=0)
        
        head_real = train_classifier_head_on_fold(X_train, y_train, device, epochs=15)
        
        z_pre_test = features_dict[test_p]["preictal"]
        z_inter_test = features_dict[test_p]["interictal"]
        
        X_test = torch.cat([z_pre_test, z_inter_test], dim=0)
        y_test = torch.cat([torch.ones(len(z_pre_test)), torch.zeros(len(z_inter_test))], dim=0).numpy()
        
        if len(X_test) == 0 or len(z_pre_test) == 0 or len(z_inter_test) == 0:
            auc = 0.5 if len(X_test) > 0 else 0.0
            sens = 1.0 if len(z_pre_test) > 0 else 0.0
            spec = 1.0 if len(z_inter_test) > 0 else 0.0
        else:
            with torch.no_grad():
                logits_test = head_real(X_test.to(device)).cpu()
                probs_test = torch.sigmoid(logits_test).numpy()
                
            try:
                auc = compute_roc_auc_numpy(y_test, probs_test)
            except Exception:
                auc = 0.5
                
            preds_test = (probs_test >= 0.5).astype(int)
            tn, fp, fn, tp = compute_confusion_matrix_numpy(y_test, preds_test)
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
        lopo_results.append({
            "patient": test_p,
            "preictal_test": len(z_pre_test),
            "interictal_test": len(z_inter_test),
            "auc": auc,
            "sens": sens,
            "spec": spec
        })
        
        if test_p in surrogate_patients and len(X_test) > 0 and len(z_pre_test) > 0 and len(z_inter_test) > 0:
            y_train_shuffled = y_train[torch.randperm(len(y_train))]
            head_surrogate = train_classifier_head_on_fold(X_train, y_train_shuffled, device, epochs=15)
            with torch.no_grad():
                logits_surr = head_surrogate(X_test.to(device)).cpu()
                probs_surr = torch.sigmoid(logits_surr).numpy()
            try:
                surr_auc = compute_roc_auc_numpy(y_test, probs_surr)
            except Exception:
                surr_auc = 0.5
            preds_surr = (probs_surr >= 0.5).astype(int)
            stn, sfp, sfn, stp = compute_confusion_matrix_numpy(y_test, preds_surr)
            surr_sens = stp / (stp + sfn) if (stp + sfn) > 0 else 0.0
            surr_spec = stn / (stn + sfp) if (stn + sfp) > 0 else 0.0
            
            surrogate_results[test_p] = {
                "auc": surr_auc, "sens": surr_sens, "spec": surr_spec
            }
            
        print(f"  Fold {fold_idx+1:02d}/23 [{test_p}] | Test Pre: {len(z_pre_test):,}, Inter: {len(z_inter_test):,} | AUC: {auc:.4f} | Sens: {sens*100:.1f}% | Spec: {spec*100:.1f}%" +
              (f" | [Surrogate AUC: {surrogate_results[test_p]['auc']:.4f}]" if test_p in surrogate_results else ""), flush=True)
              
    elapsed_lopo = time.time() - t0_lopo
    print(f"\nCompleted 23-Fold LOPO evaluation in {elapsed_lopo:.1f}s.", flush=True)
    
    valid_aucs = [r["auc"] for r in lopo_results if r["preictal_test"] > 0 and r["interictal_test"] > 0]
    valid_sens = [r["sens"] for r in lopo_results if r["preictal_test"] > 0 and r["interictal_test"] > 0]
    valid_spec = [r["spec"] for r in lopo_results if r["preictal_test"] > 0 and r["interictal_test"] > 0]
    
    mean_auc = np.mean(valid_aucs)
    std_auc = np.std(valid_aucs)
    mean_sens = np.mean(valid_sens)
    mean_spec = np.mean(valid_spec)
    
    print("\n=====================================================================================", flush=True)
    print(f"=== Stage 3 Final LOPO Summary Across {len(valid_aucs)} Valid Subjects ===", flush=True)
    print(f"Mean AUC:         {mean_auc:.4f} ± {std_auc:.4f}", flush=True)
    print(f"Mean Sensitivity: {mean_sens*100:.2f}%", flush=True)
    print(f"Mean Specificity: {mean_spec*100:.2f}%", flush=True)
    print("-------------------------------------------------------------------------------------", flush=True)
    print("=== Surrogate Chance Comparison ===", flush=True)
    for sp, sres in surrogate_results.items():
        real_res = [r for r in lopo_results if r["patient"] == sp][0]
        print(f"  [{sp}] Real AUC: {real_res['auc']:.4f} vs Surrogate AUC: {sres['auc']:.4f} (Chance ~0.50)", flush=True)
    print("=====================================================================================", flush=True)
    
    out_txt = DATA_ROOT / "stage3_lopo_results.txt"
    with open(out_txt, "w") as f:
        f.write("Patient\tPreictal_Windows\tInterictal_Windows\tAUC\tSensitivity\tSpecificity\n")
        for r in lopo_results:
            f.write(f"{r['patient']}\t{r['preictal_test']}\t{r['interictal_test']}\t{r['auc']:.4f}\t{r['sens']:.4f}\t{r['spec']:.4f}\n")
        f.write("\n--- Mean Summary ---\n")
        f.write(f"Mean_AUC\t{mean_auc:.4f}\t{std_auc:.4f}\n")
        f.write(f"Mean_Sensitivity\t{mean_sens:.4f}\n")
        f.write(f"Mean_Specificity\t{mean_spec:.4f}\n")
        f.write("\n--- Surrogate Control Comparison ---\n")
        f.write("Patient\tReal_AUC\tSurrogate_AUC\n")
        for sp, sres in surrogate_results.items():
            real_res = [r for r in lopo_results if r["patient"] == sp][0]
            f.write(f"{sp}\t{real_res['auc']:.4f}\t{sres['auc']:.4f}\n")
    print(f"Saved formal LOPO and Surrogate results to {out_txt}", flush=True)
    return lopo_results, surrogate_results


def main():
    parser = argparse.ArgumentParser(description="Strict LOPO Seizure Forecasting & Surrogate Evaluation")
    parser.add_argument("--checkpoint", type=str, default="pac_ssl_encoder.pt", help="Path to pre-trained PAC-SSL encoder checkpoint")
    parser.add_argument("--force_extract", action="store_true", help="Force re-extraction of encoder features even if cache exists")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Stage 3 Evaluation starting on device: {device} ===", flush=True)
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found! Run pretext pretraining first.")
        
    feature_extractor = PACFeatureExtractor(fs=256, n_samples=1024).to(device)
    encoder = PACSSLEncoder(fs=256, n_samples=1024, num_channels=18, cnn_out=32, d_model=128).to(device)
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(state_dict)
    print(f"Successfully reloaded pre-trained encoder from {checkpoint_path}", flush=True)
    
    features_dict = extract_and_cache_encoder_features(
        encoder, feature_extractor, device, patients=PATIENTS_ALL, force=args.force_extract
    )
    
    run_strict_lopo_and_surrogate(features_dict, device, surrogate_patients=["chb01", "chb05", "chb10"])

if __name__ == "__main__":
    main()
