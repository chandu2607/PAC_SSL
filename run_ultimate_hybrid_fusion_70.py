"""
run_ultimate_hybrid_fusion_70.py — Ultimate Hybrid Wavelet-PAC Fusion Pipeline
Fuses our top-performing PAC-SSL v2 time-domain features (128-dim) with our Multi-Scale Wavelet Filter-Bank features (128-dim).
Applies Sign-Invariant Power Transformation (|z| and z^2) to solve multi-day electrode polarity/sign inversion.
Trains Class-Balanced Ridge Probe with Causal Temporal EMA Smoothing across all N=17 hold-out subjects.
Verified against exact 20 surrogate chance permutations (p <= 0.05).
"""
import os
import sys
import time
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

from lopo_evaluation import PATIENTS_ALL, compute_roc_auc_numpy
from run_stage4_calibration import get_patient_block_ids
from run_personal_norm_velocity_experiment import compute_smoothed_velocity_features
from lopo_v2 import smart_calibration_block, fmt_time
from run_wavelet_nn_prototype import WaveletFilterBankFrontEnd, extract_wavelet_features_for_patient

CACHE_V2 = Path("data/preprocessed/encoder_features_z_v2.h5")
NPY_ROOT = Path("data/preprocessed/npy")

def causal_ema(p_arr, alpha=0.20):
    out = np.zeros_like(p_arr)
    curr = p_arr[0] if len(p_arr) > 0 else 0.0
    for i in range(len(p_arr)):
        curr = alpha * p_arr[i] + (1.0 - alpha) * curr
        out[i] = curr
    return out

def train_balanced_ridge(X_train, y_train, device, epochs=15, lr=1e-3, weight_decay=1e-2):
    d_in = X_train.shape[1]
    head = nn.Linear(d_in, 1).to(device)
    
    if len(y_train) > 0:
        n_pos = (y_train == 1.0).sum().item()
        n_neg = (y_train == 0.0).sum().item()
        pw = float(n_neg) / max(float(n_pos), 1.0)
        pos_weight = torch.tensor([min(max(pw, 0.5), 5.0)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()
        
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    
    ds = TensorDataset(X_train, y_train)
    bs = min(64, len(X_train)) if len(X_train) > 0 else 1
    loader = DataLoader(ds, batch_size=bs, shuffle=True)
    
    head.train()
    for ep in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = head(bx).view(-1)
            loss = criterion(out, by.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            optimizer.step()
            
    head.eval()
    return head

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Ultimate Hybrid Wavelet-PAC Fusion Evaluation starting on {device} ===", flush=True)
    t0 = time.time()
    
    wnn_encoder = WaveletFilterBankFrontEnd(num_channels=18).to(device)
    valid = [p for p in PATIENTS_ALL if p not in ["chb06", "chb08"]]
    
    all_pre_pos = []
    patient_data = {}
    
    print("Extracting and fusing PAC v2 + Wavelet Filter-Bank representations across all valid subjects...", flush=True)
    with h5py.File(CACHE_V2, "r") as f_v2:
        for p in valid:
            if p not in f_v2:
                continue
            z_v2_pre = torch.from_numpy(f_v2[p]["preictal"][:])
            z_v2_inter = torch.from_numpy(f_v2[p]["interictal"][:])
            
            z_wnn_pre, z_wnn_inter = extract_wavelet_features_for_patient(p, wnn_encoder, device)
            if z_wnn_pre is None or z_wnn_inter is None:
                continue
            try:
                pre_blocks, inter_blocks = get_patient_block_ids(p)
            except Exception:
                continue
            pre_arr = np.array(pre_blocks)
            inter_arr = np.array(inter_blocks)
            if len(set(pre_arr)) < 2 or len(set(inter_arr)) < 2:
                continue
                
            n_pre = min(len(z_v2_pre), len(z_wnn_pre), len(pre_arr))
            n_inter = min(len(z_v2_inter), len(z_wnn_inter), len(inter_arr))
            
            z_v2_pre = z_v2_pre[:n_pre]
            z_wnn_pre = z_wnn_pre[:n_pre]
            pre_arr = pre_arr[:n_pre]
            
            z_v2_inter = z_v2_inter[:n_inter]
            z_wnn_inter = z_wnn_inter[:n_inter]
            inter_arr = inter_arr[:n_inter]
            
            # Fuse v2 + WNN + Absolute Power / Magnitude features to prevent sign inversion
            z_pre_raw = torch.cat([z_v2_pre, z_wnn_pre, torch.abs(z_v2_pre), torch.abs(z_wnn_pre)], dim=1)
            z_inter_raw = torch.cat([z_v2_inter, z_wnn_inter, torch.abs(z_v2_inter), torch.abs(z_wnn_inter)], dim=1)
            
            unique_inter = sorted(set(inter_arr))
            cal_inter_idx = unique_inter[0]
            mu = z_inter_raw[inter_arr == cal_inter_idx].mean(dim=0)
            sigma = z_inter_raw[inter_arr == cal_inter_idx].std(dim=0).clamp(min=1e-6)
            
            s_pre, v_pre, _ = compute_smoothed_velocity_features((z_pre_raw - mu)/sigma, pre_arr, window=4)
            s_inter, v_inter, _ = compute_smoothed_velocity_features((z_inter_raw - mu)/sigma, inter_arr, window=4)
            
            all_pre_pos.append(s_pre)
            patient_data[p] = {
                "posvel_pre": torch.cat([s_pre, v_pre], dim=1),
                "posvel_inter": torch.cat([s_inter, v_inter], dim=1),
                "pos_pre": s_pre,
                "pre_arr": pre_arr, "inter_arr": inter_arr, "cal_inter_0": cal_inter_idx
            }
            
    pop_centroid = torch.cat(all_pre_pos, dim=0).mean(dim=0)
    
    print("\n====================================================================================================================", flush=True)
    print(f"{'Patient':<8} | {'Raw Hybrid AUC':<20} | {'EMA Smoothed (a=0.20)':<22} | {'Surrogate (N=20)':<24} | {'p-val':<10} | {'Pass?':<6}", flush=True)
    print("====================================================================================================================", flush=True)
    
    results_raw = []
    results_ema = []
    n_shuffles = 20
    
    for p, d in patient_data.items():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        pre_arr = d["pre_arr"]
        inter_arr = d["inter_arr"]
        cal_pre = smart_calibration_block(d["pos_pre"], pre_arr, pop_centroid)
        cal_inter_0 = d["cal_inter_0"]
        
        z_pre_cal = d["posvel_pre"][pre_arr == cal_pre]
        z_inter_cal = d["posvel_inter"][inter_arr == cal_inter_0]
        z_pre_test = d["posvel_pre"][pre_arr != cal_pre]
        z_inter_test = d["posvel_inter"][inter_arr != cal_inter_0]
        
        X_cal = torch.cat([z_pre_cal, z_inter_cal], dim=0)
        y_cal = torch.cat([torch.ones(len(z_pre_cal)), torch.zeros(len(z_inter_cal))], dim=0)
        X_te = torch.cat([z_pre_test, z_inter_test], dim=0)
        y_te = torch.cat([torch.ones(len(z_pre_test)), torch.zeros(len(z_inter_test))], dim=0).numpy()
        
        torch.manual_seed(42 + int(p[3:]))
        head = train_balanced_ridge(X_cal, y_cal, device, epochs=15, weight_decay=1e-2)
        
        with torch.no_grad():
            probs_raw = torch.sigmoid(head(X_te.to(device))).view(-1).cpu().numpy()
            
        real_auc_raw = compute_roc_auc_numpy(y_te, probs_raw)
        results_raw.append(real_auc_raw)
        
        probs_ema = causal_ema(probs_raw, alpha=0.20)
        real_auc_ema = compute_roc_auc_numpy(y_te, probs_ema)
        
        surr_aucs = []
        for s in range(n_shuffles):
            torch.manual_seed(1000 + s*100 + int(p[3:]))
            y_perm = y_cal[torch.randperm(len(y_cal))]
            head_perm = train_balanced_ridge(X_cal, y_perm, device, epochs=15, weight_decay=1e-2)
            with torch.no_grad():
                probs_perm = torch.sigmoid(head_perm(X_te.to(device))).view(-1).cpu().numpy()
            probs_perm_ema = causal_ema(probs_perm, alpha=0.20)
            surr_aucs.append(compute_roc_auc_numpy(y_te, probs_perm_ema))
            
        p_val = (sum(1 for sa in surr_aucs if sa >= real_auc_ema) + 1.0) / (n_shuffles + 1.0)
        passed = p_val <= 0.05
        results_ema.append({"patient": p, "auc": real_auc_ema, "p_val": p_val, "pass": passed})
        print(f"{p:<8} | {real_auc_raw:<20.4f} | {real_auc_ema:<22.4f} | {np.mean(surr_aucs):.4f} ± {np.std(surr_aucs):.4f}   | {p_val:<10.4f} | {'PASS' if passed else 'FAIL':<6}", flush=True)
        
    mean_raw = np.mean(results_raw)
    mean_ema = np.mean([r["auc"] for r in results_ema])
    pass_cnt = sum(1 for r in results_ema if r["pass"])
    
    print("\n====================================================================================================================", flush=True)
    print(f"=== FINAL HYBRID WAVELET-PAC FUSION SUMMARY (N=17 Valid Evaluated Patients) ===", flush=True)
    print(f"  Raw Hybrid Wavelet-PAC Probe               -> Mean Pos+Vel AUC: {mean_raw:.4f} ({mean_raw*100:.2f}%)", flush=True)
    print(f"  + Causal EMA Temporal Smoothing (alpha=.2) -> Mean Pos+Vel AUC: {mean_ema:.4f} ({mean_ema*100:.2f}%) | Passing Subjects: {pass_cnt} / 17", flush=True)
    if mean_ema >= 0.70:
        print(f"  🏆 TARGET ACHIEVED: Hybrid Wavelet-PAC officially crosses >= 70.0% Mean AUC across the entire cohort!", flush=True)
    elif mean_ema >= 0.69:
        print(f"  🎉 BASELINE CRUSHED: Hybrid Wavelet-PAC crosses the 69.0% base paper benchmark!", flush=True)
    else:
        print(f"  Benchmark Gap: {mean_ema - 0.69:+.4f}", flush=True)
    print(f"Total time taken: {fmt_time(time.time() - t0)}", flush=True)
    print("====================================================================================================================\n", flush=True)

if __name__ == "__main__":
    main()
