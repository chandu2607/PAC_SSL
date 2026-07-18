"""
run_nonlinear_calibration_check.py — 2-Layer Non-Linear Calibration Probe Check
Evaluates whether replacing the 1-layer linear classifier head with a lightweight 2-layer MLP (128 -> 32 -> ReLU -> Dropout -> 1)
during patient-specific calibration crosses the >= 0.70 Mean AUC (70% accuracy) target across all 17 hold-out evaluated subjects
under exact 20-shuffle surrogate control.
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

CACHE_V2 = Path("data/preprocessed/encoder_features_z_v2.h5")

def train_mlp_classifier_on_fold(X_train, y_train, device, epochs=20, lr=1e-3, weight_decay=1e-3):
    d_in = X_train.shape[1]
    head = nn.Sequential(
        nn.Linear(d_in, 32),
        nn.BatchNorm1d(32) if len(X_train) > 1 else nn.Identity(),
        nn.ReLU(inplace=True),
        nn.Dropout(0.25),
        nn.Linear(32, 1)
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    
    ds = TensorDataset(X_train, y_train)
    batch_size = min(64, len(X_train)) if len(X_train) > 0 else 1
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    
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
    print(f"=== 2-Layer Non-Linear Calibration Probe Check starting on {device} ===", flush=True)
    t0 = time.time()
    
    with h5py.File(CACHE_V2, "r") as f:
        loaded_patients = list(f.keys())
    valid = [p for p in PATIENTS_ALL if p not in ["chb06", "chb08"] and p in loaded_patients]
    
    all_pre_pos_0 = []
    all_pre_pos_rob = []
    patient_objs = {}
    
    with h5py.File(CACHE_V2, "r") as f:
        for p in valid:
            z_pre = torch.from_numpy(f[p]["preictal"][:])
            z_inter = torch.from_numpy(f[p]["interictal"][:])
            try:
                pre_blocks, inter_blocks = get_patient_block_ids(p)
            except Exception:
                continue
            pre_arr = np.array(pre_blocks)
            inter_arr = np.array(inter_blocks)
            if len(set(pre_arr)) < 2 or len(set(inter_arr)) < 2:
                continue
                
            # 1. Earliest Block 0 norm
            cal_inter_0 = sorted(set(inter_arr))[0]
            mu_0 = z_inter[inter_arr == cal_inter_0].mean(dim=0)
            sigma_0 = z_inter[inter_arr == cal_inter_0].std(dim=0).clamp(min=1e-6)
            s_pre_0, v_pre_0, _ = compute_smoothed_velocity_features((z_pre - mu_0)/sigma_0, pre_arr, window=4)
            s_inter_0, v_inter_0, _ = compute_smoothed_velocity_features((z_inter - mu_0)/sigma_0, inter_arr, window=4)
            all_pre_pos_0.append(s_pre_0)
            
            # 2. Robust MAD Median norm across all interictal data
            mu_rob = z_inter.median(dim=0).values
            mad = (z_inter - mu_rob).abs().median(dim=0).values * 1.4826
            sigma_rob = mad.clamp(min=1e-6)
            s_pre_rob, v_pre_rob, _ = compute_smoothed_velocity_features((z_pre - mu_rob)/sigma_rob, pre_arr, window=4)
            s_inter_rob, v_inter_rob, _ = compute_smoothed_velocity_features((z_inter - mu_rob)/sigma_rob, inter_arr, window=4)
            all_pre_pos_rob.append(s_pre_rob)
            
            patient_objs[p] = {
                "posvel_pre_0": torch.cat([s_pre_0, v_pre_0], dim=1),
                "posvel_inter_0": torch.cat([s_inter_0, v_inter_0], dim=1),
                "pos_pre_0": s_pre_0,
                "posvel_pre_rob": torch.cat([s_pre_rob, v_pre_rob], dim=1),
                "posvel_inter_rob": torch.cat([s_inter_rob, v_inter_rob], dim=1),
                "pos_pre_rob": s_pre_rob,
                "pre_arr": pre_arr, "inter_arr": inter_arr, "cal_inter_0": cal_inter_0
            }
            
    pop_centroid_0 = torch.cat(all_pre_pos_0, dim=0).mean(dim=0)
    pop_centroid_rob = torch.cat(all_pre_pos_rob, dim=0).mean(dim=0)
    
    print("\n====================================================================================================================", flush=True)
    print(f"{'Patient':<8} | {'Mode':<12} | {'Non-Linear MLP AUC':<20} | {'Surrogate (N=20)':<24} | {'p-val':<10} | {'Pass?':<6}", flush=True)
    print("====================================================================================================================", flush=True)
    
    results_0 = []
    results_rob = []
    n_shuffles = 20
    
    for p, d in patient_objs.items():
        pre_arr = d["pre_arr"]
        inter_arr = d["inter_arr"]
        
        # --- Evaluate Mode 1: Earliest Block 0 ---
        cal_pre_0 = smart_calibration_block(d["pos_pre_0"], pre_arr, pop_centroid_0)
        cal_inter_0 = d["cal_inter_0"]
        
        z_pre_cal = d["posvel_pre_0"][pre_arr == cal_pre_0]
        z_inter_cal = d["posvel_inter_0"][inter_arr == cal_inter_0]
        z_pre_test = d["posvel_pre_0"][pre_arr != cal_pre_0]
        z_inter_test = d["posvel_inter_0"][inter_arr != cal_inter_0]
        
        n_cal = min(len(z_pre_cal), len(z_inter_cal))
        X_cal = torch.cat([z_pre_cal[:n_cal], z_inter_cal[:n_cal]], dim=0)
        y_cal = torch.cat([torch.ones(n_cal), torch.zeros(n_cal)], dim=0)
        
        torch.manual_seed(42 + int(p[3:]))
        head_0 = train_mlp_classifier_on_fold(X_cal, y_cal, device, epochs=20)
        
        X_te = torch.cat([z_pre_test, z_inter_test], dim=0)
        y_te = torch.cat([torch.ones(len(z_pre_test)), torch.zeros(len(z_inter_test))], dim=0).numpy()
        
        with torch.no_grad():
            probs_0 = torch.sigmoid(head_0(X_te.to(device))).view(-1).cpu().numpy()
        real_auc_0 = compute_roc_auc_numpy(y_te, probs_0)
        
        surr_aucs_0 = []
        for s in range(n_shuffles):
            torch.manual_seed(1000 + s*100 + int(p[3:]))
            y_perm = y_cal[torch.randperm(len(y_cal))]
            head_perm = train_mlp_classifier_on_fold(X_cal, y_perm, device, epochs=20)
            with torch.no_grad():
                probs_perm = torch.sigmoid(head_perm(X_te.to(device))).view(-1).cpu().numpy()
            surr_aucs_0.append(compute_roc_auc_numpy(y_te, probs_perm))
            
        p_val_0 = (sum(1 for sa in surr_aucs_0 if sa >= real_auc_0) + 1.0) / (n_shuffles + 1.0)
        pass_0 = p_val_0 <= 0.05
        results_0.append({"patient": p, "auc": real_auc_0, "p_val": p_val_0, "pass": pass_0})
        print(f"{p:<8} | {'Earliest 0':<12} | {real_auc_0:<20.4f} | {np.mean(surr_aucs_0):.4f} ± {np.std(surr_aucs_0):.4f}   | {p_val_0:<10.4f} | {'PASS' if pass_0 else 'FAIL':<6}", flush=True)
        
        # --- Evaluate Mode 2: Robust MAD Median ---
        cal_pre_rob = smart_calibration_block(d["pos_pre_rob"], pre_arr, pop_centroid_rob)
        
        z_pre_cal = d["posvel_pre_rob"][pre_arr == cal_pre_rob]
        z_inter_cal = d["posvel_inter_rob"][inter_arr == cal_inter_0] # Use same cal block for interictal
        z_pre_test = d["posvel_pre_rob"][pre_arr != cal_pre_rob]
        z_inter_test = d["posvel_inter_rob"][inter_arr != cal_inter_0]
        
        n_cal_rob = min(len(z_pre_cal), len(z_inter_cal))
        X_cal_rob = torch.cat([z_pre_cal[:n_cal_rob], z_inter_cal[:n_cal_rob]], dim=0)
        y_cal_rob = torch.cat([torch.ones(n_cal_rob), torch.zeros(n_cal_rob)], dim=0)
        
        torch.manual_seed(42 + int(p[3:]))
        head_rob = train_mlp_classifier_on_fold(X_cal_rob, y_cal_rob, device, epochs=20)
        
        with torch.no_grad():
            probs_rob = torch.sigmoid(head_rob(X_te.to(device))).view(-1).cpu().numpy()
        real_auc_rob = compute_roc_auc_numpy(y_te, probs_rob)
        
        surr_aucs_rob = []
        for s in range(n_shuffles):
            torch.manual_seed(2000 + s*100 + int(p[3:]))
            y_perm = y_cal_rob[torch.randperm(len(y_cal_rob))]
            head_perm = train_mlp_classifier_on_fold(X_cal_rob, y_perm, device, epochs=20)
            with torch.no_grad():
                probs_perm = torch.sigmoid(head_perm(X_te.to(device))).view(-1).cpu().numpy()
            surr_aucs_rob.append(compute_roc_auc_numpy(y_te, probs_perm))
            
        p_val_rob = (sum(1 for sa in surr_aucs_rob if sa >= real_auc_rob) + 1.0) / (n_shuffles + 1.0)
        pass_rob = p_val_rob <= 0.05
        results_rob.append({"patient": p, "auc": real_auc_rob, "p_val": p_val_rob, "pass": pass_rob})
        print(f"{p:<8} | {'Robust MAD':<12} | {real_auc_rob:<20.4f} | {np.mean(surr_aucs_rob):.4f} ± {np.std(surr_aucs_rob):.4f}   | {p_val_rob:<10.4f} | {'PASS' if pass_rob else 'FAIL':<6}", flush=True)
        print("-" * 116, flush=True)
        
    mean_0 = np.mean([r["auc"] for r in results_0])
    pass_cnt_0 = sum(1 for r in results_0 if r["pass"])
    mean_rob = np.mean([r["auc"] for r in results_rob])
    pass_cnt_rob = sum(1 for r in results_rob if r["pass"])
    
    print("\n====================================================================================================================", flush=True)
    print("=== FINAL NON-LINEAR CALIBRATION PROBE SUMMARY (N=17 Valid Evaluated Patients) ===", flush=True)
    print(f"  Mode 1 (Earliest Block 0 Norm + MLP) -> Mean Pos+Vel AUC: {mean_0:.4f} | Passing Subjects (p<=0.05): {pass_cnt_0} / 17", flush=True)
    print(f"  Mode 2 (Robust MAD Median Norm + MLP) -> Mean Pos+Vel AUC: {mean_rob:.4f} | Passing Subjects (p<=0.05): {pass_cnt_rob} / 17", flush=True)
    print(f"Total time taken: {fmt_time(time.time() - t0)}", flush=True)
    print("====================================================================================================================\n", flush=True)

if __name__ == "__main__":
    main()
