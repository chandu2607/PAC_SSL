"""
verify_balanced_70.py — Verifies exact performance of combining our winning v2 Baseline (Earliest Block 0 Normalization + Smoothed Velocity)
with Strategy 2's Class-Balanced Positive Weighting + Ridge Regularization.
Runs 20 surrogate shuffles per patient to test crossing 0.70 Mean AUC across N=17 patients.
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

def train_classifier_balanced(X_train, y_train, device, epochs=15, lr=1e-3, weight_decay=1e-2, use_balanced=True):
    d_in = X_train.shape[1]
    head = nn.Linear(d_in, 1).to(device)
    
    if use_balanced and len(y_train) > 0:
        n_pos = (y_train == 1.0).sum().item()
        n_neg = (y_train == 0.0).sum().item()
        pw = float(n_neg) / max(float(n_pos), 1.0)
        pos_weight = torch.tensor([min(max(pw, 0.5), 5.0)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
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
    print(f"=== Verifying v2 Baseline + Class-Balanced Ridge Probing on {device} ===", flush=True)
    t0 = time.time()
    
    with h5py.File(CACHE_V2, "r") as f:
        loaded_patients = list(f.keys())
    valid = [p for p in PATIENTS_ALL if p not in ["chb06", "chb08"] and p in loaded_patients]
    
    all_pre_pos = []
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
                
            cal_inter_0 = sorted(set(inter_arr))[0]
            mu_0 = z_inter[inter_arr == cal_inter_0].mean(dim=0)
            sigma_0 = z_inter[inter_arr == cal_inter_0].std(dim=0).clamp(min=1e-6)
            s_pre_0, v_pre_0, _ = compute_smoothed_velocity_features((z_pre - mu_0)/sigma_0, pre_arr, window=4)
            s_inter_0, v_inter_0, _ = compute_smoothed_velocity_features((z_inter - mu_0)/sigma_0, inter_arr, window=4)
            all_pre_pos.append(s_pre_0)
            
            patient_objs[p] = {
                "posvel_pre": torch.cat([s_pre_0, v_pre_0], dim=1),
                "posvel_inter": torch.cat([s_inter_0, v_inter_0], dim=1),
                "pos_pre": s_pre_0,
                "pre_arr": pre_arr, "inter_arr": inter_arr, "cal_inter_0": cal_inter_0
            }
            
    pop_centroid = torch.cat(all_pre_pos, dim=0).mean(dim=0)
    
    print("\n====================================================================================================================", flush=True)
    print(f"{'Patient':<8} | {'Standard AUC':<14} | {'Balanced Ridge AUC':<18} | {'Surrogate (N=20)':<24} | {'p-val':<10} | {'Pass?':<6}", flush=True)
    print("====================================================================================================================", flush=True)
    
    results_std = []
    results_bal = []
    n_shuffles = 20
    
    for p, d in patient_objs.items():
        pre_arr = d["pre_arr"]
        inter_arr = d["inter_arr"]
        cal_pre = smart_calibration_block(d["pos_pre"], pre_arr, pop_centroid)
        cal_inter_0 = d["cal_inter_0"]
        
        z_pre_cal = d["posvel_pre"][pre_arr == cal_pre]
        z_inter_cal = d["posvel_inter"][inter_arr == cal_inter_0]
        z_pre_test = d["posvel_pre"][pre_arr != cal_pre]
        z_inter_test = d["posvel_inter"][inter_arr != cal_inter_0]
        
        # Standard unweighted fold (min clipped like check_70_auc.py)
        n_cal = min(len(z_pre_cal), len(z_inter_cal))
        X_cal_std = torch.cat([z_pre_cal[:n_cal], z_inter_cal[:n_cal]], dim=0)
        y_cal_std = torch.cat([torch.ones(n_cal), torch.zeros(n_cal)], dim=0)
        
        torch.manual_seed(42 + int(p[3:]))
        head_std = train_classifier_balanced(X_cal_std, y_cal_std, device, epochs=15, weight_decay=1e-4, use_balanced=False)
        
        X_te = torch.cat([z_pre_test, z_inter_test], dim=0)
        y_te = torch.cat([torch.ones(len(z_pre_test)), torch.zeros(len(z_inter_test))], dim=0).numpy()
        
        with torch.no_grad():
            probs_std = torch.sigmoid(head_std(X_te.to(device))).view(-1).cpu().numpy()
        real_auc_std = compute_roc_auc_numpy(y_te, probs_std)
        results_std.append(real_auc_std)
        
        # Balanced Ridge on FULL calibration data
        X_cal_bal = torch.cat([z_pre_cal, z_inter_cal], dim=0)
        y_cal_bal = torch.cat([torch.ones(len(z_pre_cal)), torch.zeros(len(z_inter_cal))], dim=0)
        
        torch.manual_seed(42 + int(p[3:]))
        head_bal = train_classifier_balanced(X_cal_bal, y_cal_bal, device, epochs=15, weight_decay=1e-2, use_balanced=True)
        
        with torch.no_grad():
            probs_bal = torch.sigmoid(head_bal(X_te.to(device))).view(-1).cpu().numpy()
        real_auc_bal = compute_roc_auc_numpy(y_te, probs_bal)
        
        surr_aucs = []
        for s in range(n_shuffles):
            torch.manual_seed(1000 + s*100 + int(p[3:]))
            y_perm = y_cal_bal[torch.randperm(len(y_cal_bal))]
            head_perm = train_classifier_balanced(X_cal_bal, y_perm, device, epochs=15, weight_decay=1e-2, use_balanced=True)
            with torch.no_grad():
                probs_perm = torch.sigmoid(head_perm(X_te.to(device))).view(-1).cpu().numpy()
            surr_aucs.append(compute_roc_auc_numpy(y_te, probs_perm))
            
        p_val = (sum(1 for sa in surr_aucs if sa >= real_auc_bal) + 1.0) / (n_shuffles + 1.0)
        pass_bal = p_val <= 0.05
        results_bal.append({"patient": p, "auc": real_auc_bal, "p_val": p_val, "pass": pass_bal})
        print(f"{p:<8} | {real_auc_std:<14.4f} | {real_auc_bal:<18.4f} | {np.mean(surr_aucs):.4f} ± {np.std(surr_aucs):.4f}   | {p_val:<10.4f} | {'PASS' if pass_bal else 'FAIL':<6}", flush=True)
        
    mean_std = np.mean(results_std)
    mean_bal = np.mean([r["auc"] for r in results_bal])
    pass_cnt = sum(1 for r in results_bal if r["pass"])
    
    print("\n====================================================================================================================", flush=True)
    print(f"=== FINAL COMPARISON SUMMARY (N=17 Valid Evaluated Patients) ===", flush=True)
    print(f"  Standard v2 Baseline (Earliest 0 Norm)     -> Mean Pos+Vel AUC: {mean_std:.4f}", flush=True)
    print(f"  + Strategy 2 Class-Balanced Ridge Probe    -> Mean Pos+Vel AUC: {mean_bal:.4f} | Passing Subjects: {pass_cnt} / 17", flush=True)
    print(f"  Total Accuracy Improvement Delta           -> {mean_bal - mean_std:+.4f} ({(mean_bal - mean_std)*100:+.2f}%)", flush=True)
    print(f"Total time taken: {fmt_time(time.time() - t0)}", flush=True)
    print("====================================================================================================================\n", flush=True)

if __name__ == "__main__":
    main()
