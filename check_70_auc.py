"""
check_70_auc.py — Testing whether Robust MAD Baseline Normalization + Smoothed Velocity Dynamics
crosses the 0.70 Mean AUC (70% Accuracy) target across all valid patients under exact 20-shuffle surrogate control.
"""
import os
import sys
import h5py
import numpy as np
import torch
from pathlib import Path

from lopo_evaluation import PATIENTS_ALL, compute_roc_auc_numpy, train_classifier_head_on_fold
from run_stage4_calibration import get_patient_block_ids
from run_personal_norm_velocity_experiment import compute_smoothed_velocity_features
from lopo_v2 import smart_calibration_block

CACHE_V2 = Path("data/preprocessed/encoder_features_z_v2.h5")

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Testing Target: Crossing 0.70 Mean AUC (70% Accuracy) on {device} ===", flush=True)
    
    with h5py.File(CACHE_V2, "r") as f:
        loaded_patients = list(f.keys())
    valid = [p for p in PATIENTS_ALL if p not in ["chb06", "chb08"] and p in loaded_patients]
    
    # Precompute population centroid under Robust MAD Normalization + Velocity
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
                
            # Robust MAD median across all interictal data (removes long-term interictal drift across days)
            mu_rob = z_inter.median(dim=0).values
            mad = (z_inter - mu_rob).abs().median(dim=0).values * 1.4826
            sigma_rob = mad.clamp(min=1e-6)
            
            z_pre_norm = (z_pre - mu_rob) / sigma_rob
            z_inter_norm = (z_inter - mu_rob) / sigma_rob
            
            # Compute smoothed velocity features
            s_pre, v_pre, _ = compute_smoothed_velocity_features(z_pre_norm, pre_arr, window=4)
            s_inter, v_inter, _ = compute_smoothed_velocity_features(z_inter_norm, inter_arr, window=4)
            
            posvel_pre = torch.cat([s_pre, v_pre], dim=1)
            posvel_inter = torch.cat([s_inter, v_inter], dim=1)
            
            all_pre_pos.append(s_pre)
            patient_objs[p] = {
                "posvel_pre": posvel_pre, "posvel_inter": posvel_inter, "pos_pre": s_pre,
                "pre_arr": pre_arr, "inter_arr": inter_arr
            }
            
    pop_centroid = torch.cat(all_pre_pos, dim=0).mean(dim=0)
    
    print("\n--------------------------------------------------------------------------------------------------------------", flush=True)
    print(f"{'Patient':<8} | {'Robust MAD Pos+Vel AUC':<22} | {'Surrogate Mean±Std (N=20)':<26} | {'Empirical p-val':<16} | {'Pass?':<6}", flush=True)
    print("--------------------------------------------------------------------------------------------------------------", flush=True)
    
    results = []
    n_shuffles = 20
    
    for p, d in patient_objs.items():
        pre_arr = d["pre_arr"]
        inter_arr = d["inter_arr"]
        posvel_pre = d["posvel_pre"]
        posvel_inter = d["posvel_inter"]
        
        cal_pre = smart_calibration_block(d["pos_pre"], pre_arr, pop_centroid)
        cal_inter = sorted(set(inter_arr))[0] # Earliest interictal block or closest match
        
        z_pre_cal = posvel_pre[pre_arr == cal_pre]
        z_inter_cal = posvel_inter[inter_arr == cal_inter]
        z_pre_test = posvel_pre[pre_arr != cal_pre]
        z_inter_test = posvel_inter[inter_arr != cal_inter]
        
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
        
        # 20 Surrogate shuffles
        surr_aucs = []
        for s in range(n_shuffles):
            torch.manual_seed(2000 + s*100 + int(p[3:]))
            y_perm = y_cal[torch.randperm(len(y_cal))]
            head_perm = train_classifier_head_on_fold(X_cal, y_perm, device, epochs=15)
            with torch.no_grad():
                probs_perm = torch.sigmoid(head_perm(X_te.to(device))).cpu().numpy()
            surr_aucs.append(compute_roc_auc_numpy(y_te, probs_perm))
            
        p_val = (sum(1 for sa in surr_aucs if sa >= real_auc) + 1.0) / (n_shuffles + 1.0)
        passed = p_val <= 0.05
        results.append({"patient": p, "auc": real_auc, "p_val": p_val, "pass": passed})
        print(f"{p:<8} | {real_auc:<22.4f} | {np.mean(surr_aucs):.4f} ± {np.std(surr_aucs):.4f}       | {p_val:<16.4f} | {'PASS' if passed else 'FAIL':<6}", flush=True)
        
    mean_auc = np.mean([r["auc"] for r in results])
    n_pass = sum(1 for r in results if r["pass"])
    print("--------------------------------------------------------------------------------------------------------------", flush=True)
    print(f"MEAN across {len(results)} evaluated patients: {mean_auc:.4f} | Passing Patients (p<=0.05): {n_pass} / {len(results)}", flush=True)
    print("--------------------------------------------------------------------------------------------------------------\n", flush=True)

if __name__ == "__main__":
    main()
