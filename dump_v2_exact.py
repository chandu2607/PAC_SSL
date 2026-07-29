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
from verify_balanced_70 import train_classifier_balanced

CACHE_V2 = Path("data/preprocessed/encoder_features_z_v2.h5")

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== INSTANT DUMP OF V2 STANDARD, V2 BALANCED RIDGE, V2 ROBUST MAD ===", flush=True)
    
    valid = [p for p in PATIENTS_ALL if p not in ["chb06", "chb08"]]
    
    all_pre_pos_rob = []
    all_pre_pos_loc = []
    patient_data = {}
    
    with h5py.File(CACHE_V2, "r") as f_v2:
        for p in valid:
            if p not in f_v2:
                continue
            z_v2_pre = torch.from_numpy(f_v2[p]["preictal"][:])
            z_v2_inter = torch.from_numpy(f_v2[p]["interictal"][:])
            try:
                pre_blocks, inter_blocks = get_patient_block_ids(p)
            except Exception:
                continue
            pre_arr = np.array(pre_blocks)
            inter_arr = np.array(inter_blocks)
            if len(set(pre_arr)) < 2 or len(set(inter_arr)) < 2:
                continue
                
            # Robust MAD norm on v2
            mu_rob = z_v2_inter.median(dim=0).values
            mad = (z_v2_inter - mu_rob).abs().median(dim=0).values * 1.4826
            sigma_rob = mad.clamp(min=1e-6)
            s_pre_rob, v_pre_rob, _ = compute_smoothed_velocity_features((z_v2_pre - mu_rob)/sigma_rob, pre_arr, window=4)
            s_inter_rob, v_inter_rob, _ = compute_smoothed_velocity_features((z_v2_inter - mu_rob)/sigma_rob, inter_arr, window=4)
            all_pre_pos_rob.append(s_pre_rob)
            
            # Local block 0 norm on v2
            cal_inter_0 = sorted(set(inter_arr))[0]
            mu_loc = z_v2_inter[inter_arr == cal_inter_0].mean(dim=0)
            sigma_loc = z_v2_inter[inter_arr == cal_inter_0].std(dim=0).clamp(min=1e-6)
            s_pre_loc, v_pre_loc, _ = compute_smoothed_velocity_features((z_v2_pre - mu_loc)/sigma_loc, pre_arr, window=4)
            s_inter_loc, v_inter_loc, _ = compute_smoothed_velocity_features((z_v2_inter - mu_loc)/sigma_loc, inter_arr, window=4)
            all_pre_pos_loc.append(s_pre_loc)
            
            patient_data[p] = {
                "pre_arr": pre_arr, "inter_arr": inter_arr, "cal_inter_0": cal_inter_0,
                "v2_rob_pre": torch.cat([s_pre_rob, v_pre_rob], dim=1), "v2_rob_inter": torch.cat([s_inter_rob, v_inter_rob], dim=1), "pos_rob": s_pre_rob,
                "v2_loc_pre": torch.cat([s_pre_loc, v_pre_loc], dim=1), "v2_loc_inter": torch.cat([s_inter_loc, v_inter_loc], dim=1), "pos_loc": s_pre_loc
            }
            
    pop_centroid_rob = torch.cat(all_pre_pos_rob, dim=0).mean(dim=0)
    pop_centroid_loc = torch.cat(all_pre_pos_loc, dim=0).mean(dim=0)
    
    print(f"{'Patient':<8} | {'v2 Standard (loc)':<18} | {'v2+BalRidge':<14} | {'v2 RobustMAD':<14}", flush=True)
    print("-" * 60, flush=True)
    
    cols = { "std": [], "bal": [], "rob": [] }
    
    for p in sorted(patient_data.keys()):
        d = patient_data[p]
        pre_arr = d["pre_arr"]
        inter_arr = d["inter_arr"]
        cal_inter_0 = d["cal_inter_0"]
        
        # 1. v2 Standard & 2. v2 Balanced Ridge (both use cal_pre_loc)
        cal_pre_loc = smart_calibration_block(d["pos_loc"], pre_arr, pop_centroid_loc)
        z_pre_cal = d["v2_loc_pre"][pre_arr == cal_pre_loc]
        z_inter_cal = d["v2_loc_inter"][inter_arr == cal_inter_0]
        z_pre_test = d["v2_loc_pre"][pre_arr != cal_pre_loc]
        z_inter_test = d["v2_loc_inter"][inter_arr != cal_inter_0]
        
        n_cal = min(len(z_pre_cal), len(z_inter_cal))
        X_cal_std = torch.cat([z_pre_cal[:n_cal], z_inter_cal[:n_cal]], dim=0)
        y_cal_std = torch.cat([torch.ones(n_cal), torch.zeros(n_cal)], dim=0)
        torch.manual_seed(42 + int(p[3:]))
        head_std = train_classifier_balanced(X_cal_std, y_cal_std, device, epochs=15, weight_decay=1e-4, use_balanced=False)
        X_te_loc = torch.cat([z_pre_test, z_inter_test], dim=0)
        y_te_loc = torch.cat([torch.ones(len(z_pre_test)), torch.zeros(len(z_inter_test))], dim=0).numpy()
        with torch.no_grad():
            probs_std = torch.sigmoid(head_std(X_te_loc.to(device))).view(-1).cpu().numpy()
        auc_std = compute_roc_auc_numpy(y_te_loc, probs_std)
        cols["std"].append(auc_std)
        
        X_cal_bal = torch.cat([z_pre_cal, z_inter_cal], dim=0)
        y_cal_bal = torch.cat([torch.ones(len(z_pre_cal)), torch.zeros(len(z_inter_cal))], dim=0)
        torch.manual_seed(42 + int(p[3:]))
        head_bal = train_classifier_balanced(X_cal_bal, y_cal_bal, device, epochs=15, weight_decay=1e-2, use_balanced=True)
        with torch.no_grad():
            probs_bal = torch.sigmoid(head_bal(X_te_loc.to(device))).view(-1).cpu().numpy()
        auc_bal = compute_roc_auc_numpy(y_te_loc, probs_bal)
        cols["bal"].append(auc_bal)
        
        # 3. v2 Robust MAD Norm (uses cal_pre_rob)
        cal_pre_rob = smart_calibration_block(d["pos_rob"], pre_arr, pop_centroid_rob)
        z_pre_cal_r = d["v2_rob_pre"][pre_arr == cal_pre_rob]
        z_inter_cal_r = d["v2_rob_inter"][inter_arr == cal_inter_0]
        z_pre_test_r = d["v2_rob_pre"][pre_arr != cal_pre_rob]
        z_inter_test_r = d["v2_rob_inter"][inter_arr != cal_inter_0]
        n_cal_r = min(len(z_pre_cal_r), len(z_inter_cal_r))
        X_cal_rob = torch.cat([z_pre_cal_r[:n_cal_r], z_inter_cal_r[:n_cal_r]], dim=0)
        y_cal_rob = torch.cat([torch.ones(n_cal_r), torch.zeros(n_cal_r)], dim=0)
        torch.manual_seed(42 + int(p[3:]))
        head_rob = train_classifier_head_on_fold(X_cal_rob, y_cal_rob, device, epochs=15)
        X_te_r = torch.cat([z_pre_test_r, z_inter_test_r], dim=0)
        y_te_r = torch.cat([torch.ones(len(z_pre_test_r)), torch.zeros(len(z_inter_test_r))], dim=0).numpy()
        with torch.no_grad():
            probs_rob = torch.sigmoid(head_rob(X_te_r.to(device))).view(-1).cpu().numpy()
        auc_rob = compute_roc_auc_numpy(y_te_r, probs_rob)
        cols["rob"].append(auc_rob)
        
        print(f"{p:<8} | {auc_std:<18.4f} | {auc_bal:<14.4f} | {auc_rob:<14.4f}", flush=True)
        
    print("-" * 60, flush=True)
    print(f"{'MEAN':<8} | {np.mean(cols['std']):<18.4f} | {np.mean(cols['bal']):<14.4f} | {np.mean(cols['rob']):<14.4f}", flush=True)

if __name__ == "__main__":
    main()
