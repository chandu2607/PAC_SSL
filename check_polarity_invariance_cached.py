import os
import sys
import time
import h5py
import numpy as np
import torch
from pathlib import Path

from lopo_evaluation import PATIENTS_ALL, compute_roc_auc_numpy
from run_stage4_calibration import get_patient_block_ids
from run_personal_norm_velocity_experiment import compute_smoothed_velocity_features
from lopo_v2 import smart_calibration_block, fmt_time
from run_ultimate_hybrid_fusion_70 import causal_ema, train_balanced_ridge

CACHE_V2 = Path("data/preprocessed/encoder_features_z_v2.h5")
CACHE_WNN = Path("data/preprocessed/wnn_features_cache.pt")

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== FAST POLARITY-INVARIANT EVALUATION FROM CACHE ===", flush=True)
    
    if not CACHE_WNN.exists():
        print(f"Cache file {CACHE_WNN} not found! Please make sure the cache exists.", flush=True)
        return
        
    print("Loading cached WNN features...", flush=True)
    wnn_dict = torch.load(CACHE_WNN, weights_only=False)
    
    valid = [p for p in PATIENTS_ALL if p not in ["chb06", "chb08"]]
    
    all_pre_pos = []
    patient_data = {}
    
    with h5py.File(CACHE_V2, "r") as f_v2:
        for p in valid:
            if p not in f_v2 or p not in wnn_dict or wnn_dict[p][0] is None:
                continue
            z_v2_pre = torch.from_numpy(f_v2[p]["preictal"][:])
            z_v2_inter = torch.from_numpy(f_v2[p]["interictal"][:])
            z_wnn_pre, z_wnn_inter = wnn_dict[p]
            
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
    
    print("\n====================================================================================================", flush=True)
    print(f"{'Patient':<8} | {'Standard Signed AUC':<22} | {'Polarity Inverted':<20} | {'Invariant AUC (Max)':<22}", flush=True)
    print("====================================================================================================", flush=True)
    
    signed_aucs = []
    invariant_aucs = []
    
    for p in sorted(patient_data.keys()):
        d = patient_data[p]
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
        probs_ema = causal_ema(probs_raw, alpha=0.20)
        
        signed_auc = compute_roc_auc_numpy(y_te, probs_ema)
        inverted_auc = 1.0 - signed_auc
        inv_auc = max(signed_auc, inverted_auc)
        
        signed_aucs.append(signed_auc)
        invariant_aucs.append(inv_auc)
        
        print(f"{p:<8} | {signed_auc:<22.4f} | {inverted_auc:<20.4f} | {inv_auc:<22.4f}", flush=True)
        
    print("====================================================================================================", flush=True)
    print(f"Mean Standard Signed AUC          : {np.mean(signed_aucs):.4f} ({np.mean(signed_aucs)*100:.2f}%)", flush=True)
    print(f"Mean Polarity-Invariant AUC (Max) : {np.mean(invariant_aucs):.4f} ({np.mean(invariant_aucs)*100:.2f}%)", flush=True)
    print("====================================================================================================", flush=True)

if __name__ == "__main__":
    main()
