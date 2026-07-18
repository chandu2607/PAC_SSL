"""
analyze_direction_a.py — Deep Analysis of Direction A Responders vs Non-Responders
Investigates:
  1. Why chb11, chb15, chb23, chb24 were omitted or how smart_calibration_block handled them.
  2. Patient characteristics: Preictal vs Interictal feature variance, velocity drift magnitude, signal SNR across all 21 valid subjects.
  3. Tests whether robust quantile normalization or moving-window local interictal reference (instead of single earliest block unique_inter[0])
     generalizes the high AUC effect across the non-responder patients (e.g. chb01, chb02, chb04, chb05, chb21).
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

DATA_ROOT = Path("data/preprocessed")
CACHE_V2 = Path("data/preprocessed/encoder_features_z_v2.h5")

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    device = torch.device("cpu") # Pure CPU analysis, zero interference with background GPU task!
    print("=== Direction A Deep Analysis: Responders vs Non-Responders ===", flush=True)
    
    with h5py.File(CACHE_V2, "r") as f:
        loaded_patients = list(f.keys())
    print(f"Loaded subjects in cache: {len(loaded_patients)} subjects.", flush=True)
    
    valid = [p for p in PATIENTS_ALL if p not in ["chb06", "chb08"] and p in loaded_patients]
    print(f"Valid subjects: {valid}\n", flush=True)
    
    # 1. Investigate omitted patients or block counts
    print("--- 1. Preictal & Interictal Block Counts across All 21 Valid Subjects ---", flush=True)
    print(f"{'Patient':<8} | {'Preictal Blocks':<16} | {'Interictal Blocks':<18} | {'Pre Windows':<12} | {'Inter Windows':<14} | {'Status in Baseline':<20}", flush=True)
    print("-" * 95, flush=True)
    
    with h5py.File(CACHE_V2, "r") as f:
        for p in valid:
            z_pre = torch.from_numpy(f[p]["preictal"][:])
            z_inter = torch.from_numpy(f[p]["interictal"][:])
            try:
                pre_blocks, inter_blocks = get_patient_block_ids(p)
                n_pre_blk = len(set(pre_blocks))
                n_inter_blk = len(set(inter_blocks))
                status = "Evaluated"
                if n_pre_blk < 2 or n_inter_blk < 2:
                    status = f"Omitted (Pre:{n_pre_blk}, Int:{n_inter_blk})"
            except Exception as e:
                n_pre_blk, n_inter_blk = 0, 0
                status = f"Error: {e}"
                
            print(f"{p:<8} | {n_pre_blk:<16} | {n_inter_blk:<18} | {len(z_pre):<12} | {len(z_inter):<14} | {status:<20}", flush=True)
            
    # 2. Analyze feature variance, velocity magnitude, and interictal drift between Responders vs Non-Responders
    print("\n--- 2. Characteristics Comparison: High Responders vs Low/Non-Responders ---", flush=True)
    print(f"{'Patient':<8} | {'Group':<14} | {'Pre/Inter Var Ratio':<20} | {'Velocity Peak-to-Mean':<22} | {'Baseline Drift (mu_std)':<24}", flush=True)
    print("-" * 95, flush=True)
    
    high_responders = ["chb10", "chb16", "chb17", "chb19", "chb20", "chb22"]
    
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
                
            group = "High Responder" if p in high_responders else "Non-Responder"
            
            # Preictal / Interictal variance ratio
            var_ratio = (z_pre.var(dim=0).mean() / (z_inter.var(dim=0).mean() + 1e-6)).item()
            
            # Velocity peak-to-mean
            cal_inter = sorted(set(inter_arr))[0]
            mu = z_inter[inter_arr == cal_inter].mean(dim=0)
            sigma = z_inter[inter_arr == cal_inter].std(dim=0).clamp(min=1e-6)
            z_pre_norm = (z_pre - mu) / sigma
            _, v_pre, _ = compute_smoothed_velocity_features(z_pre_norm, pre_arr, window=4)
            vel_p2m = (v_pre.abs().max() / (v_pre.abs().mean() + 1e-6)).item()
            
            # Baseline drift across multiple interictal blocks
            unique_inter = sorted(set(inter_arr))
            inter_means = [z_inter[inter_arr == b].mean(dim=0) for b in unique_inter]
            drift = torch.stack(inter_means).std(dim=0).mean().item() if len(unique_inter) > 1 else 0.0
            
            print(f"{p:<8} | {group:<14} | {var_ratio:<20.3f} | {vel_p2m:<22.2f} | {drift:<24.4f}", flush=True)
            
    # 3. Experiment: Can Robust Multi-Block Reference (Median across ALL Interictal Blocks instead of earliest block 0)
    #    generalize signal and boost the Non-Responders?
    print("\n--- 3. Preprocessing Generalization Check: Robust Multi-Block Median Normalization vs Earliest Block 0 ---", flush=True)
    print(f"{'Patient':<8} | {'Earliest Block 0 AUC':<22} | {'Robust All-Interictal Median AUC':<32} | {'Delta':<10}", flush=True)
    print("-" * 80, flush=True)
    
    with h5py.File(CACHE_V2, "r") as f:
        # Build population centroid first
        all_pre_pos = []
        patient_objs = {}
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
                
            # Earliest block norm
            cal_inter = sorted(set(inter_arr))[0]
            mu_0 = z_inter[inter_arr == cal_inter].mean(dim=0)
            sigma_0 = z_inter[inter_arr == cal_inter].std(dim=0).clamp(min=1e-6)
            s_pre_0, v_pre_0, _ = compute_smoothed_velocity_features((z_pre - mu_0)/sigma_0, pre_arr, window=4)
            s_inter_0, v_inter_0, _ = compute_smoothed_velocity_features((z_inter - mu_0)/sigma_0, inter_arr, window=4)
            all_pre_pos.append(s_pre_0)
            
            # Robust median norm
            mu_rob = z_inter.median(dim=0).values
            # MAD std
            mad = (z_inter - mu_rob).abs().median(dim=0).values * 1.4826
            sigma_rob = mad.clamp(min=1e-6)
            s_pre_rob, v_pre_rob, _ = compute_smoothed_velocity_features((z_pre - mu_rob)/sigma_rob, pre_arr, window=4)
            s_inter_rob, v_inter_rob, _ = compute_smoothed_velocity_features((z_inter - mu_rob)/sigma_rob, inter_arr, window=4)
            
            patient_objs[p] = {
                "z0_pre": torch.cat([s_pre_0, v_pre_0], dim=1),
                "z0_inter": torch.cat([s_inter_0, v_inter_0], dim=1),
                "pos_pre": s_pre_0,
                "zrob_pre": torch.cat([s_pre_rob, v_pre_rob], dim=1),
                "zrob_inter": torch.cat([s_inter_rob, v_inter_rob], dim=1),
                "pre_arr": pre_arr, "inter_arr": inter_arr, "cal_inter": cal_inter
            }
            
        pop_centroid = torch.cat(all_pre_pos, dim=0).mean(dim=0)
        
        aucs_0 = []
        aucs_rob = []
        for p, d in patient_objs.items():
            pre_arr = d["pre_arr"]
            inter_arr = d["inter_arr"]
            cal_inter = d["cal_inter"]
            cal_pre = smart_calibration_block(d["pos_pre"], pre_arr, pop_centroid)
            
            # Eval 0
            n_cal = min((pre_arr == cal_pre).sum(), (inter_arr == cal_inter).sum())
            X_cal_0 = torch.cat([d["z0_pre"][pre_arr == cal_pre][:n_cal], d["z0_inter"][inter_arr == cal_inter][:n_cal]], dim=0)
            y_cal_0 = torch.cat([torch.ones(n_cal), torch.zeros(n_cal)], dim=0)
            torch.manual_seed(42 + int(p[3:]))
            head_0 = train_classifier_head_on_fold(X_cal_0, y_cal_0, device, epochs=15)
            X_te_0 = torch.cat([d["z0_pre"][pre_arr != cal_pre], d["z0_inter"][inter_arr != cal_inter]], dim=0)
            y_te = torch.cat([torch.ones((pre_arr != cal_pre).sum()), torch.zeros((inter_arr != cal_inter).sum())], dim=0).numpy()
            with torch.no_grad():
                probs_0 = torch.sigmoid(head_0(X_te_0)).numpy()
            auc_0 = compute_roc_auc_numpy(y_te, probs_0)
            aucs_0.append(auc_0)
            
            # Eval rob
            X_cal_rob = torch.cat([d["zrob_pre"][pre_arr == cal_pre][:n_cal], d["zrob_inter"][inter_arr == cal_inter][:n_cal]], dim=0)
            torch.manual_seed(42 + int(p[3:]))
            head_rob = train_classifier_head_on_fold(X_cal_rob, y_cal_0, device, epochs=15)
            X_te_rob = torch.cat([d["zrob_pre"][pre_arr != cal_pre], d["zrob_inter"][inter_arr != cal_inter]], dim=0)
            with torch.no_grad():
                probs_rob = torch.sigmoid(head_rob(X_te_rob)).numpy()
            auc_rob = compute_roc_auc_numpy(y_te, probs_rob)
            aucs_rob.append(auc_rob)
            
            delta = auc_rob - auc_0
            print(f"{p:<8} | {auc_0:<22.4f} | {auc_rob:<32.4f} | {delta:+.4f}", flush=True)
            
        print("-" * 80, flush=True)
        print(f"MEAN across {len(aucs_0)} patients: Earliest Block 0 = {np.mean(aucs_0):.4f} | Robust MAD Median = {np.mean(aucs_rob):.4f} | Delta = {np.mean(aucs_rob)-np.mean(aucs_0):+.4f}", flush=True)

if __name__ == "__main__":
    main()
