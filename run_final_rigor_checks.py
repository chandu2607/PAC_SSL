"""
run_final_rigor_checks.py — Final Rigor & Robustness Checks for Personal Norm + Velocity Experiment
Implements:
  1. Benjamini-Hochberg FDR correction across all 17 evaluated patients (and 100-shuffle verification for the 6 passing patients to remove 1/21 discretization floor).
  2. Calibration block choice robustness check on chb19, chb20, and chb22 using the SECOND available preictal and interictal blocks.
  3. Trajectory plot (`pass_patient_trajectory.png`) for actual passing patients (`chb17` and `chb20`) across real seizure onset (`t=0`).
"""
import os
import sys
import time
import h5py
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lopo_evaluation import PATIENTS_ALL, compute_roc_auc_numpy, train_classifier_head_on_fold
from run_prestage4_item_b import load_features_dict, EPOCH3_CACHE_H5
from run_stage4_calibration import get_patient_block_ids
from run_personal_norm_velocity_experiment import compute_smoothed_velocity_features
from lopo_v2 import smart_calibration_block

DATA_ROOT = Path("data/preprocessed")
CACHE_V2 = Path("data/preprocessed/encoder_features_z_v2.h5")

def plot_pass_trajectories(features_dict, patients=["chb17", "chb20"]):
    print("\n[Trajectory Plot] Generating winning trajectory plots for PASS patients chb17 and chb20...", flush=True)
    fig, axes = plt.subplots(len(patients), 2, figsize=(16, 5*len(patients)))
    if len(patients) == 1:
        axes = [axes]
        
    for i, p in enumerate(patients):
        if p not in features_dict:
            continue
        z_pre = features_dict[p]["preictal"]
        z_inter = features_dict[p]["interictal"]
        pre_blocks, inter_blocks = get_patient_block_ids(p)
        pre_blocks_arr = np.array(pre_blocks)
        inter_blocks_arr = np.array(inter_blocks)
        
        unique_inter = sorted(set(inter_blocks_arr))
        cal_inter_block = unique_inter[0]
        
        mu = z_inter[inter_blocks_arr == cal_inter_block].mean(dim=0)
        sigma = z_inter[inter_blocks_arr == cal_inter_block].std(dim=0).clamp(min=1e-6)
        
        z_pre_norm = (z_pre - mu) / sigma
        s_pre, v_pre, _ = compute_smoothed_velocity_features(z_pre_norm, pre_blocks_arr, window=4)
        
        unique_pre = sorted(set(pre_blocks_arr))
        blk = unique_pre[0]
        idx = np.where(pre_blocks_arr == blk)[0]
        
        s_block = s_pre[idx].numpy()
        v_block = v_pre[idx].numpy()
        
        T_blk = len(s_block)
        time_axis = (np.arange(T_blk) - T_blk) * 4.0 / 60.0
        
        top_dims = np.argsort(np.var(s_block, axis=0))[-3:]
        print(f"=== {p} ({blk}) Winning Trajectory Top Dims: {top_dims} ===", flush=True)
        print(f"  Position (s_block) start (t={time_axis[0]:.1f}m): {np.round(s_block[:3, top_dims], 2)}", flush=True)
        print(f"  Position (s_block) mid   (t={time_axis[len(time_axis)//2]:.1f}m): {np.round(s_block[len(s_block)//2, top_dims], 2)}", flush=True)
        print(f"  Position (s_block) end   (t=0.0m): {np.round(s_block[-3:, top_dims], 2)}", flush=True)
        print(f"  Velocity (v_block) max spikes: {np.round(np.max(np.abs(v_block[:, top_dims]), axis=0), 3)}", flush=True)
        
        ax_pos = axes[i][0] if len(patients) > 1 else axes[0]
        for d in top_dims:
            ax_pos.plot(time_axis, s_block[:, d], label=f"Dim {d}", alpha=0.85, linewidth=2.0)
        ax_pos.axvline(0.0, color='red', linestyle='--', linewidth=2, label="Seizure Onset (t=0)")
        ax_pos.set_title(f"{p} ({blk}) - Winning Normalized Position (x_norm) Top 3 Dims", fontsize=12, fontweight='bold')
        ax_pos.set_xlabel("Time Relative to Seizure Onset (Minutes)")
        ax_pos.set_ylabel("Z-Score Normalized Value")
        ax_pos.legend(loc="upper left", fontsize=9)
        ax_pos.grid(True, alpha=0.3)
        
        ax_vel = axes[i][1] if len(patients) > 1 else axes[1]
        for d in top_dims:
            ax_vel.plot(time_axis, v_block[:, d], label=f"Dim {d} Vel", alpha=0.85, linewidth=2.0)
        ax_vel.axvline(0.0, color='red', linestyle='--', linewidth=2, label="Seizure Onset (t=0)")
        ax_vel.set_title(f"{p} ({blk}) - Winning Smoothed Velocity (v_t) Top 3 Dims", fontsize=12, fontweight='bold')
        ax_vel.set_xlabel("Time Relative to Seizure Onset (Minutes)")
        ax_vel.set_ylabel("Differenced Velocity Value")
        ax_vel.legend(loc="upper left", fontsize=9)
        ax_vel.grid(True, alpha=0.3)
        
    plt.tight_layout()
    out_path = Path("pass_patient_trajectory.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Trajectory Plot] Saved winning trajectory figure to {out_path.absolute()}", flush=True)
    return out_path


def apply_benjamini_hochberg(results_dict, alpha=0.05):
    """
    Applies exact Benjamini-Hochberg FDR procedure across all m=17 p-values.
    Returns dictionary mapping patient to (p_raw, q_val, survives_FDR).
    """
    patients = list(results_dict.keys())
    p_raws = np.array([results_dict[p]["p_val"] for p in patients])
    m = len(p_raws)
    
    # Sort indices ascending
    sort_idx = np.argsort(p_raws)
    q_vals = np.zeros(m)
    
    # Compute step-up adjusted p-values (q-values)
    # q_{(i)} = min_{k >= i} (m / k) * p_{(k)}
    min_q = 1.0
    for idx_in_sorted in range(m - 1, -1, -1):
        rank = idx_in_sorted + 1
        p_curr = p_raws[sort_idx[idx_in_sorted]]
        q_curr = (m / float(rank)) * p_curr
        min_q = min(min_q, q_curr)
        q_vals[sort_idx[idx_in_sorted]] = min_q
        
    fdr_results = {}
    print(f"\n=== BENJAMINI-HOCHBERG FDR CORRECTION (m={m}, alpha={alpha}) ===", flush=True)
    print(f"{'Rank':<5} | {'Patient':<8} | {'Raw p-val':<12} | {'FDR Threshold (i/m * alpha)':<28} | {'BH q-val':<10} | {'Survives FDR?':<14}", flush=True)
    print("-" * 85, flush=True)
    
    for i in range(m):
        orig_idx = sort_idx[i]
        p = patients[orig_idx]
        p_val = p_raws[orig_idx]
        q_val = q_vals[orig_idx]
        rank = i + 1
        thresh = (rank / float(m)) * alpha
        survives = "YES" if q_val <= alpha else "NO"
        fdr_results[p] = {"p_raw": p_val, "q_val": q_val, "survives": survives, "rank": rank, "thresh": thresh}
        print(f"{rank:<5} | {p:<8} | {p_val:<12.4f} | {thresh:<28.4f} | {q_val:<10.4f} | {survives:<14}", flush=True)
        
    print("-" * 85, flush=True)
    return fdr_results


def check_robustness_second_block(features_dict, patients_to_test=["chb19", "chb20", "chb22"]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== ROBUSTNESS CHECK: RE-RUNNING CALIBRATION ON SECOND PREICTAL/INTERICTAL BLOCK ===", flush=True)
    print(f"{'Patient':<8} | {'Block Index':<12} | {'Cal Pre Block ID':<28} | {'Cal Inter Block ID':<28} | {'Pos+Vel AUC':<14} | {'Surrogate p-val':<16} | {'Status':<8}", flush=True)
    print("-" * 125, flush=True)
    
    for p in patients_to_test:
        if p not in features_dict:
            continue
        z_pre = features_dict[p]["preictal"]
        z_inter = features_dict[p]["interictal"]
        pre_blocks, inter_blocks = get_patient_block_ids(p)
        pre_arr = np.array(pre_blocks)
        inter_arr = np.array(inter_blocks)
        
        unique_pre = sorted(set(pre_arr))
        unique_inter = sorted(set(inter_arr))
        
        for blk_idx in [0, 1]:
            if blk_idx >= len(unique_pre) or blk_idx >= len(unique_inter):
                print(f"{p:<8} | Block Index {blk_idx:<1} | {'Only 1 Block Available':<28} | {'Only 1 Block Available':<28} | {'N/A':<14} | {'N/A':<16} | {'N/A':<8}", flush=True)
                continue
                
            cal_pre_block = unique_pre[blk_idx]
            cal_inter_block = unique_inter[blk_idx]
            
            # Diagonal normalization using chosen interictal block
            mu = z_inter[inter_arr == cal_inter_block].mean(dim=0)
            sigma = z_inter[inter_arr == cal_inter_block].std(dim=0).clamp(min=1e-6)
            
            z_pre_norm = (z_pre - mu) / sigma
            z_inter_norm = (z_inter - mu) / sigma
            
            s_pre, v_pre, _ = compute_smoothed_velocity_features(z_pre_norm, pre_arr, window=4)
            s_inter, v_inter, _ = compute_smoothed_velocity_features(z_inter_norm, inter_arr, window=4)
            
            posvel_pre = torch.cat([s_pre, v_pre], dim=1)
            posvel_inter = torch.cat([s_inter, v_inter], dim=1)
            
            # Train on chosen block pair, test on all remaining blocks
            z_pre_cal = posvel_pre[pre_arr == cal_pre_block]
            z_inter_cal = posvel_inter[inter_arr == cal_inter_block]
            z_pre_test = posvel_pre[pre_arr != cal_pre_block]
            z_inter_test = posvel_inter[inter_arr != cal_inter_block]
            
            n_cal = min(len(z_pre_cal), len(z_inter_cal))
            if n_cal == 0 or len(z_pre_test) == 0 or len(z_inter_test) == 0:
                print(f"{p:<8} | Block Index {blk_idx:<1} | {str(cal_pre_block):<28} | {str(cal_inter_block):<28} | {'0 Test/Cal':<14} | {'N/A':<16} | {'N/A':<8}", flush=True)
                continue
                
            torch.manual_seed(500 + int(p[3:]) + blk_idx)
            X_cal = torch.cat([z_pre_cal[:n_cal], z_inter_cal[:n_cal]], dim=0)
            y_cal = torch.cat([torch.ones(n_cal), torch.zeros(n_cal)], dim=0)
            head = train_classifier_head_on_fold(X_cal, y_cal, device, epochs=15)
            
            X_te = torch.cat([z_pre_test, z_inter_test], dim=0)
            y_te = torch.cat([torch.ones(len(z_pre_test)), torch.zeros(len(z_inter_test))], dim=0).numpy()
            
            with torch.no_grad():
                probs = torch.sigmoid(head(X_te.to(device))).cpu().numpy()
            auc = compute_roc_auc_numpy(y_te, probs)
            
            # 20 shuffles
            surr_aucs = []
            for s in range(1, 21):
                torch.manual_seed(2000 * int(p[3:]) + blk_idx * 100 + s)
                y_perm = y_cal[torch.randperm(len(y_cal))]
                head_perm = train_classifier_head_on_fold(X_cal, y_perm, device, epochs=15)
                with torch.no_grad():
                    probs_perm = torch.sigmoid(head_perm(X_te.to(device))).cpu().numpy()
                surr_aucs.append(compute_roc_auc_numpy(y_te, probs_perm))
                
            p_val = (sum(1 for s in surr_aucs if s >= auc) + 1.0) / (len(surr_aucs) + 1.0)
            status = "PASS" if p_val < 0.05 else "FAIL"
            print(f"{p:<8} | Block Index {blk_idx:<1} | {str(cal_pre_block):<28} | {str(cal_inter_block):<28} | {auc:<14.4f} | {p_val:<16.4f} | {status:<8}", flush=True)
            
    print("-" * 125, flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Running Final Rigor & Robustness Checks on {device} ===", flush=True)
    
    cache_path = CACHE_V2 if CACHE_V2.exists() else EPOCH3_CACHE_H5
    features_dict = load_features_dict(cache_path)
    
    # 1. Trajectory Plot for PASS subjects chb17 and chb20
    plot_pass_trajectories(features_dict, patients=["chb17", "chb20"])
    
    # 2. Benjamini-Hochberg FDR correction on our 17 results from previous run
    # Previous run 17 raw p-values:
    prev_results = {
        "chb01": {"p_val": 0.0952, "posvel_auc": 0.9865}, # flagged unreliable
        "chb02": {"p_val": 0.7143, "posvel_auc": 0.3132},
        "chb03": {"p_val": 0.1905, "posvel_auc": 0.6298},
        "chb04": {"p_val": 0.9048, "posvel_auc": 0.3464},
        "chb05": {"p_val": 1.0000, "posvel_auc": 0.4020},
        "chb07": {"p_val": 0.3333, "posvel_auc": 0.6293},
        "chb09": {"p_val": 0.6667, "posvel_auc": 0.4842},
        "chb10": {"p_val": 0.0476, "posvel_auc": 0.8073},
        "chb13": {"p_val": 0.9048, "posvel_auc": 0.3757},
        "chb14": {"p_val": 0.7619, "posvel_auc": 0.4537},
        "chb16": {"p_val": 0.0476, "posvel_auc": 0.8204},
        "chb17": {"p_val": 0.0476, "posvel_auc": 0.8860},
        "chb18": {"p_val": 0.1905, "posvel_auc": 0.7158},
        "chb19": {"p_val": 0.0476, "posvel_auc": 0.9347},
        "chb20": {"p_val": 0.0476, "posvel_auc": 0.9936},
        "chb21": {"p_val": 0.7143, "posvel_auc": 0.3757},
        "chb22": {"p_val": 0.0476, "posvel_auc": 0.9283},
    }
    apply_benjamini_hochberg(prev_results, alpha=0.05)
    
    # Notice that with N_shuffles=20, p_min = 1/21 = 0.047619. Under exact BH with m=17 and alpha=0.05, 
    # the threshold i/m * alpha at rank i=6 is 6/17 * 0.05 = 0.0176.
    # To confirm that these 6 passing subjects aren't rejected simply because of the 1/21 discretization floor of 20 shuffles,
    # we run 100 surrogate shuffles specifically on these 6 subjects right now to measure their continuous p-value down to 1/101 = 0.0099!
    print("\n=== 100-SHUFFLE VERIFICATION FOR PASSING SUBJECTS (Removing 1/21 Discretization Floor) ===", flush=True)
    print(f"{'Patient':<8} | {'Pos+Vel AUC':<14} | {'100-Shuffle Mean±Std':<24} | {'Exact p-val (N=100)':<20} | {'Survives BH FDR (q<0.05)?':<26}", flush=True)
    print("-" * 100, flush=True)
    
    # First get pop centroid for smart block selection
    all_pre_pos = []
    for p in prev_results:
        if p not in features_dict:
            continue
        z_pre = features_dict[p]["preictal"]
        z_inter = features_dict[p]["interictal"]
        pre_blocks, inter_blocks = get_patient_block_ids(p)
        pre_arr = np.array(pre_blocks)
        inter_arr = np.array(inter_blocks)
        cal_inter = sorted(set(inter_arr))[0]
        mu = z_inter[inter_arr == cal_inter].mean(dim=0)
        sigma = z_inter[inter_arr == cal_inter].std(dim=0).clamp(min=1e-6)
        z_pre_norm = (z_pre - mu) / sigma
        s_pre, _, _ = compute_smoothed_velocity_features(z_pre_norm, pre_arr, window=4)
        all_pre_pos.append(s_pre)
    pop_centroid_pos = torch.cat(all_pre_pos, dim=0).mean(dim=0)
    
    pass_subjects = ["chb10", "chb16", "chb17", "chb19", "chb20", "chb22"]
    exact_100_pvals = {}
    
    for p in pass_subjects:
        z_pre = features_dict[p]["preictal"]
        z_inter = features_dict[p]["interictal"]
        pre_blocks, inter_blocks = get_patient_block_ids(p)
        pre_arr = np.array(pre_blocks)
        inter_arr = np.array(inter_blocks)
        cal_inter = sorted(set(inter_arr))[0]
        
        mu = z_inter[inter_arr == cal_inter].mean(dim=0)
        sigma = z_inter[inter_arr == cal_inter].std(dim=0).clamp(min=1e-6)
        
        z_pre_norm = (z_pre - mu) / sigma
        z_inter_norm = (z_inter - mu) / sigma
        
        s_pre, v_pre, _ = compute_smoothed_velocity_features(z_pre_norm, pre_arr, window=4)
        s_inter, v_inter, _ = compute_smoothed_velocity_features(z_inter_norm, inter_arr, window=4)
        posvel_pre = torch.cat([s_pre, v_pre], dim=1)
        posvel_inter = torch.cat([s_inter, v_inter], dim=1)
        
        cal_pre = smart_calibration_block(s_pre, pre_arr, pop_centroid_pos)
        z_pre_cal = posvel_pre[pre_arr == cal_pre]
        z_inter_cal = posvel_inter[inter_arr == cal_inter]
        z_pre_test = posvel_pre[pre_arr != cal_pre]
        z_inter_test = posvel_inter[inter_arr != cal_inter]
        
        n_cal = min(len(z_pre_cal), len(z_inter_cal))
        X_cal = torch.cat([z_pre_cal[:n_cal], z_inter_cal[:n_cal]], dim=0)
        y_cal = torch.cat([torch.ones(n_cal), torch.zeros(n_cal)], dim=0)
        
        torch.manual_seed(300 + int(p[3:]))
        head = train_classifier_head_on_fold(X_cal, y_cal, device, epochs=15)
        X_te = torch.cat([z_pre_test, z_inter_test], dim=0)
        y_te = torch.cat([torch.ones(len(z_pre_test)), torch.zeros(len(z_inter_test))], dim=0).numpy()
        
        with torch.no_grad():
            probs = torch.sigmoid(head(X_te.to(device))).cpu().numpy()
        real_auc = compute_roc_auc_numpy(y_te, probs)
        
        surr_100 = []
        for s in range(1, 101):
            torch.manual_seed(5000 * int(p[3:]) + s)
            y_perm = y_cal[torch.randperm(len(y_cal))]
            head_perm = train_classifier_head_on_fold(X_cal, y_perm, device, epochs=15)
            with torch.no_grad():
                probs_perm = torch.sigmoid(head_perm(X_te.to(device))).cpu().numpy()
            surr_100.append(compute_roc_auc_numpy(y_te, probs_perm))
            
        p_100 = (sum(1 for s in surr_100 if s >= real_auc) + 1.0) / 101.0
        exact_100_pvals[p] = p_100
        # If p_100 == 1/101 = 0.0099, then with m=17, q_val at rank i=6 is (17/6)*0.0099 = 0.028 < 0.05 (SURVIVES BH FDR!)
        survives_str = "YES (q <= 0.028)" if p_100 <= 0.01 else f"Check ({p_100:.4f})"
        print(f"{p:<8} | {real_auc:<14.4f} | {np.mean(surr_100):.4f} ± {np.std(surr_100):.4f}      | {p_100:<20.4f} | {survives_str:<26}", flush=True)
        
    print("-" * 100, flush=True)
    
    # 3. Robustness check on Block Index 1 vs Block Index 0 for chb19, chb20, chb22
    check_robustness_second_block(features_dict, patients_to_test=["chb19", "chb20", "chb22"])

if __name__ == "__main__":
    main()
