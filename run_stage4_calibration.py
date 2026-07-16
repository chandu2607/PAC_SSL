"""
run_stage4_calibration.py — Stage 4: Patient-Specific Calibration & Calibration Surrogate Control
For each of the 21 valid patients (excluding chb06/chb08):
  1. Calibrates linear probe on exactly 1 preictal block + 1 matched interictal block (earliest block_id, deterministic).
  2. Tests strictly on all remaining blocks for that patient (never on calibration blocks).
  3. Reports: Pre-Calibration AUC (from Stage 3 LOPO), Post-Calibration AUC, and Delta.
  4. Runs Surrogate/Chance Control on Calibration itself for chb20, chb19, and chb03 (20 shuffles of calibration labels).
"""
import os
import sys
import time
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

from lopo_evaluation import PATIENTS_ALL, compute_roc_auc_numpy, compute_confusion_matrix_numpy, train_classifier_head_on_fold
from run_prestage4_item_b import load_features_dict, EPOCH1_CACHE_H5, EPOCH3_CACHE_H5

DATA_ROOT = Path("data/preprocessed")

def get_patient_block_ids(patient):
    h5_path = DATA_ROOT / f"{patient}_segments.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Segment file {h5_path} not found!")
    with h5py.File(h5_path, "r") as f:
        pre_blocks = [b.decode("utf-8") if isinstance(b, bytes) else str(b) for b in f["preictal"]["block_id"][:]]
        inter_blocks = [b.decode("utf-8") if isinstance(b, bytes) else str(b) for b in f["interictal"]["block_id"][:]]
    return pre_blocks, inter_blocks


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Stage 4: Patient-Specific Calibration starting on {device} ===", flush=True)
    
    # Check if Item B decided on Epoch 1 or Epoch 3 (defaulting to Epoch 3 if cache doesn't specify otherwise, we check what features exist or allow flag)
    # By default we check which cache we decided to use, but let's load both or load Epoch 3 unless specified
    cache_path = EPOCH3_CACHE_H5
    if len(sys.argv) > 1 and sys.argv[1] == "--epoch1":
        cache_path = EPOCH1_CACHE_H5
        print(f"Using Epoch 1 encoder representations from {cache_path}", flush=True)
    else:
        print(f"Using Epoch 3 encoder representations from {cache_path}", flush=True)
        
    features_dict = load_features_dict(cache_path)
    print(f"Loaded {len(features_dict)} subjects from cache.", flush=True)
    
    # 1. First run exact Stage 3 LOPO on these features to get our baseline Pre-Calibration AUCs
    print("\nPre-computing Stage 3 LOPO baseline AUCs (Pre-Calibration baseline)...", flush=True)
    pre_aucs = {}
    for test_p in PATIENTS_ALL:
        if test_p not in features_dict:
            continue
        train_patients = [p for p in PATIENTS_ALL if p != test_p and p in features_dict]
        pre_train_list = [features_dict[p]["preictal"] for p in train_patients if len(features_dict[p]["preictal"]) > 0]
        inter_train_list = [features_dict[p]["interictal"] for p in train_patients if len(features_dict[p]["interictal"]) > 0]
        
        z_pre_train = torch.cat(pre_train_list, dim=0)
        z_inter_train = torch.cat(inter_train_list, dim=0)
        n_sample = min(len(z_pre_train), len(z_inter_train))
        
        torch.manual_seed(42 + int(test_p[3:]))
        perm_inter = torch.randperm(len(z_inter_train))[:n_sample]
        perm_pre = torch.randperm(len(z_pre_train))[:n_sample]
        X_train = torch.cat([z_pre_train[perm_pre], z_inter_train[perm_inter]], dim=0)
        y_train = torch.cat([torch.ones(n_sample), torch.zeros(n_sample)], dim=0)
        
        head = train_classifier_head_on_fold(X_train, y_train, device, epochs=15)
        
        z_pre_test = features_dict[test_p]["preictal"]
        z_inter_test = features_dict[test_p]["interictal"]
        X_test = torch.cat([z_pre_test, z_inter_test], dim=0)
        y_test = torch.cat([torch.ones(len(z_pre_test)), torch.zeros(len(z_inter_test))], dim=0).numpy()
        
        if len(X_test) > 0 and len(z_pre_test) > 0 and len(z_inter_test) > 0:
            with torch.no_grad():
                probs = torch.sigmoid(head(X_test.to(device))).cpu().numpy()
            pre_aucs[test_p] = compute_roc_auc_numpy(y_test, probs)
        else:
            pre_aucs[test_p] = 0.5
            
    # 2. Stage 4 Calibration across 21 valid subjects (excluding chb06, chb08)
    print("\n=========================================================================================================", flush=True)
    print("=== STAGE 4: PATIENT-SPECIFIC CALIBRATION TABLE (1 Preictal Block + 1 Interictal Block Calibration) ===", flush=True)
    print("=========================================================================================================", flush=True)
    print(f"{'Patient':<10} | {'Pre-Cal AUC':<14} | {'Post-Cal AUC':<14} | {'Delta':<12} | {'Cal Blocks (Pre / Inter)':<30}", flush=True)
    print("-" * 88, flush=True)
    
    post_aucs = {}
    deltas = {}
    cal_block_info = {}
    
    valid_patients = [p for p in PATIENTS_ALL if p not in ["chb06", "chb08"]]
    
    for p in valid_patients:
        if p not in features_dict:
            continue
            
        z_pre = features_dict[p]["preictal"]
        z_inter = features_dict[p]["interictal"]
        
        pre_blocks, inter_blocks = get_patient_block_ids(p)
        pre_blocks = np.array(pre_blocks)
        inter_blocks = np.array(inter_blocks)
        
        unique_pre = sorted(list(set(pre_blocks)))
        unique_inter = sorted(list(set(inter_blocks)))
        
        if len(unique_pre) == 0 or len(unique_inter) == 0:
            continue
            
        cal_pre_block = unique_pre[0]
        cal_inter_block = unique_inter[0]
        
        cal_pre_mask = (pre_blocks == cal_pre_block)
        test_pre_mask = (pre_blocks != cal_pre_block)
        
        cal_inter_mask = (inter_blocks == cal_inter_block)
        test_inter_mask = (inter_blocks != cal_inter_block)
        
        # If a patient has only 1 preictal block, we can't test on remaining blocks if all preictal are in cal block!
        # Let's check: if test_pre_mask.sum() == 0, note that all preictal data is in the single block
        z_pre_cal = z_pre[cal_pre_mask]
        z_inter_cal = z_inter[cal_inter_mask]
        
        z_pre_test = z_pre[test_pre_mask]
        z_inter_test = z_inter[test_inter_mask]
        
        n_cal_sample = min(len(z_pre_cal), len(z_inter_cal))
        if n_cal_sample == 0 or len(z_pre_test) == 0 or len(z_inter_test) == 0:
            # Cannot calibrate or test out-of-block if missing split
            post_auc = 0.5
            info_str = f"{cal_pre_block[:12]}* (Single Block / Insufficient test)"
        else:
            torch.manual_seed(100 + int(p[3:]))
            perm_pre = torch.randperm(len(z_pre_cal))[:n_cal_sample]
            perm_inter = torch.randperm(len(z_inter_cal))[:n_cal_sample]
            
            X_cal = torch.cat([z_pre_cal[perm_pre], z_inter_cal[perm_inter]], dim=0)
            y_cal = torch.cat([torch.ones(n_cal_sample), torch.zeros(n_cal_sample)], dim=0)
            
            head_cal = train_classifier_head_on_fold(X_cal, y_cal, device, epochs=15)
            
            X_test = torch.cat([z_pre_test, z_inter_test], dim=0)
            y_test = torch.cat([torch.ones(len(z_pre_test)), torch.zeros(len(z_inter_test))], dim=0).numpy()
            
            with torch.no_grad():
                probs_post = torch.sigmoid(head_cal(X_test.to(device))).cpu().numpy()
            post_auc = compute_roc_auc_numpy(y_test, probs_post)
            info_str = f"{cal_pre_block[:12]}... / {cal_inter_block[:12]}..."
            
        pre_auc = pre_aucs.get(p, 0.5)
        delta = post_auc - pre_auc
        post_aucs[p] = post_auc
        deltas[p] = delta
        cal_block_info[p] = (cal_pre_block, cal_inter_block, len(z_pre_test), len(z_inter_test))
        
        print(f"{p:<10} | {pre_auc:<14.4f} | {post_auc:<14.4f} | {delta:<+12.4f} | {info_str:<30}", flush=True)
        
    print("-" * 88, flush=True)
    mean_pre = np.mean([pre_aucs[p] for p in valid_patients if p in post_aucs])
    mean_post = np.mean([post_aucs[p] for p in valid_patients if p in post_aucs])
    mean_delta = np.mean([deltas[p] for p in valid_patients if p in post_aucs])
    print(f"{'Mean':<10} | {mean_pre:<14.4f} | {mean_post:<14.4f} | {mean_delta:<+12.4f} | {'Across valid calibrated patients':<30}", flush=True)
    print("=========================================================================================================\n", flush=True)

    # 3. Surrogate / Chance Control on Calibration across Headline Subjects (chb20, chb19, chb03)
    print("=========================================================================================================", flush=True)
    print("=== SURROGATE CONTROL ON CALIBRATION (20 Shuffles of Calibration Labels on chb20, chb19, chb03) ===", flush=True)
    print("=========================================================================================================", flush=True)
    
    headline_subjects = ["chb20", "chb19", "chb03"]
    n_shuffles = 20
    
    for p in headline_subjects:
        if p not in features_dict or p not in cal_block_info:
            continue
            
        z_pre = features_dict[p]["preictal"]
        z_inter = features_dict[p]["interictal"]
        pre_blocks, inter_blocks = get_patient_block_ids(p)
        pre_blocks = np.array(pre_blocks)
        inter_blocks = np.array(inter_blocks)
        
        cal_pre_block, cal_inter_block, _, _ = cal_block_info[p]
        z_pre_cal = z_pre[pre_blocks == cal_pre_block]
        z_inter_cal = z_inter[inter_blocks == cal_inter_block]
        
        z_pre_test = z_pre[pre_blocks != cal_pre_block]
        z_inter_test = z_inter[inter_blocks != cal_inter_block]
        
        n_cal_sample = min(len(z_pre_cal), len(z_inter_cal))
        if n_cal_sample == 0 or len(z_pre_test) == 0 or len(z_inter_test) == 0:
            continue
            
        X_cal = torch.cat([z_pre_cal[:n_cal_sample], z_inter_cal[:n_cal_sample]], dim=0)
        X_test = torch.cat([z_pre_test, z_inter_test], dim=0)
        y_test = torch.cat([torch.ones(len(z_pre_test)), torch.zeros(len(z_inter_test))], dim=0).numpy()
        
        real_post_auc = post_aucs[p]
        surr_aucs = []
        
        print(f"\n[{p}] Real Calibrated AUC: {real_post_auc:.4f} | Running {n_shuffles} calibration shuffles...", flush=True)
        for s in range(1, n_shuffles + 1):
            torch.manual_seed(5000 + int(p[3:]) * 100 + s)
            perm_pre = torch.randperm(len(z_pre_cal))[:n_cal_sample]
            perm_inter = torch.randperm(len(z_inter_cal))[:n_cal_sample]
            X_cal_s = torch.cat([z_pre_cal[perm_pre], z_inter_cal[perm_inter]], dim=0)
            
            # Shuffle calibration labels exactly 50/50
            y_cal_s = torch.cat([torch.ones(n_cal_sample), torch.zeros(n_cal_sample)], dim=0)
            y_cal_s = y_cal_s[torch.randperm(2 * n_cal_sample)]
            
            head_s = train_classifier_head_on_fold(X_cal_s, y_cal_s, device, epochs=15)
            with torch.no_grad():
                probs_s = torch.sigmoid(head_s(X_test.to(device))).cpu().numpy()
            s_auc = compute_roc_auc_numpy(y_test, probs_s)
            surr_aucs.append(s_auc)
            print(f"  Shuffle {s:02d}/{n_shuffles}: Surrogate Calibrated AUC = {s_auc:.4f}", flush=True)
            
        surr_aucs = np.array(surr_aucs)
        count_exceed = np.sum(surr_aucs >= real_post_auc)
        p_val = count_exceed / n_shuffles
        verdict = "PASS" if p_val < 0.05 else "FAIL"
        
        print(f"\n--- [{p}] Calibration Surrogate Summary ---", flush=True)
        print(f"Real Calibrated AUC:    {real_post_auc:.4f}", flush=True)
        print(f"Surrogate Cal Mean:     {surr_aucs.mean():.4f} +/- {surr_aucs.std():.4f}", flush=True)
        print(f"Surrogate Cal Range:    [{surr_aucs.min():.4f}, {surr_aucs.max():.4f}]", flush=True)
        print(f"Exceeding Real Cal:     {count_exceed}/{n_shuffles} ({p_val*100:.1f}%)", flush=True)
        print(f"Final Calibration Check: {verdict} (Threshold < 5%)", flush=True)

if __name__ == "__main__":
    main()
