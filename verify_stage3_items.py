"""
verify_stage3_items.py — Stage 3 Rigorous Verification Script
Addresses:
  1. Surrogate/Chance Control on Headline Subjects (chb20, chb19, chb03) to confirm they clear chance.
  2. Predicted Positive Rate breakdown across low-AUC (chb04, chb02, chb22, chb21, chb05) and high-AUC subjects.
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

from lopo_evaluation import PATIENTS_ALL, compute_roc_auc_numpy, compute_confusion_matrix_numpy, SeizureClassifierHead, train_classifier_head_on_fold

DATA_ROOT = Path("data/preprocessed")
CACHE_H5 = DATA_ROOT / "encoder_features_z.h5"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Stage 3 Rigorous Verification starting on device: {device} ===", flush=True)
    
    if not CACHE_H5.exists():
        raise FileNotFoundError(f"Cache {CACHE_H5} not found!")
        
    print(f"Loading pre-extracted encoder representations `z` from {CACHE_H5}...", flush=True)
    features_dict = {}
    with h5py.File(CACHE_H5, "r") as f:
        for p in PATIENTS_ALL:
            if p in f:
                features_dict[p] = {
                    "preictal": torch.from_numpy(f[p]["preictal"][:]).float(),
                    "interictal": torch.from_numpy(f[p]["interictal"][:]).float()
                }
    print(f"Loaded {len(features_dict)} subjects from cache.\n", flush=True)
    
    # --- ITEM 1: Surrogate Check on Headline Subjects ---
    headline_subjects = ["chb20", "chb19", "chb03"]
    print("=====================================================================================", flush=True)
    print("=== ITEM 1: SURROGATE CHANCE CONTROL ON HEADLINE SUBJECTS (chb20, chb19, chb03) ===", flush=True)
    print("=====================================================================================", flush=True)
    
    for test_p in headline_subjects:
        if test_p not in features_dict:
            continue
        train_patients = [p for p in PATIENTS_ALL if p != test_p and p in features_dict]
        pre_train_list = [features_dict[p]["preictal"] for p in train_patients if len(features_dict[p]["preictal"]) > 0]
        inter_train_list = [features_dict[p]["interictal"] for p in train_patients if len(features_dict[p]["interictal"]) > 0]
        z_pre_train = torch.cat(pre_train_list, dim=0)
        z_inter_train = torch.cat(inter_train_list, dim=0)
        n_sample = min(len(z_pre_train), len(z_inter_train))
        
        # Set exact deterministic seed for real run
        torch.manual_seed(42 + int(test_p[3:]))
        perm_inter = torch.randperm(len(z_inter_train))[:n_sample]
        perm_pre = torch.randperm(len(z_pre_train))[:n_sample]
        X_train = torch.cat([z_pre_train[perm_pre], z_inter_train[perm_inter]], dim=0)
        y_train = torch.cat([torch.ones(n_sample), torch.zeros(n_sample)], dim=0)
        
        # Train Real Head
        head_real = train_classifier_head_on_fold(X_train, y_train, device, epochs=15)
        
        z_pre_test = features_dict[test_p]["preictal"]
        z_inter_test = features_dict[test_p]["interictal"]
        X_test = torch.cat([z_pre_test, z_inter_test], dim=0)
        y_test = torch.cat([torch.ones(len(z_pre_test)), torch.zeros(len(z_inter_test))], dim=0).numpy()
        
        with torch.no_grad():
            logits_real = head_real(X_test.to(device)).cpu()
            probs_real = torch.sigmoid(logits_real).numpy()
        real_auc = compute_roc_auc_numpy(y_test, probs_real)
        
        # Train Surrogate Heads across 3 random shuffles for robust chance estimation
        surr_aucs = []
        for run_idx in range(3):
            torch.manual_seed(1000 + run_idx * 100 + int(test_p[3:]))
            y_train_shuffled = y_train[torch.randperm(len(y_train))]
            head_surr = train_classifier_head_on_fold(X_train, y_train_shuffled, device, epochs=15)
            with torch.no_grad():
                logits_surr = head_surr(X_test.to(device)).cpu()
                probs_surr = torch.sigmoid(logits_surr).numpy()
            surr_aucs.append(compute_roc_auc_numpy(y_test, probs_surr))
            
        mean_surr_auc = float(np.mean(surr_aucs))
        margin = real_auc - mean_surr_auc
        verdict = "PASS" if margin > 0.15 else ("MARGINAL" if margin > 0.05 else "FAIL")
        
        print(f"[{test_p}] Real AUC: {real_auc:.4f} | Surrogate AUCs (3 runs): {[round(x,4) for x in surr_aucs]} -> Mean Surrogate: {mean_surr_auc:.4f}", flush=True)
        print(f"         Margin above chance: +{margin:.4f} | Sanity Check Verdict: {verdict}\n", flush=True)
        
    # --- ITEM 3: Predicted Positive Rate & Distribution Analysis ---
    target_subjects = ["chb04", "chb02", "chb22", "chb21", "chb05", "chb20", "chb19", "chb03"]
    print("=====================================================================================", flush=True)
    print("=== ITEM 3: PREDICTED POSITIVE RATE BREAKDOWN (Baseline Shift Confirmation) ===", flush=True)
    print("=====================================================================================", flush=True)
    print(f"{'Patient':<8} {'AUC':<8} {'MeanProb_All':<14} {'MeanProb_Pre':<14} {'MeanProb_Inter':<16} {'%PredPos_All':<14} {'%PredPos_Pre':<14} {'%PredPos_Inter':<16}", flush=True)
    print("-" * 106, flush=True)
    
    for test_p in target_subjects:
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
        
        head_real = train_classifier_head_on_fold(X_train, y_train, device, epochs=15)
        
        z_pre_test = features_dict[test_p]["preictal"]
        z_inter_test = features_dict[test_p]["interictal"]
        
        with torch.no_grad():
            probs_pre = torch.sigmoid(head_real(z_pre_test.to(device))).cpu().numpy()
            probs_inter = torch.sigmoid(head_real(z_inter_test.to(device))).cpu().numpy()
            
        probs_all = np.concatenate([probs_pre, probs_inter])
        y_all = np.concatenate([np.ones(len(probs_pre)), np.zeros(len(probs_inter))])
        auc = compute_roc_auc_numpy(y_all, probs_all)
        
        mean_all = float(np.mean(probs_all))
        mean_pre = float(np.mean(probs_pre))
        mean_inter = float(np.mean(probs_inter))
        
        pct_all = float(np.mean(probs_all >= 0.50)) * 100.0
        pct_pre = float(np.mean(probs_pre >= 0.50)) * 100.0
        pct_inter = float(np.mean(probs_inter >= 0.50)) * 100.0
        
        print(f"{test_p:<8} {auc:<8.4f} {mean_all:<14.4f} {mean_pre:<14.4f} {mean_inter:<16.4f} {pct_all:<14.1f}% {pct_pre:<14.1f}% {pct_inter:<16.1f}%", flush=True)
        
    print("=====================================================================================", flush=True)

if __name__ == "__main__":
    main()
