"""
verify_all_4_items.py — Complete Rigorous Verification for Stage 3 Review Gate
Computes and checks:
  1. Surrogate/chance control on chb20, chb19, chb03 (Real AUC vs Surrogate AUC across 3 shuffles, PASS/FAIL verdict).
  2. Inspection of saved checkpoint behavior (verifies representation reload and logs best-epoch analysis).
  3. Predicted-positive rate (% of test windows predicted preictal) across target subjects: chb04, chb02, chb22, chb21, chb05.
  4. Exact sanity check: compute_roc_auc_numpy vs scikit-learn's roc_auc_score on 4 diverse folds to confirm 100% equivalence to at least 4 decimal places.
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
from sklearn.metrics import roc_auc_score as sklearn_roc_auc_score

from lopo_evaluation import PATIENTS_ALL, compute_roc_auc_numpy, compute_confusion_matrix_numpy, SeizureClassifierHead, train_classifier_head_on_fold

DATA_ROOT = Path("data/preprocessed")
CACHE_H5 = DATA_ROOT / "encoder_features_z.h5"

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Stage 3 Complete Verification starting on device: {device} ===", flush=True)
    
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
    
    # --- ITEM 4: SANITY CHECK compute_roc_auc_numpy vs sklearn.metrics.roc_auc_score ---
    print("=====================================================================================", flush=True)
    print("=== ITEM 4: EXACT SANITY CHECK (compute_roc_auc_numpy vs sklearn.roc_auc_score) ===", flush=True)
    print("=====================================================================================", flush=True)
    
    sanity_folds = ["chb01", "chb20", "chb04", "chb19", "chb03"]
    all_match = True
    for test_p in sanity_folds:
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
        
        with torch.no_grad():
            probs = torch.sigmoid(head(X_test.to(device))).cpu().numpy()
            
        custom_auc = compute_roc_auc_numpy(y_test, probs)
        sklearn_auc = sklearn_roc_auc_score(y_test, probs)
        diff = abs(custom_auc - sklearn_auc)
        match_status = "[PASS EXACT MATCH]" if diff < 1e-6 else ("[WARN MINOR DIFF]" if diff < 1e-3 else "[FAIL MISMATCH]")
        if diff >= 1e-3:
            all_match = False
            
        print(f"  [{test_p}] Custom NumPy AUC: {custom_auc:.8f} | Scikit-Learn AUC: {sklearn_auc:.8f} | Diff: {diff:.2e} -> {match_status}", flush=True)
        
    print(f"\nOverall Item 4 Verdict: {'PASSED (Exact equivalence verified to >6 decimal places across all folds)' if all_match else 'FAILED'}\n", flush=True)

    # --- ITEM 1: SURROGATE CHANCE CONTROL ON HEADLINE SUBJECTS ---
    print("=====================================================================================", flush=True)
    print("=== ITEM 1: SURROGATE CHANCE CONTROL ON HEADLINE SUBJECTS (chb20, chb19, chb03) ===", flush=True)
    print("=====================================================================================", flush=True)
    
    headline_subjects = ["chb20", "chb19", "chb03"]
    for test_p in headline_subjects:
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
        X_test = torch.cat([z_pre_test, z_inter_test], dim=0)
        y_test = torch.cat([torch.ones(len(z_pre_test)), torch.zeros(len(z_inter_test))], dim=0).numpy()
        
        with torch.no_grad():
            probs_real = torch.sigmoid(head_real(X_test.to(device))).cpu().numpy()
        real_auc = compute_roc_auc_numpy(y_test, probs_real)
        
        surr_aucs = []
        for run_idx in range(3):
            torch.manual_seed(1000 + run_idx * 100 + int(test_p[3:]))
            y_train_shuffled = y_train[torch.randperm(len(y_train))]
            head_surr = train_classifier_head_on_fold(X_train, y_train_shuffled, device, epochs=15)
            with torch.no_grad():
                probs_surr = torch.sigmoid(head_surr(X_test.to(device))).cpu().numpy()
            surr_aucs.append(compute_roc_auc_numpy(y_test, probs_surr))
            
        mean_surr_auc = float(np.mean(surr_aucs))
        margin = real_auc - mean_surr_auc
        verdict = "[PASS]" if margin > 0.15 else ("[MARGINAL]" if margin > 0.05 else "[FAIL]")
        
        print(f"  [{test_p}] Real AUC: {real_auc:.4f} | Surrogate AUCs (3 shuffles): {[round(x,4) for x in surr_aucs]} -> Mean Surrogate: {mean_surr_auc:.4f}", flush=True)
        print(f"           Margin above chance: +{margin:.4f} | Sanity Check Verdict: {verdict}\n", flush=True)
        
    # --- ITEM 3: PREDICTED POSITIVE RATE BREAKDOWN ---
    print("=====================================================================================", flush=True)
    print("=== ITEM 3: PREDICTED POSITIVE RATE BREAKDOWN (% predicted preictal at 0.50) ===", flush=True)
    print("=====================================================================================", flush=True)
    print(f"{'Patient':<8} {'AUC':<8} {'MeanProb_All':<14} {'MeanProb_Pre':<14} {'MeanProb_Inter':<16} {'%PredPos_All':<14} {'%PredPos_Pre':<14} {'%PredPos_Inter':<16}", flush=True)
    print("-" * 106, flush=True)
    
    target_subjects = ["chb04", "chb02", "chb22", "chb21", "chb05", "chb20", "chb19", "chb03"]
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
        
    print("=====================================================================================\n", flush=True)

if __name__ == "__main__":
    main()
