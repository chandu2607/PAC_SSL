"""
run_prestage4_item_a.py — Strengthened Surrogate Control (20 Shuffles)
For headline subjects: chb20, chb19, chb03.
Runs 20 surrogate LOPO folds with shuffled labels, computes full distribution,
empirical p-value (percentage of surrogate AUCs >= real AUC), and PASS/FAIL verdict (< 5%).
"""
import time
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

from lopo_evaluation import PATIENTS_ALL, compute_roc_auc_numpy, train_classifier_head_on_fold

CACHE_H5 = Path("data/preprocessed/encoder_features_z.h5")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Pre-Stage-4 Item A: Strengthened Surrogate Control starting on {device} ===", flush=True)
    
    if not CACHE_H5.exists():
        raise FileNotFoundError(f"Cache {CACHE_H5} not found!")
        
    features_dict = {}
    with h5py.File(CACHE_H5, "r") as f:
        for p in PATIENTS_ALL:
            if p in f:
                features_dict[p] = {
                    "preictal": torch.from_numpy(f[p]["preictal"][:]).float(),
                    "interictal": torch.from_numpy(f[p]["interictal"][:]).float()
                }
    print(f"Loaded {len(features_dict)} subjects from cache.", flush=True)
    
    headline_subjects = ["chb20", "chb19", "chb03"]
    n_shuffles = 20
    
    for test_p in headline_subjects:
        if test_p not in features_dict:
            continue
            
        train_patients = [p for p in PATIENTS_ALL if p != test_p and p in features_dict]
        pre_train_list = [features_dict[p]["preictal"] for p in train_patients if len(features_dict[p]["preictal"]) > 0]
        inter_train_list = [features_dict[p]["interictal"] for p in train_patients if len(features_dict[p]["interictal"]) > 0]
        
        z_pre_train = torch.cat(pre_train_list, dim=0)
        z_inter_train = torch.cat(inter_train_list, dim=0)
        n_sample = min(len(z_pre_train), len(z_inter_train))
        
        # Real AUC first
        torch.manual_seed(42 + int(test_p[3:]))
        perm_inter = torch.randperm(len(z_inter_train))[:n_sample]
        perm_pre = torch.randperm(len(z_pre_train))[:n_sample]
        X_train_real = torch.cat([z_pre_train[perm_pre], z_inter_train[perm_inter]], dim=0)
        y_train_real = torch.cat([torch.ones(n_sample), torch.zeros(n_sample)], dim=0)
        
        head_real = train_classifier_head_on_fold(X_train_real, y_train_real, device, epochs=15)
        
        z_pre_test = features_dict[test_p]["preictal"]
        z_inter_test = features_dict[test_p]["interictal"]
        X_test = torch.cat([z_pre_test, z_inter_test], dim=0)
        y_test = torch.cat([torch.ones(len(z_pre_test)), torch.zeros(len(z_inter_test))], dim=0).numpy()
        
        with torch.no_grad():
            probs_real = torch.sigmoid(head_real(X_test.to(device))).cpu().numpy()
        real_auc = compute_roc_auc_numpy(y_test, probs_real)
        
        print(f"\n==========================================", flush=True)
        print(f"[{test_p}] Real LOPO AUC: {real_auc:.4f}", flush=True)
        print(f"Running {n_shuffles} surrogate shuffles across training fold...", flush=True)
        
        surrogate_aucs = []
        for s in range(1, n_shuffles + 1):
            torch.manual_seed(1000 + int(test_p[3:]) * 100 + s)
            perm_inter_s = torch.randperm(len(z_inter_train))[:n_sample]
            perm_pre_s = torch.randperm(len(z_pre_train))[:n_sample]
            X_train_s = torch.cat([z_pre_train[perm_pre_s], z_inter_train[perm_inter_s]], dim=0)
            
            # Shuffled labels: exactly half ones, half zeros randomized
            y_train_s = torch.cat([torch.ones(n_sample), torch.zeros(n_sample)], dim=0)
            perm_y = torch.randperm(2 * n_sample)
            y_train_s = y_train_s[perm_y]
            
            head_s = train_classifier_head_on_fold(X_train_s, y_train_s, device, epochs=15)
            with torch.no_grad():
                probs_s = torch.sigmoid(head_s(X_test.to(device))).cpu().numpy()
            s_auc = compute_roc_auc_numpy(y_test, probs_s)
            surrogate_aucs.append(s_auc)
            print(f"  Shuffle {s:02d}/{n_shuffles}: Surrogate AUC = {s_auc:.4f}", flush=True)
            
        surrogate_aucs = np.array(surrogate_aucs)
        count_exceed = np.sum(surrogate_aucs >= real_auc)
        p_val = count_exceed / n_shuffles
        verdict = "PASS" if p_val < 0.05 else "FAIL"
        
        print(f"\n--- [{test_p}] Surrogate Distribution Summary ---", flush=True)
        print(f"Real AUC:        {real_auc:.4f}", flush=True)
        print(f"Surrogate Mean:  {surrogate_aucs.mean():.4f} +/- {surrogate_aucs.std():.4f}", flush=True)
        print(f"Surrogate Range: [{surrogate_aucs.min():.4f}, {surrogate_aucs.max():.4f}]", flush=True)
        print(f"Exceeding Real:  {count_exceed}/{n_shuffles} ({p_val*100:.1f}%)", flush=True)
        print(f"Final Verdict:   {verdict} (Threshold < 5%)", flush=True)

if __name__ == "__main__":
    main()
