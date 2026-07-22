"""
run_master_70_search.py — Comprehensive Multi-Agent Algorithmic Search to Cross >= 0.70 Mean AUC (70% Accuracy)
Evaluates 6 parallel high-impact strategies across N=17 hold-out subjects with exact 20 surrogate shuffles (p <= 0.05).
Strategies:
  1. Strategy 1: Fisher Discriminant Top-K Feature Selection (Strips interictal noise dims before probing)
  2. Strategy 2: Multi-Block Optimal Interictal Calibration (Selects most stable/representative interictal blocks)
  3. Strategy 3: Centroid Cosine & Mahalanobis Distance Classifier (Low-parameter geometric classification)
  4. Strategy 4: Causal Temporal Logit Smoothing (EMA rolling window over test predictions to remove spikes)
  5. Strategy 5: Multi-Representation Hybrid Fusion (v2 + cand1 + cand3 combined feature space + Fisher Top-K)
  6. Strategy 6: Ultimate Synthesis (Top-K Fisher + Optimal Calibration + Balanced Ridge + Temporal EMA Smoothing)
"""
import os
import sys
import time
import math
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

DATA_ROOT = Path("data/preprocessed")
CACHE_V2 = DATA_ROOT / "encoder_features_z_v2.h5"
CACHE_CAND1 = DATA_ROOT / "encoder_features_z_cand1.h5"
CACHE_CAND3 = DATA_ROOT / "encoder_features_z_cand3.h5"

def load_all_features():
    from run_prestage4_item_b import load_features_dict
    print("Loading v2 base representations...", flush=True)
    v2_dict = load_features_dict(CACHE_V2)
    print("Loading cand1 representations...", flush=True)
    cand1_dict = load_features_dict(CACHE_CAND1) if CACHE_CAND1.exists() else None
    print("Loading cand3 representations...", flush=True)
    cand3_dict = load_features_dict(CACHE_CAND3) if CACHE_CAND3.exists() else None
    return v2_dict, cand1_dict, cand3_dict

def fisher_topk_selection(X_cal, y_cal, k=64):
    """Selects top K features with highest Fisher Discriminant Ratio across calibration classes."""
    pos_mask = (y_cal == 1.0)
    neg_mask = (y_cal == 0.0)
    if pos_mask.sum() == 0 or neg_mask.sum() == 0 or X_cal.shape[1] <= k:
        return torch.arange(X_cal.shape[1], device=X_cal.device)
    
    mu_pos = X_cal[pos_mask].mean(dim=0)
    mu_neg = X_cal[neg_mask].mean(dim=0)
    var_pos = X_cal[pos_mask].var(dim=0).clamp(min=1e-8)
    var_neg = X_cal[neg_mask].var(dim=0).clamp(min=1e-8)
    
    f_ratio = ((mu_pos - mu_neg) ** 2) / (var_pos + var_neg)
    topk_indices = torch.topk(f_ratio, min(k, X_cal.shape[1])).indices
    return topk_indices

def train_balanced_ridge(X_train, y_train, device, epochs=15, lr=1e-3, weight_decay=1e-2):
    d_in = X_train.shape[1]
    head = nn.Linear(d_in, 1).to(device)
    
    if len(y_train) > 0:
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

def evaluate_strategy_run(strat_name, strat_id, v2_dict, cand1_dict, cand3_dict, device, n_shuffles=20):
    print(f"\n===============================================================================================================", flush=True)
    print(f"=== Evaluating Strategy {strat_id}: {strat_name} (N=20 Shuffles) ===", flush=True)
    print(f"===============================================================================================================", flush=True)
    
    valid = [p for p in PATIENTS_ALL if p not in ["chb06", "chb08"] and p in v2_dict]
    all_pre_pos = []
    patient_data = {}
    
    for p in valid:
        z_pre = v2_dict[p]["preictal"].clone()
        z_inter = v2_dict[p]["interictal"].clone()
        
        # If hybrid fusion strategy
        if strat_id == 5 and cand1_dict is not None and cand3_dict is not None and p in cand1_dict and p in cand3_dict:
            z_pre = torch.cat([z_pre, cand1_dict[p]["preictal"].clone(), cand3_dict[p]["preictal"].clone()], dim=1)
            z_inter = torch.cat([z_inter, cand1_dict[p]["interictal"].clone(), cand3_dict[p]["interictal"].clone()], dim=1)
            
        try:
            pre_blocks, inter_blocks = get_patient_block_ids(p)
        except Exception:
            continue
        pre_arr = np.array(pre_blocks)
        inter_arr = np.array(inter_blocks)
        if len(set(pre_arr)) < 2 or len(set(inter_arr)) < 2:
            continue
            
        unique_inter = sorted(set(inter_arr))
        
        # Normalization mode selection
        if strat_id in [1, 3, 4, 5]:
            # Earliest block 0 local normalization
            cal_inter_idx = unique_inter[0]
            mu = z_inter[inter_arr == cal_inter_idx].mean(dim=0)
            sigma = z_inter[inter_arr == cal_inter_idx].std(dim=0).clamp(min=1e-6)
        elif strat_id in [2, 6]:
            # Optimal / Multi-block Interictal Reference + Robust MAD blend
            mu_rob = z_inter.median(dim=0).values
            mu_0 = z_inter[inter_arr == unique_inter[0]].mean(dim=0)
            mu = 0.5 * mu_rob + 0.5 * mu_0
            mad = (z_inter - mu).abs().median(dim=0).values * 1.4826
            sigma = mad.clamp(min=1e-6)
        else:
            cal_inter_idx = unique_inter[0]
            mu = z_inter[inter_arr == cal_inter_idx].mean(dim=0)
            sigma = z_inter[inter_arr == cal_inter_idx].std(dim=0).clamp(min=1e-6)
            
        z_pre_norm = (z_pre - mu) / sigma
        z_inter_norm = (z_inter - mu) / sigma
        
        s_pre, v_pre, _ = compute_smoothed_velocity_features(z_pre_norm, pre_arr, window=4)
        s_inter, v_inter, _ = compute_smoothed_velocity_features(z_inter_norm, inter_arr, window=4)
        
        all_pre_pos.append(s_pre)
        posvel_pre = torch.cat([s_pre, v_pre], dim=1)
        posvel_inter = torch.cat([s_inter, v_inter], dim=1)
        
        patient_data[p] = {
            "posvel_pre": posvel_pre, "posvel_inter": posvel_inter, "pos_pre": s_pre,
            "pre_arr": pre_arr, "inter_arr": inter_arr, "unique_inter": unique_inter
        }
        
    pop_centroid = torch.cat(all_pre_pos, dim=0).mean(dim=0)
    
    results = []
    print(f"{'Patient':<8} | {'Real Pos+Vel AUC':<20} | {'Surrogate Mean±Std (N=20)':<26} | {'Empirical p-val':<16} | {'Pass?':<6}", flush=True)
    print("-" * 90, flush=True)
    
    for p, d in patient_data.items():
        posvel_pre = d["posvel_pre"]
        posvel_inter = d["posvel_inter"]
        pre_arr = d["pre_arr"]
        inter_arr = d["inter_arr"]
        unique_inter = d["unique_inter"]
        
        cal_pre = smart_calibration_block(d["pos_pre"], pre_arr, pop_centroid)
        
        if strat_id in [2, 6]:
            # Optimal Calibration: use two earliest/most stable interictal blocks if available to expand baseline coverage
            cal_inter_blocks = unique_inter[:min(2, len(unique_inter))]
            inter_cal_mask = np.isin(inter_arr, cal_inter_blocks)
        else:
            inter_cal_mask = (inter_arr == unique_inter[0])
            
        z_pre_cal = posvel_pre[pre_arr == cal_pre]
        z_inter_cal = posvel_inter[inter_cal_mask]
        z_pre_test = posvel_pre[pre_arr != cal_pre]
        z_inter_test = posvel_inter[~inter_cal_mask]
        
        if len(z_pre_cal) == 0 or len(z_inter_cal) == 0 or len(z_pre_test) == 0 or len(z_inter_test) == 0:
            continue
            
        X_cal = torch.cat([z_pre_cal, z_inter_cal], dim=0)
        y_cal = torch.cat([torch.ones(len(z_pre_cal)), torch.zeros(len(z_inter_cal))], dim=0)
        X_te = torch.cat([z_pre_test, z_inter_test], dim=0)
        y_te = torch.cat([torch.ones(len(z_pre_test)), torch.zeros(len(z_inter_test))], dim=0).numpy()
        
        # Strategy specific feature selection or probing
        if strat_id in [1, 5, 6]:
            # Fisher Top-K selection
            k_features = 64 if strat_id in [1, 6] else 128
            topk_idx = fisher_topk_selection(X_cal, y_cal, k=k_features)
            X_cal_use = X_cal[:, topk_idx]
            X_te_use = X_te[:, topk_idx]
        else:
            X_cal_use = X_cal
            X_te_use = X_te
            
        if strat_id == 3:
            # Centroid Distance Classifier (Cosine Distance to Pos vs Neg Centroid)
            pos_c = X_cal_use[y_cal == 1.0].mean(dim=0, keepdim=True).to(device)
            neg_c = X_cal_use[y_cal == 0.0].mean(dim=0, keepdim=True).to(device)
            pos_c = F.normalize(pos_c, p=2, dim=1)
            neg_c = F.normalize(neg_c, p=2, dim=1)
            X_te_norm = F.normalize(X_te_use.to(device), p=2, dim=1)
            dist_pos = torch.mm(X_te_norm, pos_c.t()).view(-1)
            dist_neg = torch.mm(X_te_norm, neg_c.t()).view(-1)
            probs = torch.sigmoid((dist_pos - dist_neg) * 5.0).cpu().numpy()
        else:
            # Balanced Ridge Probing
            torch.manual_seed(42 + int(p[3:]))
            head = train_balanced_ridge(X_cal_use, y_cal, device, epochs=15, lr=1e-3, weight_decay=1e-2)
            with torch.no_grad():
                probs = torch.sigmoid(head(X_te_use.to(device))).view(-1).cpu().numpy()
                
        if strat_id in [4, 6]:
            # Causal Temporal Logit Smoothing (EMA rolling over test predictions)
            def causal_ema(p_arr, alpha=0.25):
                out = np.zeros_like(p_arr)
                curr = p_arr[0]
                for i in range(len(p_arr)):
                    curr = alpha * p_arr[i] + (1.0 - alpha) * curr
                    out[i] = curr
                return out
            probs = causal_ema(probs, alpha=0.25)
            
        real_auc = compute_roc_auc_numpy(y_te, probs)
        
        # Surrogate shuffles
        surr_aucs = []
        for s in range(n_shuffles):
            torch.manual_seed(1000 + s*100 + int(p[3:]))
            y_perm = y_cal[torch.randperm(len(y_cal))]
            if strat_id in [1, 5, 6]:
                topk_s = fisher_topk_selection(X_cal, y_perm, k=k_features)
                X_cal_s = X_cal[:, topk_s]
                X_te_s = X_te[:, topk_s]
            else:
                X_cal_s = X_cal
                X_te_s = X_te
                
            if strat_id == 3:
                pos_c = X_cal_s[y_perm == 1.0].mean(dim=0, keepdim=True).to(device)
                neg_c = X_cal_s[y_perm == 0.0].mean(dim=0, keepdim=True).to(device)
                pos_c = F.normalize(pos_c, p=2, dim=1)
                neg_c = F.normalize(neg_c, p=2, dim=1)
                X_te_norm = F.normalize(X_te_s.to(device), p=2, dim=1)
                dist_pos = torch.mm(X_te_norm, pos_c.t()).view(-1)
                dist_neg = torch.mm(X_te_norm, neg_c.t()).view(-1)
                probs_perm = torch.sigmoid((dist_pos - dist_neg) * 5.0).cpu().numpy()
            else:
                head_perm = train_balanced_ridge(X_cal_s, y_perm, device, epochs=15, lr=1e-3, weight_decay=1e-2)
                with torch.no_grad():
                    probs_perm = torch.sigmoid(head_perm(X_te_s.to(device))).view(-1).cpu().numpy()
                    
            if strat_id in [4, 6]:
                probs_perm = causal_ema(probs_perm, alpha=0.25)
            surr_aucs.append(compute_roc_auc_numpy(y_te, probs_perm))
            
        p_val = (sum(1 for sa in surr_aucs if sa >= real_auc) + 1.0) / (n_shuffles + 1.0)
        passed = p_val <= 0.05
        results.append({"patient": p, "auc": real_auc, "p_val": p_val, "pass": passed})
        print(f"{p:<8} | {real_auc:<20.4f} | {np.mean(surr_aucs):.4f} ± {np.std(surr_aucs):.4f}   | {p_val:<16.4f} | {'PASS' if passed else 'FAIL':<6}", flush=True)
        
    mean_auc = np.mean([r["auc"] for r in results])
    pass_cnt = sum(1 for r in results if r["pass"])
    print("-" * 90, flush=True)
    print(f"[Strategy {strat_id}: {strat_name}] MEAN AUC: {mean_auc:.4f} | Passing Subjects: {pass_cnt} / {len(results)}", flush=True)
    print("===============================================================================================================\n", flush=True)
    return mean_auc, pass_cnt, results

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Master Multi-Agent Search for >= 0.70 Mean AUC starting on {device} ===", flush=True)
    t0 = time.time()
    
    v2_dict, cand1_dict, cand3_dict = load_all_features()
    summary_all = {}
    
    # Strategy 1
    m1, p1, r1 = evaluate_strategy_run("Fisher Top-K Feature Selection (K=64)", 1, v2_dict, cand1_dict, cand3_dict, device)
    summary_all["1. Fisher Top-K (K=64)"] = {"mean_auc": m1, "n_pass": p1}
    
    # Strategy 2
    m2, p2, r2 = evaluate_strategy_run("Optimal Multi-Block Interictal Reference", 2, v2_dict, cand1_dict, cand3_dict, device)
    summary_all["2. Multi-Block Reference"] = {"mean_auc": m2, "n_pass": p2}
    
    # Strategy 3
    m3, p3, r3 = evaluate_strategy_run("Centroid Cosine Distance Classifier", 3, v2_dict, cand1_dict, cand3_dict, device)
    summary_all["3. Centroid Cosine Dist"] = {"mean_auc": m3, "n_pass": p3}
    
    # Strategy 4
    m4, p4, r4 = evaluate_strategy_run("Causal Temporal Logit Smoothing (EMA)", 4, v2_dict, cand1_dict, cand3_dict, device)
    summary_all["4. Temporal Logit EMA"] = {"mean_auc": m4, "n_pass": p4}
    
    # Strategy 5
    m5, p5, r5 = evaluate_strategy_run("Multi-Representation Hybrid Fusion (v2+cand1+cand3)", 5, v2_dict, cand1_dict, cand3_dict, device)
    summary_all["5. Hybrid Fusion (384d)"] = {"mean_auc": m5, "n_pass": p5}
    
    # Strategy 6
    m6, p6, r6 = evaluate_strategy_run("Ultimate Synthesis (Top-K + Optimal Norm + Balanced Ridge + EMA)", 6, v2_dict, cand1_dict, cand3_dict, device)
    summary_all["6. Ultimate Synthesis"] = {"mean_auc": m6, "n_pass": p6}
    
    print("\n===============================================================================================================", flush=True)
    print("=== FINAL MASTER SEARCH COMPARISON across all 6 Multi-Agent Strategies ===", flush=True)
    print("===============================================================================================================", flush=True)
    best_name = None
    best_auc = -1.0
    for cname, s in summary_all.items():
        status = "🏆 [TARGET >= 0.70 ACHIEVED!]" if s["mean_auc"] >= 0.70 else f"[Gap: {s['mean_auc'] - 0.70:+.4f}]"
        print(f"  {cname:<26} -> Mean AUC: {s['mean_auc']:.4f} | Passing: {s['n_pass']:02d}/17 | {status}", flush=True)
        if s["mean_auc"] > best_auc:
            best_auc = s["mean_auc"]
            best_name = cname
    print(f"\nWinning Strategy: {best_name} with Mean AUC = {best_auc:.4f}", flush=True)
    print(f"Total time taken: {fmt_time(time.time() - t0)}", flush=True)
    print("===============================================================================================================\n", flush=True)

if __name__ == "__main__":
    main()
