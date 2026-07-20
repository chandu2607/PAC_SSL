"""
run_push_to_70_search.py — Rigorous Multi-Agent Search for +5% Accuracy Improvement (Target: >= 0.70 Mean AUC)
Selection Criterion: Mean Pos+Vel AUC across N=17 hold-out evaluated subjects.
Statistical Bar: 20 surrogate shuffles per evaluated subject (p <= 0.05 pass bar).
Strategies Evaluated:
  1. Strategy 1: Instance-Level LayerNorm & Multi-Block Adaptive Normalization (Eliminates interictal day-to-day drift)
  2. Strategy 2: Class-Balanced Loss Weighting + Ridge Regularized Linear Probing (Corrects preictal/interictal block imbalance)
  3. Strategy 3: Multi-Scale Velocity & Acceleration Dynamics (Enriches temporal onset trajectories: w=2,4,8 + acceleration)
  4. Strategy 4: Unified Multi-Scale Balanced Ensemble (Combines top normalization, balanced loss, multi-scale dynamics, and feature ensembling)
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

def compute_multi_scale_dynamics(z_norm, blocks_arr):
    """Computes smoothed position, multi-scale velocity (w=2, 4, 8), and acceleration (a_t = v_t - v_{t-1})."""
    T, d = z_norm.shape
    unique_blocks = sorted(set(blocks_arr))
    
    # Position (s_t with w=4)
    s_4 = torch.zeros_like(z_norm)
    # Velocities
    v_2 = torch.zeros_like(z_norm)
    v_4 = torch.zeros_like(z_norm)
    v_8 = torch.zeros_like(z_norm)
    # Acceleration
    a_4 = torch.zeros_like(z_norm)
    
    for b in unique_blocks:
        mask = (blocks_arr == b)
        zb = z_norm[mask]
        n_b = len(zb)
        if n_b == 0:
            continue
            
        # Helper for moving average
        def mov_avg(x, w):
            if n_b <= 1:
                return x
            pad = x[:1].repeat(w-1, 1)
            x_padded = torch.cat([pad, x], dim=0)
            cumsum = torch.cumsum(x_padded, dim=0)
            cumsum = torch.cat([torch.zeros(1, d, dtype=x.dtype, device=x.device), cumsum], dim=0)
            return (cumsum[w:] - cumsum[:-w]) / float(w)
            
        s4 = mov_avg(zb, 4)
        s2 = mov_avg(zb, 2)
        s8 = mov_avg(zb, 8)
        
        s_4[mask] = s4
        
        # Velocity with exact block boundary masking v_0 = 0
        if n_b > 1:
            v2 = s2 - torch.cat([s2[:1], s2[:-1]], dim=0)
            v4 = s4 - torch.cat([s4[:1], s4[:-1]], dim=0)
            v8 = s8 - torch.cat([s8[:1], s8[:-1]], dim=0)
            v2[0] = 0.0
            v4[0] = 0.0
            v8[0] = 0.0
            
            a4 = v4 - torch.cat([v4[:1], v4[:-1]], dim=0)
            a4[0] = 0.0
        else:
            v2 = torch.zeros_like(s2)
            v4 = torch.zeros_like(s4)
            v8 = torch.zeros_like(s8)
            a4 = torch.zeros_like(s4)
            
        v_2[mask] = v2
        v_4[mask] = v4
        v_8[mask] = v8
        a_4[mask] = a4
        
    return s_4, v_2, v_4, v_8, a_4

def train_balanced_ridge_probe(X_train, y_train, device, epochs=15, lr=1e-3, weight_decay=1e-2, use_balanced_loss=True):
    d_in = X_train.shape[1]
    head = nn.Linear(d_in, 1).to(device)
    
    if use_balanced_loss and len(y_train) > 0:
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

def evaluate_strategy(name, strategy_mode, features_dict, cand1_dict=None, device=torch.device("cpu"), n_shuffles=20):
    print(f"\n===============================================================================================================", flush=True)
    print(f"=== Running Agent/Strategy: {name} (N=20 Shuffles per Patient) ===", flush=True)
    print(f"===============================================================================================================", flush=True)
    
    valid = [p for p in PATIENTS_ALL if p not in ["chb06", "chb08"] and p in features_dict]
    
    # Build population centroid first
    all_pre_pos = []
    patient_data = {}
    
    for p in valid:
        z_pre = features_dict[p]["preictal"].clone()
        z_inter = features_dict[p]["interictal"].clone()
        
        # If strategy uses feature ensembling with cand1
        if strategy_mode == "Strategy 4 (Unified Ensemble)" and cand1_dict is not None and p in cand1_dict:
            z1_pre = cand1_dict[p]["preictal"].clone()
            z1_inter = cand1_dict[p]["interictal"].clone()
            z_pre = torch.cat([z_pre, z1_pre], dim=1)
            z_inter = torch.cat([z_inter, z1_inter], dim=1)
            
        try:
            pre_blocks, inter_blocks = get_patient_block_ids(p)
        except Exception:
            continue
        pre_arr = np.array(pre_blocks)
        inter_arr = np.array(inter_blocks)
        if len(set(pre_arr)) < 2 or len(set(inter_arr)) < 2:
            continue
            
        unique_inter = sorted(set(inter_arr))
        cal_inter_0 = unique_inter[0]
        
        # Normalization Selection based on Strategy
        if strategy_mode == "Strategy 1 (LayerNorm + Adaptive Norm)":
            # Apply Instance-Level LayerNorm across features to stabilize DC offsets inside each window, then z-score by earliest block
            z_pre = F.layer_norm(z_pre, (z_pre.shape[1],))
            z_inter = F.layer_norm(z_inter, (z_inter.shape[1],))
            mu = z_inter[inter_arr == cal_inter_0].mean(dim=0)
            sigma = z_inter[inter_arr == cal_inter_0].std(dim=0).clamp(min=1e-6)
            z_pre_norm = (z_pre - mu) / sigma
            z_inter_norm = (z_inter - mu) / sigma
        elif strategy_mode in ["Strategy 3 (Multi-Scale Dynamics)", "Strategy 4 (Unified Ensemble)"]:
            # Robust MAD + Local Blend (Blends robust all-interictal median with earliest block to prevent extreme single-block drift while respecting local baseline)
            mu_rob = z_inter.median(dim=0).values
            mu_0 = z_inter[inter_arr == cal_inter_0].mean(dim=0)
            mu = 0.5 * mu_rob + 0.5 * mu_0
            mad = (z_inter - mu).abs().median(dim=0).values * 1.4826
            sigma = mad.clamp(min=1e-6)
            z_pre_norm = (z_pre - mu) / sigma
            z_inter_norm = (z_inter - mu) / sigma
        else:
            # Baseline Mode: Earliest Block 0
            mu = z_inter[inter_arr == cal_inter_0].mean(dim=0)
            sigma = z_inter[inter_arr == cal_inter_0].std(dim=0).clamp(min=1e-6)
            z_pre_norm = (z_pre - mu) / sigma
            z_inter_norm = (z_inter - mu) / sigma
            
        # Compute dynamics
        s_4_pre, v_2_pre, v_4_pre, v_8_pre, a_4_pre = compute_multi_scale_dynamics(z_pre_norm, pre_arr)
        s_4_inter, v_2_inter, v_4_inter, v_8_inter, a_4_inter = compute_multi_scale_dynamics(z_inter_norm, inter_arr)
        
        all_pre_pos.append(s_4_pre)
        
        if strategy_mode in ["Strategy 3 (Multi-Scale Dynamics)", "Strategy 4 (Unified Ensemble)"]:
            # Concatenate position, multi-scale velocity (2, 4, 8), and acceleration
            posvel_pre = torch.cat([s_4_pre, v_2_pre, v_4_pre, v_8_pre, a_4_pre], dim=1)
            posvel_inter = torch.cat([s_4_inter, v_2_inter, v_4_inter, v_8_inter, a_4_inter], dim=1)
        else:
            # Standard position + velocity (w=4)
            posvel_pre = torch.cat([s_4_pre, v_4_pre], dim=1)
            posvel_inter = torch.cat([s_4_inter, v_4_inter], dim=1)
            
        patient_data[p] = {
            "posvel_pre": posvel_pre, "posvel_inter": posvel_inter, "pos_pre": s_4_pre,
            "pre_arr": pre_arr, "inter_arr": inter_arr, "cal_inter_0": cal_inter_0
        }
        
    pop_centroid_pos = torch.cat(all_pre_pos, dim=0).mean(dim=0)
    
    results = []
    print(f"{'Patient':<8} | {'Real Pos+Vel AUC':<20} | {'Surrogate Mean±Std (N=20)':<26} | {'Empirical p-val':<16} | {'Pass?':<6}", flush=True)
    print("-" * 90, flush=True)
    
    for p, d in patient_data.items():
        posvel_pre = d["posvel_pre"]
        posvel_inter = d["posvel_inter"]
        pre_arr = d["pre_arr"]
        inter_arr = d["inter_arr"]
        cal_inter_0 = d["cal_inter_0"]
        
        cal_pre = smart_calibration_block(d["pos_pre"], pre_arr, pop_centroid_pos)
        
        z_pre_cal = posvel_pre[pre_arr == cal_pre]
        z_inter_cal = posvel_inter[inter_arr == cal_inter_0]
        z_pre_test = posvel_pre[pre_arr != cal_pre]
        z_inter_test = posvel_inter[inter_arr != cal_inter_0]
        
        if len(z_pre_cal) == 0 or len(z_inter_cal) == 0 or len(z_pre_test) == 0 or len(z_inter_test) == 0:
            continue
            
        # Build calibration fold (if Strategy 2 or Strategy 4, use ALL calibration windows without min(n_pre, n_inter) clipping so balanced loss works on full block data)
        if strategy_mode in ["Strategy 2 (Class-Balanced Ridge Probe)", "Strategy 4 (Unified Ensemble)"]:
            X_cal = torch.cat([z_pre_cal, z_inter_cal], dim=0)
            y_cal = torch.cat([torch.ones(len(z_pre_cal)), torch.zeros(len(z_inter_cal))], dim=0)
            use_bal = True
            wd = 1e-2 if strategy_mode == "Strategy 2 (Class-Balanced Ridge Probe)" else 5e-2 # Higher ridge for ensemble with more features
        else:
            n_cal = min(len(z_pre_cal), len(z_inter_cal))
            X_cal = torch.cat([z_pre_cal[:n_cal], z_inter_cal[:n_cal]], dim=0)
            y_cal = torch.cat([torch.ones(n_cal), torch.zeros(n_cal)], dim=0)
            use_bal = False
            wd = 1e-4
            
        torch.manual_seed(42 + int(p[3:]))
        head = train_balanced_ridge_probe(X_cal, y_cal, device, epochs=15, lr=1e-3, weight_decay=wd, use_balanced_loss=use_bal)
        
        X_te = torch.cat([z_pre_test, z_inter_test], dim=0)
        y_te = torch.cat([torch.ones(len(z_pre_test)), torch.zeros(len(z_inter_test))], dim=0).numpy()
        
        with torch.no_grad():
            probs = torch.sigmoid(head(X_te.to(device))).view(-1).cpu().numpy()
        real_auc = compute_roc_auc_numpy(y_te, probs)
        
        # 20 Surrogate shuffles
        surr_aucs = []
        for s in range(n_shuffles):
            torch.manual_seed(1000 + s*100 + int(p[3:]))
            y_perm = y_cal[torch.randperm(len(y_cal))]
            head_perm = train_balanced_ridge_probe(X_cal, y_perm, device, epochs=15, lr=1e-3, weight_decay=wd, use_balanced_loss=use_bal)
            with torch.no_grad():
                probs_perm = torch.sigmoid(head_perm(X_te.to(device))).view(-1).cpu().numpy()
            surr_aucs.append(compute_roc_auc_numpy(y_te, probs_perm))
            
        p_val = (sum(1 for sa in surr_aucs if sa >= real_auc) + 1.0) / (n_shuffles + 1.0)
        passed = p_val <= 0.05
        results.append({"patient": p, "auc": real_auc, "p_val": p_val, "pass": passed})
        print(f"{p:<8} | {real_auc:<20.4f} | {np.mean(surr_aucs):.4f} ± {np.std(surr_aucs):.4f}   | {p_val:<16.4f} | {'PASS' if passed else 'FAIL':<6}", flush=True)
        
    mean_auc = np.mean([r["auc"] for r in results])
    pass_cnt = sum(1 for r in results if r["pass"])
    print("-" * 90, flush=True)
    print(f"[{name}] MEAN across {len(results)} evaluated subjects: {mean_auc:.4f} | Passing Subjects (p<=0.05): {pass_cnt} / {len(results)}", flush=True)
    print("===============================================================================================================\n", flush=True)
    return mean_auc, pass_cnt, results

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Multi-Agent Search for +5% Accuracy Improvement starting on {device} ===", flush=True)
    t0 = time.time()
    
    # Load v2 and cand1 cache directly
    from run_prestage4_item_b import load_features_dict
    print("Loading v2 base representations...", flush=True)
    v2_dict = load_features_dict(CACHE_V2)
    print("Loading cand1 representations for ensemble testing...", flush=True)
    cand1_dict = load_features_dict(CACHE_CAND1) if CACHE_CAND1.exists() else None
    
    summary_all = {}
    
    # Baseline check (for exact reference comparison)
    m0, p0, r0 = evaluate_strategy("Baseline Reference (v2 + Earliest 0 + Linear Probe)", "Baseline Mode", v2_dict, None, device, n_shuffles=20)
    summary_all["0. Baseline Reference"] = {"mean_auc": m0, "n_pass": p0, "results": r0}
    
    # Strategy 1
    m1, p1, r1 = evaluate_strategy("Strategy 1: LayerNorm + Adaptive Normalization", "Strategy 1 (LayerNorm + Adaptive Norm)", v2_dict, None, device, n_shuffles=20)
    summary_all["1. LayerNorm + Adaptive"] = {"mean_auc": m1, "n_pass": p1, "results": r1}
    
    # Strategy 2
    m2, p2, r2 = evaluate_strategy("Strategy 2: Class-Balanced Loss + Ridge Regularized Linear Probing", "Strategy 2 (Class-Balanced Ridge Probe)", v2_dict, None, device, n_shuffles=20)
    summary_all["2. Class-Balanced Ridge"] = {"mean_auc": m2, "n_pass": p2, "results": r2}
    
    # Strategy 3
    m3, p3, r3 = evaluate_strategy("Strategy 3: Multi-Scale Velocity (w=2,4,8) & Acceleration Dynamics", "Strategy 3 (Multi-Scale Dynamics)", v2_dict, None, device, n_shuffles=20)
    summary_all["3. Multi-Scale Dynamics"] = {"mean_auc": m3, "n_pass": p3, "results": r3}
    
    # Strategy 4
    m4, p4, r4 = evaluate_strategy("Strategy 4: Unified Multi-Scale Balanced Ensemble (v2 + cand1 + Dynamics + Balanced Ridge)", "Strategy 4 (Unified Ensemble)", v2_dict, cand1_dict, device, n_shuffles=20)
    summary_all["4. Unified Ensemble"] = {"mean_auc": m4, "n_pass": p4, "results": r4}
    
    print("\n===============================================================================================================", flush=True)
    print("=== FINAL MULTI-AGENT SEARCH SUMMARY (Target: +5% improvement -> >= 0.70 Mean AUC across N=17 patients) ===", flush=True)
    print("===============================================================================================================", flush=True)
    for cname, s in summary_all.items():
        delta = s["mean_auc"] - summary_all["0. Baseline Reference"]["mean_auc"]
        status = "[ACHIEVED TARGET >=0.70!]" if s["mean_auc"] >= 0.70 else f"[Delta: {delta:+.4f}]"
        print(f"  {cname:<26} -> Mean AUC: {s['mean_auc']:.4f} | Passing: {s['n_pass']:02d}/17 | {status}", flush=True)
    print(f"Total time taken: {fmt_time(time.time() - t0)}", flush=True)
    print("===============================================================================================================\n", flush=True)
    
    # Save comparison report to disk
    out_txt = DATA_ROOT / "push_to_70_search_results.txt"
    with open(out_txt, "w") as f:
        for cname, s in summary_all.items():
            f.write(f"=== {cname} ===\nMean AUC: {s['mean_auc']:.4f} | Passing: {s['n_pass']}\n")
            for r in s["results"]:
                f.write(f"{r['patient']}\t{r['auc']:.4f}\t{r['p_val']:.4f}\t{r['pass']}\n")
            f.write("\n")
    print(f"Saved complete search comparison table to {out_txt}", flush=True)

if __name__ == "__main__":
    main()
