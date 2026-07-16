"""
run_stage4_addl_checks.py — Additional Stage 4 Verification Checks
Item 4: 20-shuffle calibration surrogate/leakage control on chb02, chb21, chb04.
Item 5: Investigation of chb05's anomalous post-calibration AUC (0.1488, below chance).
         Checks: class imbalance in calibration blocks, calibration block window counts,
         label flip, and predicted probability distribution on test windows.
"""
import sys
import time
import h5py
import numpy as np
import torch
from pathlib import Path

from lopo_evaluation import PATIENTS_ALL, compute_roc_auc_numpy, train_classifier_head_on_fold
from run_prestage4_item_b import load_features_dict, EPOCH3_CACHE_H5

DATA_ROOT = Path("data/preprocessed")

def get_patient_block_ids(patient):
    h5_path = DATA_ROOT / f"{patient}_segments.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Segment file {h5_path} not found!")
    with h5py.File(h5_path, "r") as f:
        pre_blocks = [b.decode("utf-8") if isinstance(b, bytes) else str(b) for b in f["preictal"]["block_id"][:]]
        inter_blocks = [b.decode("utf-8") if isinstance(b, bytes) else str(b) for b in f["interictal"]["block_id"][:]]
    return pre_blocks, inter_blocks


def run_calibration_surrogate(patient, features_dict, device, n_shuffles=20):
    """Run 20-shuffle calibration leakage check for a given patient."""
    z_pre = features_dict[patient]["preictal"]
    z_inter = features_dict[patient]["interictal"]
    pre_blocks, inter_blocks = get_patient_block_ids(patient)
    pre_blocks_arr = np.array(pre_blocks)
    inter_blocks_arr = np.array(inter_blocks)

    unique_pre = sorted(list(set(pre_blocks_arr)))
    unique_inter = sorted(list(set(inter_blocks_arr)))

    if len(unique_pre) == 0 or len(unique_inter) == 0:
        print(f"[{patient}] Cannot calibrate — missing preictal or interictal blocks.", flush=True)
        return None, None, None

    cal_pre_block = unique_pre[0]
    cal_inter_block = unique_inter[0]

    z_pre_cal = z_pre[pre_blocks_arr == cal_pre_block]
    z_inter_cal = z_inter[inter_blocks_arr == cal_inter_block]
    z_pre_test = z_pre[pre_blocks_arr != cal_pre_block]
    z_inter_test = z_inter[inter_blocks_arr != cal_inter_block]

    n_cal_sample = min(len(z_pre_cal), len(z_inter_cal))
    if n_cal_sample == 0 or len(z_pre_test) == 0 or len(z_inter_test) == 0:
        print(f"[{patient}] Insufficient windows for calibration or test split.", flush=True)
        return None, None, None

    # Real calibrated AUC
    torch.manual_seed(100 + int(patient[3:]))
    perm_pre = torch.randperm(len(z_pre_cal))[:n_cal_sample]
    perm_inter = torch.randperm(len(z_inter_cal))[:n_cal_sample]
    X_cal = torch.cat([z_pre_cal[perm_pre], z_inter_cal[perm_inter]], dim=0)
    y_cal = torch.cat([torch.ones(n_cal_sample), torch.zeros(n_cal_sample)], dim=0)
    head_real = train_classifier_head_on_fold(X_cal, y_cal, device, epochs=15)

    X_test = torch.cat([z_pre_test, z_inter_test], dim=0)
    y_test = torch.cat([torch.ones(len(z_pre_test)), torch.zeros(len(z_inter_test))], dim=0).numpy()
    with torch.no_grad():
        probs_real = torch.sigmoid(head_real(X_test.to(device))).cpu().numpy()
    real_auc = compute_roc_auc_numpy(y_test, probs_real)

    print(f"\n[{patient}] Real Calibrated AUC: {real_auc:.4f} | Running {n_shuffles} calibration surrogate shuffles...", flush=True)
    surr_aucs = []
    for s in range(1, n_shuffles + 1):
        torch.manual_seed(7000 + int(patient[3:]) * 100 + s)
        perm_pre_s = torch.randperm(len(z_pre_cal))[:n_cal_sample]
        perm_inter_s = torch.randperm(len(z_inter_cal))[:n_cal_sample]
        X_cal_s = torch.cat([z_pre_cal[perm_pre_s], z_inter_cal[perm_inter_s]], dim=0)
        y_cal_s = torch.cat([torch.ones(n_cal_sample), torch.zeros(n_cal_sample)], dim=0)
        y_cal_s = y_cal_s[torch.randperm(2 * n_cal_sample)]
        head_s = train_classifier_head_on_fold(X_cal_s, y_cal_s, device, epochs=15)
        with torch.no_grad():
            probs_s = torch.sigmoid(head_s(X_test.to(device))).cpu().numpy()
        s_auc = compute_roc_auc_numpy(y_test, probs_s)
        surr_aucs.append(s_auc)
        print(f"  Shuffle {s:02d}/{n_shuffles}: Surrogate Calibrated AUC = {s_auc:.4f}", flush=True)

    surr_aucs = np.array(surr_aucs)
    count_exceed = np.sum(surr_aucs >= real_auc)
    p_val = count_exceed / n_shuffles
    verdict = "PASS" if p_val < 0.05 else "FAIL"

    print(f"\n--- [{patient}] Calibration Surrogate Summary ---", flush=True)
    print(f"Real Calibrated AUC:    {real_auc:.4f}", flush=True)
    print(f"Surrogate Cal Mean:     {surr_aucs.mean():.4f} +/- {surr_aucs.std():.4f}", flush=True)
    print(f"Surrogate Cal Range:    [{surr_aucs.min():.4f}, {surr_aucs.max():.4f}]", flush=True)
    print(f"Exceeding Real Cal:     {count_exceed}/{n_shuffles} ({p_val*100:.1f}%)", flush=True)
    print(f"Calibration Leakage Check: {verdict} (Threshold < 5%)", flush=True)
    return real_auc, surr_aucs, verdict


def investigate_chb05(features_dict, device):
    """Investigate chb05's anomalous below-chance post-calibration AUC of 0.1488."""
    patient = "chb05"
    print(f"\n{'='*80}", flush=True)
    print(f"=== ITEM 5: chb05 ANOMALY INVESTIGATION ===", flush=True)
    print(f"{'='*80}", flush=True)

    z_pre = features_dict[patient]["preictal"]
    z_inter = features_dict[patient]["interictal"]
    pre_blocks, inter_blocks = get_patient_block_ids(patient)
    pre_blocks_arr = np.array(pre_blocks)
    inter_blocks_arr = np.array(inter_blocks)

    unique_pre = sorted(list(set(pre_blocks_arr)))
    unique_inter = sorted(list(set(inter_blocks_arr)))

    print(f"\n[chb05] Total segments: {len(z_pre)} preictal, {len(z_inter)} interictal", flush=True)
    print(f"[chb05] Unique preictal blocks ({len(unique_pre)}): {unique_pre}", flush=True)
    print(f"[chb05] Unique interictal blocks ({len(unique_inter)}): {unique_inter}", flush=True)

    cal_pre_block = unique_pre[0]
    cal_inter_block = unique_inter[0]

    z_pre_cal = z_pre[pre_blocks_arr == cal_pre_block]
    z_inter_cal = z_inter[inter_blocks_arr == cal_inter_block]
    z_pre_test = z_pre[pre_blocks_arr != cal_pre_block]
    z_inter_test = z_inter[inter_blocks_arr != cal_inter_block]

    print(f"\n[chb05] Calibration block: preictal='{cal_pre_block}' ({len(z_pre_cal)} windows), interictal='{cal_inter_block}' ({len(z_inter_cal)} windows)", flush=True)
    print(f"[chb05] Test windows: {len(z_pre_test)} preictal, {len(z_inter_test)} interictal", flush=True)
    print(f"[chb05] Calibration class ratio (preictal:interictal): {len(z_pre_cal)}:{len(z_inter_cal)}", flush=True)

    n_cal_sample = min(len(z_pre_cal), len(z_inter_cal))
    print(f"[chb05] n_cal_sample (balanced): {n_cal_sample} each class = {2*n_cal_sample} total", flush=True)

    torch.manual_seed(100 + 5)
    perm_pre = torch.randperm(len(z_pre_cal))[:n_cal_sample]
    perm_inter = torch.randperm(len(z_inter_cal))[:n_cal_sample]
    X_cal = torch.cat([z_pre_cal[perm_pre], z_inter_cal[perm_inter]], dim=0)
    y_cal = torch.cat([torch.ones(n_cal_sample), torch.zeros(n_cal_sample)], dim=0)
    head = train_classifier_head_on_fold(X_cal, y_cal, device, epochs=15)

    X_test = torch.cat([z_pre_test, z_inter_test], dim=0)
    y_test = torch.cat([torch.ones(len(z_pre_test)), torch.zeros(len(z_inter_test))], dim=0).numpy()
    with torch.no_grad():
        probs = torch.sigmoid(head(X_test.to(device))).cpu().numpy()

    post_auc = compute_roc_auc_numpy(y_test, probs)
    print(f"\n[chb05] Post-calibration AUC on test windows: {post_auc:.4f}", flush=True)

    # Examine predicted probabilities separately for preictal and interictal test windows
    probs_pre = probs[:len(z_pre_test)]
    probs_inter = probs[len(z_pre_test):]
    print(f"\n[chb05] Test Prob Distribution — Preictal ({len(z_pre_test)} windows):", flush=True)
    print(f"  Mean prob: {probs_pre.mean():.4f} | Median: {np.median(probs_pre):.4f} | %>0.5: {(probs_pre>0.5).mean()*100:.1f}%", flush=True)
    print(f"  Min: {probs_pre.min():.4f} | Max: {probs_pre.max():.4f}", flush=True)
    print(f"\n[chb05] Test Prob Distribution — Interictal ({len(z_inter_test)} windows):", flush=True)
    print(f"  Mean prob: {probs_inter.mean():.4f} | Median: {np.median(probs_inter):.4f} | %>0.5: {(probs_inter>0.5).mean()*100:.1f}%", flush=True)
    print(f"  Min: {probs_inter.min():.4f} | Max: {probs_inter.max():.4f}", flush=True)

    # Check calibration block embedding stats
    print(f"\n[chb05] Calibration block embedding stats:", flush=True)
    print(f"  Preictal cal z — Mean norm: {torch.norm(z_pre_cal, dim=1).mean():.4f} | Std: {z_pre_cal.std(dim=0).mean():.4f}", flush=True)
    print(f"  Interictal cal z — Mean norm: {torch.norm(z_inter_cal, dim=1).mean():.4f} | Std: {z_inter_cal.std(dim=0).mean():.4f}", flush=True)
    print(f"  Preictal test z — Mean norm: {torch.norm(z_pre_test, dim=1).mean():.4f} | Std: {z_pre_test.std(dim=0).mean():.4f}", flush=True)
    print(f"  Interictal test z — Mean norm: {torch.norm(z_inter_test, dim=1).mean():.4f} | Std: {z_inter_test.std(dim=0).mean():.4f}", flush=True)

    # Compute cosine similarity between cal preictal centroid and test preictal centroids
    cal_pre_centroid = z_pre_cal.mean(dim=0, keepdim=True)
    test_pre_centroid = z_pre_test.mean(dim=0, keepdim=True)
    test_inter_centroid = z_inter_test.mean(dim=0, keepdim=True)
    cos = torch.nn.CosineSimilarity(dim=1)
    sim_pre = cos(cal_pre_centroid, test_pre_centroid).item()
    sim_inter = cos(cal_pre_centroid, test_inter_centroid).item()
    print(f"\n[chb05] Cosine similarity (cal preictal centroid vs test preictal centroid): {sim_pre:.4f}", flush=True)
    print(f"[chb05] Cosine similarity (cal preictal centroid vs test interictal centroid): {sim_inter:.4f}", flush=True)

    # Cross-block preictal AUC breakdown: for each test preictal block vs all interictal test windows
    print(f"\n[chb05] Per-block AUC breakdown (each test preictal block vs all interictal test windows):", flush=True)
    for blk in unique_pre[1:]:
        mask = (pre_blocks_arr == blk)
        z_blk = z_pre[mask]
        if len(z_blk) == 0: continue
        X_blk = torch.cat([z_blk, z_inter_test], dim=0)
        y_blk = torch.cat([torch.ones(len(z_blk)), torch.zeros(len(z_inter_test))], dim=0).numpy()
        with torch.no_grad():
            p_blk = torch.sigmoid(head(X_blk.to(device))).cpu().numpy()
        auc_blk = compute_roc_auc_numpy(y_blk, p_blk)
        print(f"  Block '{blk}' ({len(z_blk)} windows): AUC = {auc_blk:.4f}", flush=True)

    print(f"\n[chb05] DIAGNOSIS:", flush=True)
    if probs_pre.mean() < 0.5 and probs_inter.mean() > 0.5:
        print(f"  INVERTED PREDICTIONS: preictal windows score LOWER than interictal windows — calibration learned an inverted decision boundary for chb05. This is the direct cause of sub-chance AUC.", flush=True)
        print(f"  LIKELY CAUSE: The first preictal seizure block for chb05 ({cal_pre_block}, {len(z_pre_cal)} windows) produces PAC embeddings that are DISSIMILAR to test preictal blocks, likely because this block captures an atypical or transitional seizure phase that the encoder maps into the interictal embedding region.", flush=True)
    elif (probs_pre < 0.5).all():
        print(f"  ALL preictal test windows predicted as interictal by calibrated head — inverted decision boundary confirmed.", flush=True)
    else:
        print(f"  Calibration block may simply have insufficient signal for chb05's preictal pattern. Mean pre-cal prob = {probs_pre.mean():.4f}.", flush=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Stage 4 Additional Checks starting on {device} ===", flush=True)

    features_dict = load_features_dict(EPOCH3_CACHE_H5)
    print(f"Loaded {len(features_dict)} subjects from Epoch 3 cache.", flush=True)

    # --- ITEM 4: Surrogate/leakage checks on the three "massive gain" patients ---
    print(f"\n{'='*80}", flush=True)
    print(f"=== ITEM 4: CALIBRATION SURROGATE CONTROL — chb02, chb21, chb04 (20 Shuffles) ===", flush=True)
    print(f"{'='*80}", flush=True)

    for patient in ["chb02", "chb21", "chb04"]:
        run_calibration_surrogate(patient, features_dict, device, n_shuffles=20)

    # --- ITEM 5: chb05 anomaly investigation ---
    investigate_chb05(features_dict, device)


if __name__ == "__main__":
    main()
