"""
lopo_v2.py — Improved LOPO Evaluation with Per-Patient Z-Score Normalization
             + Embedding-Similarity-Based Calibration Block Selection

Improvements over v1:
  Step 2: Per-patient z-score normalization of embeddings before LOPO training
          → removes inter-subject baseline shift (root cause of most failures)
  Step 3: Calibration block selected by cosine similarity to population preictal
          centroid, not by date → fixes chb05/chb02/chb04 inversions
  Rich progress output with % complete and ETA
"""
import sys
import time
import h5py
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from lopo_evaluation import (
    PATIENTS_ALL, compute_roc_auc_numpy, compute_confusion_matrix_numpy,
    train_classifier_head_on_fold
)
from run_prestage4_item_b import load_features_dict

# Use best checkpoint if available, else fall back to v2 final, then Epoch 3
CHECKPOINT_CANDIDATES = [
    Path("data/pac_ssl_encoder_best.pt"),
    Path("data/pac_ssl_encoder_v2.pt"),
    Path("data/pac_ssl_encoder.pt"),
]
CACHE_H5_V2   = Path("data/preprocessed/encoder_features_z_v2.h5")
EPOCH3_CACHE  = Path("data/preprocessed/encoder_features_z.h5")

BAR_WIDTH = 40

def progress_bar(current, total, prefix="", suffix=""):
    pct  = current / max(total, 1)
    filled = int(BAR_WIDTH * pct)
    bar  = "#" * filled + "-" * (BAR_WIDTH - filled)
    print(f"\r{prefix} [{bar}] {pct*100:5.1f}%  {suffix}", end="", flush=True)

def fmt_time(secs):
    m, s = int(secs) // 60, int(secs) % 60
    return f"{m}m{s:02d}s" if m else f"{s}s"


def extract_features_v2(checkpoint_path, device):
    """Extract embeddings using the v2 checkpoint."""
    if CACHE_H5_V2.exists():
        print(f"[Extract] v2 cache already exists at {CACHE_H5_V2}. Loading.", flush=True)
        return load_features_dict(CACHE_H5_V2)

    print(f"\n[Extract] Using checkpoint: {checkpoint_path}", flush=True)
    from pac_ssl_model import PACSSLEncoder, PACFeatureExtractor
    from lopo_evaluation import NPY_ROOT
    fe  = PACFeatureExtractor(fs=256, n_samples=1024).to(device)
    enc = PACSSLEncoder(fs=256, n_samples=1024, num_channels=18, cnn_out=32, d_model=128).to(device)
    enc.load_state_dict(torch.load(checkpoint_path, map_location=device))
    enc.eval()

    t0 = time.time()
    fd = {}
    with h5py.File(CACHE_H5_V2, "w") as out_f:
        for i, p in enumerate(PATIENTS_ALL, 1):
            progress_bar(i, len(PATIENTS_ALL), prefix="  Extracting",
                         suffix=f"{p}  ({i}/{len(PATIENTS_ALL)})")
            pre_path  = NPY_ROOT / f"{p}_preictal.npy"
            inter_path = NPY_ROOT / f"{p}_interictal.npy"
            pre_data  = np.load(pre_path,   mmap_mode='r') if pre_path.exists()  else np.zeros((0,18,1024), dtype=np.float32)
            inter_data = np.load(inter_path, mmap_mode='r') if inter_path.exists() else np.zeros((0,18,1024), dtype=np.float32)

            def encode(data):
                zs = []
                for idx in range(0, len(data), 256):
                    batch = torch.from_numpy(data[idx:idx+256].copy()).to(device)
                    with torch.no_grad():
                        zs.append(enc.forward_from_features(fe(batch)).cpu())
                return torch.cat(zs, dim=0) if zs else torch.zeros((0,128))

            z_pre  = encode(pre_data)
            z_inter = encode(inter_data)
            grp = out_f.create_group(p)
            grp.create_dataset("preictal",   data=z_pre.numpy(),   compression="gzip")
            grp.create_dataset("interictal", data=z_inter.numpy(), compression="gzip")
            fd[p] = {"preictal": z_pre, "interictal": z_inter}

    print(f"\n[Extract] Done in {fmt_time(time.time()-t0)}. Saved to {CACHE_H5_V2}", flush=True)
    return fd


def normalize_per_patient(features_dict):
    """
    Step 2: Z-score normalize each patient's embeddings using that patient's
    combined (preictal + interictal) mean and std.
    Removes inter-subject baseline shift without touching relative preictal/interictal structure.
    """
    print("\n[Normalize] Applying per-patient z-score normalization...", flush=True)
    normed = {}
    for p, fd in features_dict.items():
        z_pre   = fd["preictal"]
        z_inter = fd["interictal"]
        combined = torch.cat([z_pre, z_inter], dim=0) if (len(z_pre) > 0 and len(z_inter) > 0) else (z_pre if len(z_pre) > 0 else z_inter)
        if len(combined) == 0:
            normed[p] = fd
            continue
        mean = combined.mean(dim=0)
        std  = combined.std(dim=0).clamp(min=1e-6)
        normed[p] = {
            "preictal":   (z_pre   - mean) / std,
            "interictal": (z_inter - mean) / std,
        }
    print(f"  Done. Normalized {len(normed)} patients.", flush=True)
    return normed


def run_lopo_v2(features_dict, device):
    """
    Strict 23-fold LOPO with per-patient normalized embeddings.
    Shows per-fold progress with ETA.
    """
    print(f"\n{'─'*65}", flush=True)
    print(f" LOPO Evaluation v2 (with per-patient z-score normalization)", flush=True)
    print(f"{'─'*65}", flush=True)

    results = []
    t0 = time.time()
    patients = [p for p in PATIENTS_ALL if p in features_dict]

    for fold_idx, test_p in enumerate(patients, 1):
        elapsed = time.time() - t0
        eta = elapsed / fold_idx * (len(patients) - fold_idx) if fold_idx > 1 else 0
        progress_bar(fold_idx, len(patients),
                     prefix="  LOPO",
                     suffix=f"Fold {fold_idx}/{len(patients)} [{test_p}]  ETA: {fmt_time(eta)}")

        train_ps = [p for p in patients if p != test_p]
        pre_list  = [features_dict[p]["preictal"]   for p in train_ps if len(features_dict[p]["preictal"])   > 0]
        inter_list = [features_dict[p]["interictal"] for p in train_ps if len(features_dict[p]["interictal"]) > 0]

        z_pre_tr  = torch.cat(pre_list,   dim=0)
        z_inter_tr = torch.cat(inter_list, dim=0)
        n_sample  = min(len(z_pre_tr), len(z_inter_tr))

        torch.manual_seed(42 + int(test_p[3:]))
        X_tr = torch.cat([z_pre_tr[torch.randperm(len(z_pre_tr))[:n_sample]],
                          z_inter_tr[torch.randperm(len(z_inter_tr))[:n_sample]]], dim=0)
        y_tr = torch.cat([torch.ones(n_sample), torch.zeros(n_sample)], dim=0)

        head = train_classifier_head_on_fold(X_tr, y_tr, device, epochs=15)

        z_pre_te  = features_dict[test_p]["preictal"]
        z_inter_te = features_dict[test_p]["interictal"]

        if len(z_pre_te) == 0 or len(z_inter_te) == 0:
            auc, sens, spec = 0.5, 1.0, 0.0
        else:
            X_te = torch.cat([z_pre_te, z_inter_te], dim=0)
            y_te = torch.cat([torch.ones(len(z_pre_te)), torch.zeros(len(z_inter_te))], dim=0).numpy()
            with torch.no_grad():
                probs = torch.sigmoid(head(X_te.to(device))).cpu().numpy()
            auc  = compute_roc_auc_numpy(y_te, probs)
            preds = (probs >= 0.5).astype(int)
            tn, fp, fn, tp = compute_confusion_matrix_numpy(y_te, preds)
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        results.append({"patient": test_p, "auc": auc, "sens": sens, "spec": spec})

    print(f"\n\n[LOPO v2] Complete in {fmt_time(time.time()-t0)}", flush=True)
    return results


def smart_calibration_block(z_pre, pre_blocks_arr, pop_pre_centroid):
    """
    Step 3: Pick calibration preictal block by cosine similarity to
    population preictal centroid (not date).
    Prevents selecting atypical seizure blocks (chb05/chb02/chb04 fix).
    """
    unique_blocks = sorted(set(pre_blocks_arr))
    if len(unique_blocks) == 1:
        return unique_blocks[0]

    best_block, best_sim = unique_blocks[0], -2.0
    norm_pop = pop_pre_centroid / (pop_pre_centroid.norm() + 1e-8)
    for blk in unique_blocks:
        mask = (pre_blocks_arr == blk)
        if mask.sum() == 0:
            continue
        centroid = z_pre[mask].mean(dim=0)
        norm_c   = centroid / (centroid.norm() + 1e-8)
        sim = (norm_pop * norm_c).sum().item()
        if sim > best_sim:
            best_sim, best_block = sim, blk
    return best_block


def run_calibration_v2(features_dict, device, pre_aucs):
    """
    Stage 4 calibration with embedding-similarity block selection.
    """
    from run_stage4_calibration import get_patient_block_ids

    # Population preictal centroid (used for smart block selection)
    all_pre = torch.cat([features_dict[p]["preictal"] for p in PATIENTS_ALL
                         if p in features_dict and len(features_dict[p]["preictal"]) > 0], dim=0)
    pop_centroid = all_pre.mean(dim=0)

    valid = [p for p in PATIENTS_ALL if p not in ["chb06","chb08"] and p in features_dict]
    results = []
    print(f"\n{'─'*65}", flush=True)
    print(f" Stage 4 v2 Calibration (smart block selection, {len(valid)} patients)", flush=True)
    print(f"{'─'*65}", flush=True)

    for i, p in enumerate(valid, 1):
        progress_bar(i, len(valid), prefix="  Calibration",
                     suffix=f"{p}  ({i}/{len(valid)})")
        z_pre   = features_dict[p]["preictal"]
        z_inter = features_dict[p]["interictal"]

        try:
            pre_blocks, inter_blocks = get_patient_block_ids(p)
        except FileNotFoundError:
            continue

        pre_blocks_arr   = np.array(pre_blocks)
        inter_blocks_arr = np.array(inter_blocks)
        unique_inter = sorted(set(inter_blocks_arr))
        if len(unique_inter) == 0:
            continue

        cal_pre_block   = smart_calibration_block(z_pre, pre_blocks_arr, pop_centroid)
        cal_inter_block = unique_inter[0]

        z_pre_cal  = z_pre[pre_blocks_arr == cal_pre_block]
        z_inter_cal = z_inter[inter_blocks_arr == cal_inter_block]
        z_pre_test  = z_pre[pre_blocks_arr != cal_pre_block]
        z_inter_test = z_inter[inter_blocks_arr != cal_inter_block]

        n_cal = min(len(z_pre_cal), len(z_inter_cal))
        if n_cal == 0 or len(z_pre_test) == 0 or len(z_inter_test) == 0:
            results.append({"patient": p, "pre_auc": pre_aucs.get(p, 0.5),
                            "post_auc": 0.5, "delta": 0.5 - pre_aucs.get(p, 0.5),
                            "cal_block": cal_pre_block, "note": "Insufficient split"})
            continue

        torch.manual_seed(200 + int(p[3:]))
        X_cal = torch.cat([z_pre_cal[:n_cal], z_inter_cal[:n_cal]], dim=0)
        y_cal = torch.cat([torch.ones(n_cal), torch.zeros(n_cal)], dim=0)
        head  = train_classifier_head_on_fold(X_cal, y_cal, device, epochs=15)

        X_te = torch.cat([z_pre_test, z_inter_test], dim=0)
        y_te = torch.cat([torch.ones(len(z_pre_test)), torch.zeros(len(z_inter_test))], dim=0).numpy()
        with torch.no_grad():
            probs = torch.sigmoid(head(X_te.to(device))).cpu().numpy()
        post_auc = compute_roc_auc_numpy(y_te, probs)
        pre_auc  = pre_aucs.get(p, 0.5)
        results.append({"patient": p, "pre_auc": pre_auc, "post_auc": post_auc,
                        "delta": post_auc - pre_auc, "cal_block": cal_pre_block, "note": ""})

    print(flush=True)
    return results


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65, flush=True)
    print(" PAC-SSL v2 Evaluation Pipeline", flush=True)
    print(f" Step 2: Per-patient z-score normalization", flush=True)
    print(f" Step 3: Embedding-similarity calibration block selection", flush=True)
    print(f" Device: {device}", flush=True)
    print("=" * 65, flush=True)

    # Determine checkpoint
    ckpt = next((c for c in CHECKPOINT_CANDIDATES if c.exists()), None)
    if ckpt is None:
        print("ERROR: No checkpoint found. Run train_stage2_v2.py first.", flush=True)
        sys.exit(1)
    print(f"\n[Checkpoint] Using: {ckpt}", flush=True)

    # 0. Extract features with v2 checkpoint
    if CACHE_H5_V2.exists():
        print(f"\n[Cache] Loading v2 features from {CACHE_H5_V2}...", flush=True)
        raw_fd = load_features_dict(CACHE_H5_V2)
    else:
        raw_fd = extract_features_v2(ckpt, device)

    # 1. Per-patient normalization (Step 2)
    fd = normalize_per_patient(raw_fd)

    # 2. LOPO v2
    lopo_results = run_lopo_v2(fd, device)
    pre_aucs = {r["patient"]: r["auc"] for r in lopo_results}

    # Print LOPO table
    print(f"\n{'─'*65}", flush=True)
    print(f" {'Patient':<10} {'AUC':<10} {'Sens':<10} {'Spec':<10}", flush=True)
    print(f"{'─'*65}", flush=True)
    for r in lopo_results:
        print(f" {r['patient']:<10} {r['auc']:<10.4f} {r['sens']:<10.4f} {r['spec']:<10.4f}", flush=True)
    mean_auc  = np.mean([r["auc"]  for r in lopo_results])
    mean_sens = np.mean([r["sens"] for r in lopo_results])
    mean_spec = np.mean([r["spec"] for r in lopo_results])
    print(f"{'─'*65}", flush=True)
    print(f" {'Mean':<10} {mean_auc:<10.4f} {mean_sens:<10.4f} {mean_spec:<10.4f}", flush=True)
    print(f"{'─'*65}\n", flush=True)
    print(f" >>> v2 Mean LOPO AUC: {mean_auc:.4f}  (v1 was 0.5313)", flush=True)

    # 3. Stage 4 v2 calibration
    cal_results = run_calibration_v2(fd, device, pre_aucs)

    valid_cal = [r for r in cal_results if r["note"] != "Insufficient split"]
    print(f"\n{'─'*65}", flush=True)
    print(f" {'Patient':<10} {'Pre-Cal':<12} {'Post-Cal':<12} {'Delta':<10} {'Cal Block'}", flush=True)
    print(f"{'─'*65}", flush=True)
    for r in cal_results:
        print(f" {r['patient']:<10} {r['pre_auc']:<12.4f} {r['post_auc']:<12.4f} "
              f"{r['delta']:<+10.4f} {str(r['cal_block'])[:30]}", flush=True)
    if valid_cal:
        m_pre  = np.mean([r["pre_auc"]  for r in valid_cal])
        m_post = np.mean([r["post_auc"] for r in valid_cal])
        m_del  = np.mean([r["delta"]    for r in valid_cal])
        print(f"{'─'*65}", flush=True)
        print(f" {'Mean':<10} {m_pre:<12.4f} {m_post:<12.4f} {m_del:<+10.4f}", flush=True)
        print(f"\n >>> v2 Mean Post-Calibration AUC: {m_post:.4f}  (v1 was 0.5457)", flush=True)

    # Save summary
    out = Path("data/preprocessed/lopo_v2_results.txt")
    with open(out, "w") as f:
        f.write("=== LOPO v2 Results (z-score normalization) ===\n")
        f.write(f"Patient\tAUC\tSens\tSpec\n")
        for r in lopo_results:
            f.write(f"{r['patient']}\t{r['auc']:.4f}\t{r['sens']:.4f}\t{r['spec']:.4f}\n")
        f.write(f"\nMean_AUC\t{mean_auc:.4f}\n")
        f.write(f"Mean_Sens\t{mean_sens:.4f}\n")
        f.write(f"Mean_Spec\t{mean_spec:.4f}\n")
    print(f"\n Summary saved to {out}", flush=True)


if __name__ == "__main__":
    main()
