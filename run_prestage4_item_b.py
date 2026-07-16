"""
run_prestage4_item_b.py — Pre-Stage-4 Item B: Checkpoint Analysis (Epoch 1 vs Epoch 3)
1. Trains / loads pac_ssl_encoder_epoch1.pt (Epoch 1 checkpoint of PAC-SSL pretraining).
2. Fast pre-extraction of 128-dim embeddings `z_epoch1` to data/preprocessed/encoder_features_z_epoch1.h5 (~25 seconds).
3. Runs strict LOPO cross-validation using `z_epoch1` on target extremes: chb20, chb19, chb03, chb04, chb02.
4. Compares Epoch 1 AUC vs Epoch 3 AUC on the same subjects.
5. Decides which encoder checkpoint to carry forward into Stage 4 calibration and gives the exact one-sentence rationale.
"""
import os
import sys
import time
import argparse
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from pac_ssl_model import PACSSLEncoder, PACSSLPretextModel, PACFeatureExtractor
from train_pretext_stage2 import MmapPreprocessedEEGDataset, BlockRandomSampler, run_training_epoch, run_validation
from lopo_evaluation import PATIENTS_ALL, NPY_ROOT, compute_roc_auc_numpy, train_classifier_head_on_fold

EPOCH1_CHECKPOINT = Path("data/pac_ssl_encoder_epoch1.pt")
EPOCH3_CHECKPOINT = Path("data/pac_ssl_encoder.pt")
EPOCH1_CACHE_H5 = Path("data/preprocessed/encoder_features_z_epoch1.h5")
EPOCH3_CACHE_H5 = Path("data/preprocessed/encoder_features_z.h5")

TARGET_PATIENTS = ["chb20", "chb19", "chb03", "chb04", "chb02"]

def obtain_epoch1_checkpoint(device):
    if EPOCH1_CHECKPOINT.exists():
        print(f"[Item B] Epoch 1 checkpoint already exists at {EPOCH1_CHECKPOINT}. Skipping pretraining run.", flush=True)
        return

    print(f"\n[Item B] Running 1 epoch of Stage 2 pretraining on full dataset to obtain Epoch 1 checkpoint...", flush=True)
    dataset = MmapPreprocessedEEGDataset(patients=PATIENTS_ALL)
    n_total = len(dataset)
    n_val = int(n_total * 0.1)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    num_workers = 0
    train_sampler = BlockRandomSampler(train_ds, block_size=8192)
    train_loader = DataLoader(train_ds, batch_size=256, sampler=train_sampler, num_workers=num_workers, pin_memory=True if torch.cuda.is_available() else False)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=True if torch.cuda.is_available() else False)

    feature_extractor = PACFeatureExtractor(fs=256, n_samples=1024).to(device)
    encoder = PACSSLEncoder(fs=256, n_samples=1024, num_channels=18, cnn_out=32, d_model=128).to(device)
    model = PACSSLPretextModel(encoder=encoder, d_model=128).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print(f"Training Epoch 1 across {len(train_loader)} batches (batch_size=256)...", flush=True)
    train_loss, train_acc, train_time = run_training_epoch(model, train_loader, optimizer, criterion, device, feature_extractor)
    print(f"Validation Epoch 1 across {len(val_loader)} batches...", flush=True)
    val_loss, val_acc, val_time = run_validation(model, val_loader, criterion, device, feature_extractor)

    print(f"[Epoch 1 Result] Train Loss={train_loss:.4f}, Train Acc={train_acc*100:.2f}% | Val Loss={val_loss:.4f}, Val Acc={val_acc*100:.2f}%", flush=True)

    EPOCH1_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.encoder.state_dict(), EPOCH1_CHECKPOINT)
    print(f"Saved Epoch 1 checkpoint to {EPOCH1_CHECKPOINT}", flush=True)


def extract_features_for_checkpoint(checkpoint_path, cache_path, device):
    if cache_path.exists():
        print(f"[Item B] Features already cached at {cache_path}.", flush=True)
        return

    print(f"\n[Item B] Extracting 128-dim representations across all 23 subjects using {checkpoint_path}...", flush=True)
    feature_extractor = PACFeatureExtractor(fs=256, n_samples=1024).to(device)
    encoder = PACSSLEncoder(fs=256, n_samples=1024, num_channels=18, cnn_out=32, d_model=128).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(state_dict)
    encoder.eval()

    t0 = time.time()
    with h5py.File(cache_path, "w") as out_f:
        for p in PATIENTS_ALL:
            pre_path = NPY_ROOT / f"{p}_preictal.npy"
            inter_path = NPY_ROOT / f"{p}_interictal.npy"

            pre_data = np.load(pre_path, mmap_mode='r') if pre_path.exists() else np.zeros((0, 18, 1024), dtype=np.float32)
            inter_data = np.load(inter_path, mmap_mode='r') if inter_path.exists() else np.zeros((0, 18, 1024), dtype=np.float32)

            z_pre_list = []
            if len(pre_data) > 0:
                for idx in range(0, len(pre_data), 256):
                    batch = torch.from_numpy(pre_data[idx:idx+256].copy()).to(device)
                    with torch.no_grad():
                        feats = feature_extractor(batch)
                        z = encoder.forward_from_features(feats)
                        z_pre_list.append(z.cpu())
                z_pre = torch.cat(z_pre_list, dim=0)
            else:
                z_pre = torch.zeros((0, 128), dtype=torch.float32)

            z_inter_list = []
            if len(inter_data) > 0:
                for idx in range(0, len(inter_data), 256):
                    batch = torch.from_numpy(inter_data[idx:idx+256].copy()).to(device)
                    with torch.no_grad():
                        feats = feature_extractor(batch)
                        z = encoder.forward_from_features(feats)
                        z_inter_list.append(z.cpu())
                z_inter = torch.cat(z_inter_list, dim=0)
            else:
                z_inter = torch.zeros((0, 128), dtype=torch.float32)

            grp = out_f.create_group(p)
            grp.create_dataset("preictal", data=z_pre.numpy(), compression="gzip")
            grp.create_dataset("interictal", data=z_inter.numpy(), compression="gzip")
            print(f"  [{p}] Extracted {len(z_pre)} preictal + {len(z_inter)} interictal embeddings.", flush=True)

    print(f"Feature extraction complete in {time.time()-t0:.1f}s. Saved to {cache_path}", flush=True)


def load_features_dict(cache_path):
    features_dict = {}
    with h5py.File(cache_path, "r") as f:
        for p in PATIENTS_ALL:
            if p in f:
                features_dict[p] = {
                    "preictal": torch.from_numpy(f[p]["preictal"][:]).float(),
                    "interictal": torch.from_numpy(f[p]["interictal"][:]).float()
                }
    return features_dict


def run_lopo_on_targets(features_dict, device, targets=TARGET_PATIENTS):
    aucs = {}
    for test_p in targets:
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
        aucs[test_p] = compute_roc_auc_numpy(y_test, probs)
    return aucs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Pre-Stage-4 Item B: Checkpoint Analysis starting on {device} ===", flush=True)

    obtain_epoch1_checkpoint(device)
    extract_features_for_checkpoint(EPOCH1_CHECKPOINT, EPOCH1_CACHE_H5, device)

    print("\nLoading Epoch 1 and Epoch 3 features...", flush=True)
    dict_ep1 = load_features_dict(EPOCH1_CACHE_H5)
    dict_ep3 = load_features_dict(EPOCH3_CACHE_H5)

    print(f"\nRunning strict LOPO evaluation on target extremes ({TARGET_PATIENTS})...", flush=True)
    aucs_ep1 = run_lopo_on_targets(dict_ep1, device, TARGET_PATIENTS)
    aucs_ep3 = run_lopo_on_targets(dict_ep3, device, TARGET_PATIENTS)

    print("\n=====================================================================================", flush=True)
    print("=== ITEM B: COMPARISON TABLE (EPOCH 1 vs EPOCH 3 CHECKPOINT ON EXTREME SUBJECTS) ===", flush=True)
    print("=====================================================================================", flush=True)
    print(f"{'Patient':<10} | {'Epoch 1 AUC':<14} | {'Epoch 3 AUC':<14} | {'Delta (Ep1 - Ep3)':<18}", flush=True)
    print("-" * 62, flush=True)

    diffs = []
    for p in TARGET_PATIENTS:
        a1 = aucs_ep1.get(p, 0.5)
        a3 = aucs_ep3.get(p, 0.5)
        diff = a1 - a3
        diffs.append(diff)
        print(f"{p:<10} | {a1:<14.4f} | {a3:<14.4f} | {diff:<+18.4f}", flush=True)
    print("-" * 62, flush=True)
    print(f"{'Mean':<10} | {np.mean([aucs_ep1[p] for p in TARGET_PATIENTS]):<14.4f} | {np.mean([aucs_ep3[p] for p in TARGET_PATIENTS]):<14.4f} | {np.mean(diffs):<+18.4f}", flush=True)
    print("=====================================================================================\n", flush=True)

    if np.mean(diffs) > 0.02 or (aucs_ep1["chb20"] > aucs_ep3["chb20"] and aucs_ep1["chb03"] > aucs_ep3["chb03"]):
        print("[DECISION] Switching to Epoch 1 checkpoint (`pac_ssl_encoder_epoch1.pt`) as the encoder going into Stage 4 because it demonstrates superior generalizability and stability across responsive and challenging subjects.", flush=True)
    else:
        print("[DECISION] Keeping Epoch 3 checkpoint (`pac_ssl_encoder.pt`) as the encoder going into Stage 4 because Epoch 1 did not demonstrate meaningful or consistent AUC superiority across the evaluated extreme subjects.", flush=True)

if __name__ == "__main__":
    main()
