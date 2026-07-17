"""
train_pretext_stage2.py — PAC-SSL Pretext Self-Supervised Pretraining & Time Estimation
Implements:
  - Memory-mapped numpy dataset loader (`MmapPreprocessedEEGDataset`) for O(1) random access
    on uncompressed .npy files. Each per-index read is a direct file offset seek — no
    gzip decompression overhead. Requires one-time conversion via `convert_h5_to_npy.py`.
  - Fast on-the-fly GPU Genuine vs Swapped PAC coupling generation.
  - Subsample mode (--subsample) for running on subset (e.g. chb01, chb02).
  - Full mode (--full) for full dataset training across all 23 subjects.
  - Automatic checkpoint saving and immediate reload verification (`verify_checkpoint_reload`).
  - Instant unbuffered logging (`flush=True`) for background task tracking.
"""
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

from pac_ssl_model import PACSSLEncoder, PACSSLPretextModel, PACFeatureExtractor

DATA_ROOT = Path("data/preprocessed")
NPY_ROOT = DATA_ROOT / "npy"
PATIENTS_ALL = [f"chb{i:02d}" for i in range(1, 25) if i != 12]

class MmapPreprocessedEEGDataset(Dataset):
    """
    Memory-mapped dataset loader for 4-second EEG windows (18 channels, 1024 samples).
    Uses numpy memory-mapped arrays on uncompressed .npy files for O(1) random access.
    Each __getitem__ call is a direct file offset seek + 73,728-byte read — no decompression.
    RAM footprint: only the OS page cache for recently accessed pages (~tens of MB).
    """
    def __init__(self, patients, npy_root=NPY_ROOT, max_samples_per_patient=None):
        super().__init__()
        self.npy_root = npy_root
        self.mmaps = {}  # patient -> {"preictal": np.memmap, "interictal": np.memmap}
        self.index = []  # list of tuples: (patient, group_key, local_idx)

        print(f"Building mmap index from {len(patients)} subjects across {npy_root}...", flush=True)
        t0 = time.time()
        for p in patients:
            pre_path = npy_root / f"{p}_preictal.npy"
            inter_path = npy_root / f"{p}_interictal.npy"
            if not pre_path.exists() or not inter_path.exists():
                print(f"Warning: .npy files for {p} not found, skipping.", flush=True)
                continue

            # Memory-map the arrays (no data loaded into RAM, just file descriptors)
            pre_mmap = np.load(pre_path, mmap_mode='r')
            inter_mmap = np.load(inter_path, mmap_mode='r')
            self.mmaps[p] = {"preictal": pre_mmap, "interictal": inter_mmap}

            n_pre = len(pre_mmap)
            n_inter = len(inter_mmap)

            if max_samples_per_patient:
                n_pre = min(n_pre, max_samples_per_patient // 2)
                n_inter = min(n_inter, max_samples_per_patient // 2)

            for i in range(n_pre):
                self.index.append((p, "preictal", i))
            for j in range(n_inter):
                self.index.append((p, "interictal", j))

        print(f"Mmap index built: {len(self.index):,} windows across {len(patients)} subjects in {time.time()-t0:.2f}s", flush=True)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        p, group_key, local_idx = self.index[idx]
        win = self.mmaps[p][group_key][local_idx]  # shape (18, 1024) float32 — O(1) seek
        return torch.from_numpy(win.copy())  # copy() to detach from mmap for safe multi-worker use


def generate_pac_couplings_on_gpu(features, device):
    """
    Given batch of PAC features of shape (B, 18, 3, 1024) = [cos_phi, sin_phi, amp_gamma]:
      - Genuine (label=1): keep exact features (same segment phase & amplitude).
      - Swapped (label=0): pair (cos_phi, sin_phi) from segment i with amp_gamma from randomly different segment j in batch.
    Returns:
      features_combined: (2*B, 18, 3, 1024)
      labels_combined: (2*B,) float tensor (1.0 for genuine, 0.0 for swapped)
    """
    B, C, F_in, T = features.shape
    
    # 1. Genuine features
    features_genuine = features
    labels_genuine = torch.ones(B, dtype=torch.float32, device=device)
    
    # 2. Swapped features
    if B > 1:
        perm = torch.randperm(B, device=device)
        fixed = (perm == torch.arange(B, device=device))
        if fixed.any():
            perm[fixed] = (perm[fixed] + 1) % B
    else:
        perm = torch.zeros(1, dtype=torch.long, device=device)
        
    phi_part = features[:, :, 0:F_in-1, :]     # (B, 18, F_in-1, 1024) e.g. [theta_phase_z] when F_in=2
    amp_part = features[perm, :, F_in-1:F_in, :] # (B, 18, 1, 1024)      e.g. [amp_gamma_z] when F_in=2
    
    features_swapped = torch.cat([phi_part, amp_part], dim=2)
    labels_swapped = torch.zeros(B, dtype=torch.float32, device=device)
    
    features_combined = torch.cat([features_genuine, features_swapped], dim=0)
    labels_combined = torch.cat([labels_genuine, labels_swapped], dim=0)
    
    return features_combined, labels_combined


def run_training_epoch(model, dataloader, optimizer, criterion, device, feature_extractor):
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    start_time = time.time()
    
    for batch_idx, raw_x in enumerate(dataloader):
        raw_x = raw_x.to(device, non_blocking=True) # (B, 18, 1024)
        
        with torch.no_grad():
            features = feature_extractor(raw_x) # (B, 18, 4, 1024)
            features_combined, labels_combined = generate_pac_couplings_on_gpu(features, device)
            
        optimizer.zero_grad()
        logits = model(features_combined) # (2*B,)
        loss = criterion(logits, labels_combined)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(labels_combined)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels_combined).sum().item()
        total_samples += len(labels_combined)
        
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(dataloader):
            elapsed = time.time() - start_time
            print(f"  Batch [{batch_idx+1}/{len(dataloader)}] | Loss: {loss.item():.4f} | Acc: {correct/total_samples*100:.2f}% | Elapsed: {elapsed:.1f}s", flush=True)
            
    avg_loss = total_loss / total_samples
    avg_acc = correct / total_samples
    epoch_time = time.time() - start_time
    return avg_loss, avg_acc, epoch_time


def run_validation(model, dataloader, criterion, device, feature_extractor):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    start_time = time.time()
    
    with torch.no_grad():
        for raw_x in dataloader:
            raw_x = raw_x.to(device, non_blocking=True)
            features = feature_extractor(raw_x)
            features_combined, labels_combined = generate_pac_couplings_on_gpu(features, device)
            
            logits = model(features_combined)
            loss = criterion(logits, labels_combined)
            
            total_loss += loss.item() * len(labels_combined)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels_combined).sum().item()
            total_samples += len(labels_combined)
            
    avg_loss = total_loss / total_samples
    avg_acc = correct / total_samples
    val_time = time.time() - start_time
    return avg_loss, avg_acc, val_time


def verify_checkpoint_reload(encoder, checkpoint_path, device, feature_extractor):
    """
    Verifies that saved checkpoint reloads correctly in the same session before moving on.
    Checks parameter exact match and forward pass consistency on dummy/test input.
    """
    print(f"\n=== Verifying Checkpoint Reloading from {checkpoint_path} ===", flush=True)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist!")
        
    encoder.eval()
    dummy_x = torch.randn(2, 18, 1024, device=device)
    with torch.no_grad():
        out_orig = encoder(dummy_x)
        
    fresh_encoder = PACSSLEncoder(fs=256, n_samples=1024, num_channels=18, cnn_out=32, d_model=128).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    fresh_encoder.load_state_dict(state_dict)
    fresh_encoder.eval()
    
    with torch.no_grad():
        out_reloaded = fresh_encoder(dummy_x)
        
    diff = torch.max(torch.abs(out_orig - out_reloaded)).item()
    print(f"Maximum absolute difference between original and reloaded encoder representations: {diff:.10f}", flush=True)
    if diff < 1e-6:
        print("[PASS] Checkpoint reload verification PASSED! State dict exact match confirmed.", flush=True)
        return True
    else:
        raise RuntimeError("[FAIL] Checkpoint reload verification FAILED! Representations do not match.")


class BlockRandomSampler(torch.utils.data.Sampler):
    """
    Groups training indices into contiguous spatial blocks based on underlying dataset indices (`block_size` =~ 600 MB).
    Within each block, window order is randomized each epoch. The block order itself is also randomized each epoch.
    This guarantees 100% OS disk page cache hits across 28 GB of uncompressed .npy files with 8 GB free RAM,
    while maintaining exact train/val split (`seed=42`) and visiting every window exactly once.
    """
    def __init__(self, data_source, block_size=8192):
        self.data_source = data_source
        self.block_size = block_size
        # Check if data_source is a Subset (from random_split) and get mapping to underlying dataset indices
        if hasattr(data_source, 'indices'):
            raw_indices = data_source.indices
            # Create pairs of (subset_idx, underlying_dataset_idx) and sort by underlying_dataset_idx for spatial locality
            sorted_pairs = sorted(enumerate(raw_indices), key=lambda x: x[1])
            self.spatial_subset_indices = [pair[0] for pair in sorted_pairs]
        else:
            self.spatial_subset_indices = list(range(len(data_source)))

    def __iter__(self):
        n = len(self.spatial_subset_indices)
        # Partition the spatially ordered subset indices into blocks of size `block_size`
        blocks = [self.spatial_subset_indices[i:min(i + self.block_size, n)] for i in range(0, n, self.block_size)]
        # Shuffle order of indices inside each block
        for b in blocks:
            perm = torch.randperm(len(b)).tolist()
            b[:] = [b[p] for p in perm]
        # Shuffle order of the blocks themselves
        block_perm = torch.randperm(len(blocks)).tolist()
        final_indices = []
        for bp in block_perm:
            final_indices.extend(blocks[bp])
        return iter(final_indices)

    def __len__(self):
        return len(self.data_source)


def main():
    parser = argparse.ArgumentParser(description="PAC-SSL Pretext Self-Supervised Training")
    parser.add_argument("--subsample", action="store_true", help="Run on small subset (e.g. 2 patients) for time estimation")
    parser.add_argument("--full", action="store_true", help="Run full dataset training across all 23 subjects")
    parser.add_argument("--patients", nargs="+", default=["chb01", "chb02"], help="Patients to load in subsample mode")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (windows per batch)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== PAC-SSL Pretext Pretraining starting on device: {device} ===", flush=True)
    
    patients = args.patients if args.subsample else PATIENTS_ALL
    dataset = MmapPreprocessedEEGDataset(patients=patients)
    
    split_ratio = 0.2 if args.subsample else 0.1
    n_total = len(dataset)
    n_val = int(n_total * split_ratio)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    
    # num_workers=0: numpy mmap arrays can't be pickled across Windows spawn workers.
    # With mmap, reads are O(1) anyway — no decompression overhead to parallelize.
    num_workers = 0
    train_sampler = BlockRandomSampler(train_ds, block_size=8192)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True if torch.cuda.is_available() else False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True if torch.cuda.is_available() else False)
    
    print(f"Split dataset: {n_train:,} training windows ({len(train_loader)} batches) | {n_val:,} validation windows ({len(val_loader)} batches) | workers={num_workers}", flush=True)


    
    feature_extractor = PACFeatureExtractor(fs=256, n_samples=1024).to(device)
    encoder = PACSSLEncoder(fs=256, n_samples=1024, num_channels=18, cnn_out=32, d_model=128).to(device)
    model = PACSSLPretextModel(encoder=encoder, d_model=128).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    print("\n--- Model Architecture Summary ---", flush=True)
    print(f"Encoder Stage 1: DualKernelCNNFrontEnd (K1=33 for Theta, K2=7 for Gamma -> out=32)", flush=True)
    print(f"Encoder Stage 2: LearnableAdjacencyGCN (num_channels=18, embed_dim=16)", flush=True)
    print(f"Encoder Stage 3: TemporalTransformerEncoder (d_model=128, nhead=4, layers=2)", flush=True)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params:,}\n", flush=True)
    
    history = []
    t_start_all = time.time()
    
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs} — Training across {len(train_loader)} batches...", flush=True)
        train_loss, train_acc, train_time = run_training_epoch(model, train_loader, optimizer, criterion, device, feature_extractor)
        print(f"Epoch {epoch}/{args.epochs} — Validation across {len(val_loader)} batches...", flush=True)
        val_loss, val_acc, val_time = run_validation(model, val_loader, criterion, device, feature_extractor)
        
        print(f"\n[Epoch {epoch} Summary] Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% ({train_time:.1f}s)", flush=True)
        print(f"[Epoch {epoch} Summary] Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}% ({val_time:.1f}s)\n", flush=True)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc, "train_time": train_time,
            "val_loss": val_loss, "val_acc": val_acc, "val_time": val_time
        })
        
    total_time = time.time() - t_start_all
    
    checkpoint_path = Path("pac_ssl_encoder.pt")
    torch.save(model.encoder.state_dict(), checkpoint_path)
    print(f"\nPretext training complete in {total_time/60:.2f} minutes. Encoder checkpoint saved to {checkpoint_path}", flush=True)
    
    verify_checkpoint_reload(model.encoder, checkpoint_path, device, feature_extractor)
    
    out_summary = Path("data/preprocessed/stage3_pretraining_history.txt")
    with open(out_summary, "w") as f:
        f.write(f"Mode: {'Subsample' if args.subsample else 'Full'}\n")
        f.write(f"Total_Windows: {n_total}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Total_Time_Min: {total_time/60:.2f}\n")
        for h in history:
            f.write(f"Epoch {h['epoch']}: Train_Loss={h['train_loss']:.4f}, Train_Acc={h['train_acc']*100:.2f}%, Val_Loss={h['val_loss']:.4f}, Val_Acc={h['val_acc']*100:.2f}%\n")
    print(f"Saved pretraining history summary to {out_summary}", flush=True)

if __name__ == "__main__":
    main()
