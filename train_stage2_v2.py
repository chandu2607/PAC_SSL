"""
train_stage2_v2.py — Improved PAC-SSL Pretraining
Improvements over v1:
  - 20 epochs with CosineAnnealingLR scheduler
  - Best-epoch checkpoint saved by val accuracy (not last epoch)
  - Rich progress output: epoch %, ETA, LR tracking
  - Saves training history to data/preprocessed/stage2_v2_history.txt
"""
import os
import sys
import time
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

from pac_ssl_model import PACSSLEncoder, PACSSLPretextModel, PACFeatureExtractor
from train_pretext_stage2 import (
    MmapPreprocessedEEGDataset, BlockRandomSampler,
    run_training_epoch, run_validation, PATIENTS_ALL
)

CHECKPOINT_BEST  = Path("data/pac_ssl_encoder_best.pt")
CHECKPOINT_FINAL = Path("data/pac_ssl_encoder_v2.pt")
HISTORY_OUT      = Path("data/preprocessed/stage2_v2_history.txt")

BAR_WIDTH = 40

def progress_bar(current, total, prefix="", suffix="", bar_char="█", empty_char="░"):
    pct = current / total
    filled = int(BAR_WIDTH * pct)
    bar = bar_char * filled + empty_char * (BAR_WIDTH - filled)
    print(f"\r{prefix} [{bar}] {pct*100:5.1f}%  {suffix}", end="", flush=True)

def fmt_time(secs):
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    elif m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"

def run_epoch_with_progress(model, loader, optimizer, criterion, device, fe, scheduler=None, train=True):
    model.train(train)
    total_loss, correct, total_samples = 0.0, 0, 0
    t0 = time.time()
    n_batches = len(loader)

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for i, raw_x in enumerate(loader, 1):
            raw_x = raw_x.to(device, non_blocking=True)

            with torch.no_grad():
                from train_pretext_stage2 import generate_pac_couplings_on_gpu
                feats = fe(raw_x)
                xc, yc = generate_pac_couplings_on_gpu(feats, device)

            if train:
                optimizer.zero_grad()
                logits = model(xc)
                loss   = criterion(logits, yc)
                loss.backward()
                optimizer.step()
            else:
                logits = model(xc)
                loss   = criterion(logits, yc)

            total_loss    += loss.item() * len(yc)
            correct       += ((torch.sigmoid(logits) >= 0.5).float() == yc).sum().item()
            total_samples += len(yc)

            elapsed = time.time() - t0
            eta     = elapsed / i * (n_batches - i) if i > 0 else 0
            phase   = "Train" if train else "Val"
            suffix  = (f"Batch {i}/{n_batches} | Loss {total_loss/total_samples:.4f} | "
                       f"Acc {correct/total_samples*100:.1f}% | ETA {fmt_time(eta)}")
            progress_bar(i, n_batches, prefix=f"  {phase}", suffix=suffix)

    print()  # newline after bar
    if train and scheduler is not None:
        scheduler.step()

    return total_loss / total_samples, correct / total_samples, time.time() - t0


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--batch_size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--patients",   nargs="+",  default=None)
    args = parser.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patients = args.patients if args.patients else PATIENTS_ALL

    print("=" * 65, flush=True)
    print(" PAC-SSL v2 — Improved Pretraining with Cosine LR Scheduler", flush=True)
    print("=" * 65, flush=True)
    print(f" Device   : {device}", flush=True)
    print(f" Patients : {len(patients)} subjects", flush=True)
    print(f" Epochs   : {args.epochs}", flush=True)
    print(f" Batch    : {args.batch_size}", flush=True)
    print(f" LR Init  : {args.lr}  (CosineAnnealingLR -> 0)", flush=True)
    print("=" * 65, flush=True)

    dataset  = MmapPreprocessedEEGDataset(patients=patients)
    n_total  = len(dataset)
    n_val    = int(n_total * 0.1)
    n_train  = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=BlockRandomSampler(train_ds, 8192),
                              num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=pin)

    print(f"\n Dataset  : {n_total:,} windows -> {n_train:,} train / {n_val:,} val", flush=True)
    print(f" Batches  : {len(train_loader)} train / {len(val_loader)} val per epoch\n", flush=True)

    fe      = PACFeatureExtractor(fs=256, n_samples=1024).to(device)
    encoder = PACSSLEncoder(fs=256, n_samples=1024, num_channels=18, cnn_out=32, d_model=128).to(device)
    model   = PACSSLPretextModel(encoder=encoder, d_model=128).to(device)
    total_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" Parameters: {total_p:,}\n", flush=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    history      = []
    best_val_acc = 0.0
    t_all        = time.time()

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed_all = time.time() - t_all
        eta_all     = elapsed_all / (epoch - 1) * (args.epochs - epoch + 1) if epoch > 1 else 0

        print(f"\n{'─'*65}", flush=True)
        print(f" Epoch {epoch:2d}/{args.epochs}  |  LR={current_lr:.2e}  |  "
              f"Overall ETA: {fmt_time(eta_all)}", flush=True)
        print(f"{'─'*65}", flush=True)
        progress_bar(epoch - 1, args.epochs,
                     prefix="  Pipeline",
                     suffix=f"Epoch {epoch}/{args.epochs} starting...")
        print(flush=True)

        tr_loss, tr_acc, tr_t = run_epoch_with_progress(model, train_loader, optimizer,
                                                        criterion, device, fe,
                                                        scheduler=scheduler, train=True)
        va_loss, va_acc, va_t = run_epoch_with_progress(model, val_loader, optimizer,
                                                        criterion, device, fe,
                                                        scheduler=None, train=False)

        is_best = va_acc > best_val_acc
        if is_best:
            best_val_acc = va_acc
            torch.save(model.encoder.state_dict(), CHECKPOINT_BEST)
            best_tag = " <- NEW BEST [*]"
        else:
            best_tag = ""

        print(f"\n  [Ep {epoch:2d}] Train: loss={tr_loss:.4f} acc={tr_acc*100:.2f}%  ({fmt_time(tr_t)})", flush=True)
        print(f"  [Ep {epoch:2d}]   Val: loss={va_loss:.4f} acc={va_acc*100:.2f}%  ({fmt_time(va_t)}){best_tag}", flush=True)
        print(f"  [Ep {epoch:2d}] Best val so far: {best_val_acc*100:.2f}%", flush=True)

        progress_bar(epoch, args.epochs, prefix="  Pipeline",
                     suffix=f"Epoch {epoch}/{args.epochs} done | Best val: {best_val_acc*100:.2f}%")
        print(flush=True)

        history.append({
            "epoch": epoch, "lr": current_lr,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss":   va_loss, "val_acc":   va_acc,
            "is_best":    is_best
        })

    total_t = time.time() - t_all
    torch.save(model.encoder.state_dict(), CHECKPOINT_FINAL)

    print(f"\n{'='*65}", flush=True)
    print(f" Training complete in {fmt_time(total_t)}", flush=True)
    print(f" Best checkpoint : {CHECKPOINT_BEST}  (val acc {best_val_acc*100:.2f}%)", flush=True)
    print(f" Final checkpoint: {CHECKPOINT_FINAL}", flush=True)
    print(f"{'='*65}", flush=True)

    # Save history
    HISTORY_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_OUT, "w") as f:
        f.write(f"Epochs: {args.epochs}  |  BatchSize: {args.batch_size}  |  LR_init: {args.lr}\n")
        f.write(f"Total_Time: {fmt_time(total_t)}\n")
        f.write(f"Best_Val_Epoch: {next(h['epoch'] for h in history if h['is_best'])}\n")
        f.write(f"Best_Val_Acc: {best_val_acc*100:.2f}%\n\n")
        f.write(f"{'Epoch':<6} {'LR':<10} {'TrLoss':<10} {'TrAcc':<10} {'VaLoss':<10} {'VaAcc':<10} {'Best'}\n")
        for h in history:
            f.write(f"{h['epoch']:<6} {h['lr']:<10.2e} {h['train_loss']:<10.4f} "
                    f"{h['train_acc']*100:<10.2f} {h['val_loss']:<10.4f} "
                    f"{h['val_acc']*100:<10.2f} {'[BEST]' if h['is_best'] else ''}\n")
    print(f" History saved to {HISTORY_OUT}", flush=True)

if __name__ == "__main__":
    main()
