"""
run_epoch1_smoke_test.py — Step 3 Smoke Test (1 Epoch on 3 Patients)
Runs 1 epoch of pretraining with the reverted simple 3-layer 1D CNN encoder on 3 patients (`PATIENTS_ALL[:3]`).
Logs loss and accuracy during the first ~20 batches and at the end of epoch 1 to verify clean gradient flow and movement from 0.6931 / 50%.
"""
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pac_ssl_model import PACSSLEncoder, PACSSLPretextModel, PACFeatureExtractor
from train_pretext_stage2 import MmapPreprocessedEEGDataset, BlockRandomSampler, generate_pac_couplings_on_gpu, PATIENTS_ALL

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Step 3 Smoke Test] Device: {device}")
    
    patients = PATIENTS_ALL[:3]
    print(f"[Step 3 Smoke Test] Using 3 subjects: {patients}")
    
    dataset = MmapPreprocessedEEGDataset(patients=patients)
    n_total = len(dataset)
    n_val   = int(n_total * 0.1)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_ds, batch_size=256, sampler=BlockRandomSampler(train_ds, 8192), num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=0)
    
    print(f"[Step 3 Smoke Test] Train set: {n_train:,} windows ({len(train_loader)} batches), Val set: {n_val:,} windows")
    
    fe = PACFeatureExtractor(fs=256, n_samples=1024).to(device)
    encoder = PACSSLEncoder(fs=256, n_samples=1024, num_channels=18, cnn_out=64, d_model=128).to(device)
    model = PACSSLPretextModel(encoder=encoder, d_model=128).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    print("\n[Step 3 Smoke Test] Starting Epoch 1/1...")
    model.train()
    total_loss, correct, total_samples = 0.0, 0, 0
    t0 = time.time()
    
    early_logs = []
    
    for i, raw_x in enumerate(train_loader, 1):
        raw_x = raw_x.to(device, non_blocking=True)
        with torch.no_grad():
            feats = fe(raw_x)
            xc, yc = generate_pac_couplings_on_gpu(feats, device)
            
        optimizer.zero_grad()
        logits = model(xc)
        loss = criterion(logits, yc)
        loss.backward()
        optimizer.step()
        
        batch_loss = loss.item()
        batch_acc  = ((torch.sigmoid(logits) >= 0.5).float() == yc).float().mean().item() * 100.0
        
        total_loss += batch_loss * len(yc)
        correct    += ((torch.sigmoid(logits) >= 0.5).float() == yc).sum().item()
        total_samples += len(yc)
        
        if i <= 20 or i % 20 == 0 or i == len(train_loader):
            msg = f"  Batch {i:3d}/{len(train_loader)} | Batch Loss: {batch_loss:.4f} | Batch Acc: {batch_acc:5.1f}% | Running Mean Loss: {total_loss/total_samples:.4f} | Running Mean Acc: {correct/total_samples*100:5.1f}%"
            print(msg, flush=True)
            if i <= 20:
                early_logs.append((i, batch_loss, batch_acc))
                
    end_train_loss = total_loss / total_samples
    end_train_acc  = correct / total_samples * 100.0
    
    # Validation
    model.eval()
    val_loss, val_correct, val_samples = 0.0, 0, 0
    with torch.no_grad():
        for raw_x in val_loader:
            raw_x = raw_x.to(device, non_blocking=True)
            feats = fe(raw_x)
            xc, yc = generate_pac_couplings_on_gpu(feats, device)
            logits = model(xc)
            loss = criterion(logits, yc)
            val_loss += loss.item() * len(yc)
            val_correct += ((torch.sigmoid(logits) >= 0.5).float() == yc).sum().item()
            val_samples += len(yc)
            
    end_val_loss = val_loss / val_samples
    end_val_acc  = val_correct / val_samples * 100.0
    
    print("\n" + "="*75)
    print(" STEP 3 SMOKE TEST RESULTS (1 EPOCH ON 3 PATIENTS)")
    print("="*75)
    print(f" START of Epoch 1 (First 5 Batches Average):")
    early_5_loss = sum(x[1] for x in early_logs[:5]) / min(len(early_logs), 5)
    early_5_acc  = sum(x[2] for x in early_logs[:5]) / min(len(early_logs), 5)
    print(f"   Avg Loss: {early_5_loss:.4f} | Avg Acc: {early_5_acc:.2f}%")
    
    print(f"\n START of Epoch 1 (First 20 Batches Average):")
    early_20_loss = sum(x[1] for x in early_logs[:20]) / min(len(early_logs), 20)
    early_20_acc  = sum(x[2] for x in early_logs[:20]) / min(len(early_logs), 20)
    print(f"   Avg Loss: {early_20_loss:.4f} | Avg Acc: {early_20_acc:.2f}%")
    
    print(f"\n END of Epoch 1 (All {len(train_loader)} Batches):")
    print(f"   Train Loss: {end_train_loss:.4f} | Train Acc: {end_train_acc:.2f}%")
    print(f"   Val Loss:   {end_val_loss:.4f} | Val Acc:   {end_val_acc:.2f}% ({fmt_time(time.time()-t0)})")
    print("="*75)
    
    # Check for movement
    if abs(end_train_loss - 0.6931) > 0.001 or abs(end_train_acc - 50.0) > 0.5:
        print("[Step 3 Result] CONFIRMED: Loss and accuracy have MOVED away from 0.6931 / 50.00%!")
    else:
        print("[Step 3 Result] WARNING: Loss and accuracy remained near 0.6931 / 50.00%.")

def fmt_time(secs):
    return f"{int(secs//60)}m{int(secs%60):02d}s"

if __name__ == "__main__":
    main()
