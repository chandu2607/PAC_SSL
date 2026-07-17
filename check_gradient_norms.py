"""
check_gradient_norms.py — Diagnose Dead Gradients in Current Architecture
Step 1: Runs 1 forward + 1 backward pass on a real training batch and prints
gradient norms for every layer from input to output along with NaN/Inf checks.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pac_ssl_model import PACSSLEncoder, PACSSLPretextModel, PACFeatureExtractor
from train_pretext_stage2 import MmapPreprocessedEEGDataset, generate_pac_couplings_on_gpu, PATIENTS_ALL

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Step 1 Diagnosis] Device: {device}")

    # 1. Load data
    print("[Step 1 Diagnosis] Loading 1 batch from real training data...")
    dataset = MmapPreprocessedEEGDataset(patients=PATIENTS_ALL[:3]) # use first 3 patients for quick sampling
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    raw_batch = next(iter(loader)).to(device)

    # 2. Initialize current architecture
    print("[Step 1 Diagnosis] Initializing current DualKernelCNNFrontEnd + LearnableAdjacencyGCN + Transformer architecture...")
    fe = PACFeatureExtractor(fs=256, n_samples=1024).to(device)
    encoder = PACSSLEncoder(fs=256, n_samples=1024, num_channels=18, cnn_out=32, d_model=128).to(device)
    model = PACSSLPretextModel(encoder=encoder, d_model=128).to(device)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    model.zero_grad()

    # 3. Forward + Backward pass
    print("[Step 1 Diagnosis] Running 1 forward + 1 backward pass...")
    with torch.no_grad():
        feats = fe(raw_batch)
        xc, yc = generate_pac_couplings_on_gpu(feats, device)

    logits = model(xc)
    loss = criterion(logits, yc)
    loss.backward()

    print(f"\nForward Loss: {loss.item():.6f}")
    print("\nLayer-by-Layer Gradient Norm Diagnosis (from input to output):")
    print("-" * 85)
    print(f"{'Layer / Parameter Name':<50} {'Shape':<20} {'Grad Norm':<12} {'NaN?':<6} {'Inf?':<6}")
    print("-" * 85)

    has_dead_or_nan = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            shape_str = str(list(param.shape))
            if param.grad is not None:
                g_norm = param.grad.norm().item()
                is_nan = torch.isnan(param.grad).any().item()
                is_inf = torch.isinf(param.grad).any().item()
                print(f"{name:<50} {shape_str:<20} {g_norm:<12.4e} {str(is_nan):<6} {str(is_inf):<6}")
                if g_norm < 1e-8 or is_nan or is_inf:
                    has_dead_or_nan = True
            else:
                print(f"{name:<50} {shape_str:<20} {'NONE':<12} {'True':<6} {'False':<6}")
                has_dead_or_nan = True

    print("-" * 85)
    if has_dead_or_nan:
        print("[Step 1 Diagnosis Result] CONFIRMED: Dead gradients (< 1e-8), zero gradients, or NaN/NONE detected in early layers/parameters!")
    else:
        print("[Step 1 Diagnosis Result] All layers show non-dead gradients above 1e-8.")

if __name__ == "__main__":
    main()
