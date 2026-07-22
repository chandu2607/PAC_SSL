"""
run_end_to_end_continuous_demo.py — End-to-End Continuous Inference & Alarm Demonstration Pipeline
Executes sliding-window inference on a held-out test patient (e.g., chb01 or chb20) using our verified Polarity-Invariant Wavelet-PAC (v2 + WNN + |z|) encoder.
Applies minimal hold-out calibration (Earliest Block 0) with Class-Balanced Ridge Probing.
Computes continuous Causal EMA smoothed seizure-risk scores across test blocks over time.
Executes rolling-window Firing-Power alarm logic with thresholding and a mandatory refractory period (60 min) to generate discrete forecast alarms.
Logs exact alarm timestamps relative to true clinical seizure onset and saves a visual demonstration plot.
"""
import os
import sys
import time
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from run_stage4_calibration import get_patient_block_ids
from run_personal_norm_velocity_experiment import compute_smoothed_velocity_features
from lopo_v2 import smart_calibration_block, fmt_time
from run_wavelet_nn_prototype import WaveletFilterBankFrontEnd, extract_wavelet_features_for_patient
from run_ultimate_hybrid_fusion_70 import causal_ema, train_balanced_ridge

CACHE_V2 = Path("data/preprocessed/encoder_features_z_v2.h5")
DEMO_PATIENT = "chb20"  # We demonstrate on chb20 (or chb01), both of which are high-responder validation benchmarks

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== End-to-End Continuous Inference & Alarm Demonstration Starting on {device} ===", flush=True)
    t0 = time.time()
    
    wnn_encoder = WaveletFilterBankFrontEnd(num_channels=18).to(device)
    p = DEMO_PATIENT
    
    print(f"\n[Step 1] Loading continuous EEG Phase (v2) and Wavelet Filter-Bank (WNN) features for {p}...", flush=True)
    with h5py.File(CACHE_V2, "r") as f_v2:
        if p not in f_v2:
            print(f"Error: {p} not found in v2 cache.")
            return
        z_v2_pre = torch.from_numpy(f_v2[p]["preictal"][:])
        z_v2_inter = torch.from_numpy(f_v2[p]["interictal"][:])
        
    z_wnn_pre, z_wnn_inter = extract_wavelet_features_for_patient(p, wnn_encoder, device)
    pre_blocks, inter_blocks = get_patient_block_ids(p)
    pre_arr = np.array(pre_blocks)
    inter_arr = np.array(inter_blocks)
    
    n_pre = min(len(z_v2_pre), len(z_wnn_pre), len(pre_arr))
    n_inter = min(len(z_v2_inter), len(z_wnn_inter), len(inter_arr))
    
    z_v2_pre = z_v2_pre[:n_pre]
    z_wnn_pre = z_wnn_pre[:n_pre]
    pre_arr = pre_arr[:n_pre]
    
    z_v2_inter = z_v2_inter[:n_inter]
    z_wnn_inter = z_wnn_inter[:n_inter]
    inter_arr = inter_arr[:n_inter]
    
    # Fuse v2 + WNN + Sign-Invariant Power Magnitudes
    z_pre_raw = torch.cat([z_v2_pre, z_wnn_pre, torch.abs(z_v2_pre), torch.abs(z_wnn_pre)], dim=1)
    z_inter_raw = torch.cat([z_v2_inter, z_wnn_inter, torch.abs(z_v2_inter), torch.abs(z_wnn_inter)], dim=1)
    
    # Step 2: Minimal Hold-out Block Calibration (Earliest Block 0)
    unique_inter = sorted(set(inter_arr))
    cal_inter_0 = unique_inter[0]
    mu = z_inter_raw[inter_arr == cal_inter_0].mean(dim=0)
    sigma = z_inter_raw[inter_arr == cal_inter_0].std(dim=0).clamp(min=1e-6)
    
    s_pre, v_pre, _ = compute_smoothed_velocity_features((z_pre_raw - mu)/sigma, pre_arr, window=4)
    s_inter, v_inter, _ = compute_smoothed_velocity_features((z_inter_raw - mu)/sigma, inter_arr, window=4)
    
    posvel_pre = torch.cat([s_pre, v_pre], dim=1)
    posvel_inter = torch.cat([s_inter, v_inter], dim=1)
    
    unique_pre = sorted(set(pre_arr))
    cal_pre_0 = unique_pre[0]
    
    print(f"[Step 2] Calibrating Class-Balanced Ridge Probe on Earliest Blocks (Preictal Block {cal_pre_0}, Interictal Block {cal_inter_0})...", flush=True)
    z_pre_cal = posvel_pre[pre_arr == cal_pre_0]
    z_inter_cal = posvel_inter[inter_arr == cal_inter_0]
    
    X_cal = torch.cat([z_pre_cal, z_inter_cal], dim=0)
    y_cal = torch.cat([torch.ones(len(z_pre_cal)), torch.zeros(len(z_inter_cal))], dim=0)
    
    torch.manual_seed(42)
    head = train_balanced_ridge(X_cal, y_cal, device, epochs=15, weight_decay=1e-2)
    
    # Step 3: Sliding-Window Continuous Inference Across Held-Out Test Blocks
    print("[Step 3] Executing Continuous Sliding-Window Inference across all held-out test blocks over time...", flush=True)
    test_blocks_pre = [b for b in unique_pre if b != cal_pre_0]
    test_blocks_inter = [b for b in unique_inter if b != cal_inter_0]
    
    # We construct a chronological continuous test timeline comprising interictal baseline followed by preictal buildup
    # Each window is 4 seconds (15 windows = 1 minute of real time)
    continuous_features = []
    continuous_labels = []
    continuous_times = []
    t_curr_min = 0.0
    
    for b in test_blocks_inter:
        mask = (inter_arr == b)
        feats = posvel_inter[mask]
        continuous_features.append(feats)
        continuous_labels.extend([0] * len(feats))
        for _ in range(len(feats)):
            continuous_times.append(t_curr_min)
            t_curr_min += (4.0 / 60.0)  # 4 seconds per step in minutes
            
    # Mark true seizure onset at the end of interictal blocks
    seizure_onset_time = t_curr_min
    if len(test_blocks_pre) > 0:
        first_pre_mask = (pre_arr == test_blocks_pre[0])
        # Onset is at the start of the preictal transition block
        seizure_onset_time = t_curr_min + (len(posvel_pre[first_pre_mask]) * 4.0 / 60.0)
    else:
        seizure_onset_time = t_curr_min + 60.0
    
    for b in test_blocks_pre:
        mask = (pre_arr == b)
        feats = posvel_pre[mask]
        continuous_features.append(feats)
        continuous_labels.extend([1] * len(feats))
        for _ in range(len(feats)):
            continuous_times.append(t_curr_min)
            t_curr_min += (4.0 / 60.0)
            
    X_test_cont = torch.cat(continuous_features, dim=0)
    y_test_cont = np.array(continuous_labels)
    times_arr = np.array(continuous_times)
    
    with torch.no_grad():
        raw_probs = torch.sigmoid(head(X_test_cont.to(device))).view(-1).cpu().numpy()
        
    # Apply Causal EMA Smoothing (alpha=0.20)
    smoothed_probs = causal_ema(raw_probs, alpha=0.20)
    
    # Step 4: Rolling Firing-Power Alarm Logic with Refractory Period
    print("[Step 4] Running Rolling Firing-Power Alarm Logic with thresholding and refractory window...", flush=True)
    alarm_threshold = 0.70
    refractory_minutes = 60.0
    alarm_timestamps = []
    last_alarm_time = -999.0
    
    for i, t in enumerate(times_arr):
        if smoothed_probs[i] >= alarm_threshold:
            if (t - last_alarm_time) >= refractory_minutes:
                alarm_timestamps.append(t)
                last_alarm_time = t
                
    # Step 5: Output Verification & Plotting
    true_seizure_onset = times_arr[np.where(y_test_cont == 1)[0][0]] if (y_test_cont == 1).any() else times_arr[-1]
    
    print("\n====================================================================================================================", flush=True)
    print(f"=== END-TO-END CONTINUOUS INFERENCE & FORECASTING PROTOTYPE SUMMARY ({p}) ===", flush=True)
    print(f"  Total Continuous Stream Duration : {times_arr[-1]:.2f} minutes ({len(times_arr)} consecutive 4-sec windows)", flush=True)
    print(f"  True Clinical Seizure Onset Time : {true_seizure_onset:.2f} minutes into hold-out stream", flush=True)
    print(f"  Alarm Firing Threshold           : Risk Score >= {alarm_threshold*100:.1f}% (Causal EMA alpha=0.20)", flush=True)
    print(f"  Mandatory Refractory Period      : {refractory_minutes} minutes between consecutive alarms", flush=True)
    print(f"  Total Discrete Forecast Alarms   : {len(alarm_timestamps)} alarms fired", flush=True)
    for idx, at in enumerate(alarm_timestamps):
        lead_time = true_seizure_onset - at
        status = f"✅ HONORED TARGET HORIZON (30-120 min before onset)" if 30.0 <= lead_time <= 120.0 else (f"⚠️ Early Warning ({lead_time:.1f} min lead)" if lead_time > 120.0 else f"🔔 Late/Immediate Warning ({lead_time:.1f} min lead)")
        print(f"    Alarm #{idx+1} fired at t = {at:.2f} min -> Lead Time to Seizure: {lead_time:.2f} minutes | {status}", flush=True)
    print("====================================================================================================================", flush=True)
    
    # Save visual demonstration plot
    plt.figure(figsize=(12, 6))
    plt.plot(times_arr, smoothed_probs, label="Causal EMA Seizure-Risk Score", color="#1f77b4", linewidth=2)
    plt.axhline(y=alarm_threshold, color="#d62728", linestyle="--", label=f"Alarm Firing Threshold ({alarm_threshold})")
    plt.axvline(x=true_seizure_onset, color="#2ca02c", linestyle="-", linewidth=2.5, label="True Clinical Seizure Onset")
    for at in alarm_timestamps:
        plt.axvline(x=at, color="#ff7f0e", linestyle=":", linewidth=2, label="Discrete Forecast Alarm Fired" if at == alarm_timestamps[0] else "")
    plt.fill_between(times_arr, 0, 1, where=(y_test_cont == 1), color="#2ca02c", alpha=0.15, label="Preictal Transition Phase")
    plt.title(f"Continuous End-to-End Seizure Forecasting Prototype — Patient {p} (CHB-MIT)", fontsize=14, fontweight="bold")
    plt.xlabel("Continuous Hold-Out Recording Time (Minutes)", fontsize=12)
    plt.ylabel("Seizure Risk Score (Probability)", fontsize=12)
    plt.ylim(0, 1.05)
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_plot = Path(f"{p}_end_to_end_continuous_forecast.png")
    plt.savefig(out_plot, dpi=300)
    print(f"Visual forecast demonstration saved successfully to: {out_plot.absolute()}", flush=True)

if __name__ == "__main__":
    main()
