"""
plot_seizure_sanity.py — Generate visual sanity check plot for Review Gate Stage 1.
Plots one raw EEG channel (FP1-F7 and T7-P7) around a real seizure onset in chb01_03.edf.
Shows unfiltered raw signal vs. 0.5-100Hz + 60Hz notch filtered signal surrounding exact onset timestamp (2996s).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyedflib
from pathlib import Path
from scipy import signal

EDF_PATH = Path("data/chb-mit/chb01/chb01_03.edf")
ONSET_SEC = 2996  # Seizure 1 start in chb01_03.edf per summary file
OFFSET_SEC = 3036 # Seizure 1 end
FS = 256

def generate_sanity_plot():
    reader = pyedflib.EdfReader(str(EDF_PATH))
    labels = [reader.getLabel(i).strip().upper() for i in range(reader.signals_in_file)]
    
    ch1_idx = labels.index('FP1-F7')
    ch2_idx = labels.index('T7-P7')
    
    # Read window from onset - 60s to onset + 60s
    start_sec = ONSET_SEC - 60
    end_sec = ONSET_SEC + 60
    start_sample = int(start_sec * FS)
    n_samples = int((end_sec - start_sec) * FS)
    
    # Read signals using seek/readSignal slice or full read
    full_ch1 = reader.readSignal(ch1_idx)
    full_ch2 = reader.readSignal(ch2_idx)
    reader.close()
    
    raw_ch1 = full_ch1[start_sample : start_sample + n_samples]
    raw_ch2 = full_ch2[start_sample : start_sample + n_samples]
    
    # Apply filters (0.5-100Hz bandpass + 60Hz notch)
    sos_bp = signal.butter(4, [0.5, 100.0], btype="bandpass", fs=FS, output="sos")
    sos_notch = signal.tf2sos(*signal.iirnotch(60.0, 30.0, fs=FS))
    
    filt_ch1 = signal.sosfiltfilt(sos_notch, signal.sosfiltfilt(sos_bp, raw_ch1))
    filt_ch2 = signal.sosfiltfilt(sos_notch, signal.sosfiltfilt(sos_bp, raw_ch2))
    
    time_axis = np.linspace(start_sec, end_sec, n_samples)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Plot FP1-F7
    axes[0].plot(time_axis, raw_ch1, color='#BDC3C7', alpha=0.6, label='Raw FP1-F7', linewidth=0.8)
    axes[0].plot(time_axis, filt_ch1, color='#2980B9', label='Filtered FP1-F7 (0.5-100Hz + 60Hz Notch)', linewidth=1.2)
    axes[0].axvline(x=ONSET_SEC, color='#C0392B', linestyle='--', linewidth=2.0, label=f'Seizure Onset ({ONSET_SEC}s)')
    axes[0].axvline(x=OFFSET_SEC, color='#E67E22', linestyle=':', linewidth=2.0, label=f'Seizure Offset ({OFFSET_SEC}s)')
    axes[0].axvspan(ONSET_SEC, OFFSET_SEC, color='#E74C3C', alpha=0.15, label='Seizure Duration')
    axes[0].set_title('Bipolar Scalp Channel FP1-F7 (Left Temporal Frontal) — Seizure Onset Sanity Check', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Amplitude (µV)', fontsize=11)
    axes[0].legend(loc='upper left', framealpha=0.9)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # Plot T7-P7
    axes[1].plot(time_axis, raw_ch2, color='#BDC3C7', alpha=0.6, label='Raw T7-P7', linewidth=0.8)
    axes[1].plot(time_axis, filt_ch2, color='#8E44AD', label='Filtered T7-P7 (0.5-100Hz + 60Hz Notch)', linewidth=1.2)
    axes[1].axvline(x=ONSET_SEC, color='#C0392B', linestyle='--', linewidth=2.0, label=f'Seizure Onset ({ONSET_SEC}s)')
    axes[1].axvline(x=OFFSET_SEC, color='#E67E22', linestyle=':', linewidth=2.0, label=f'Seizure Offset ({OFFSET_SEC}s)')
    axes[1].axvspan(ONSET_SEC, OFFSET_SEC, color='#E74C3C', alpha=0.15, label='Seizure Duration')
    axes[1].set_title('Bipolar Scalp Channel T7-P7 (Left Temporal Posterior) — Seizure Onset Sanity Check', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Time in Recording (Seconds)', fontsize=11)
    axes[1].set_ylabel('Amplitude (µV)', fontsize=11)
    axes[1].legend(loc='upper left', framealpha=0.9)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    out_dir = Path("C:/Users/chand/.gemini/antigravity-ide/brain/51cd7a4d-dc6f-4334-a0d8-ad7888d3a6f1/artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chb01_seizure_sanity_plot.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Sanity check plot saved successfully to {out_path}")

if __name__ == "__main__":
    generate_sanity_plot()
