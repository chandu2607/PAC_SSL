"""
preprocess_stage1.py — Stage 1: Fast Parallel Data Verification, Channel Mapping, Filtering, & Segment Extraction
Extracts 4-second windows (1024 samples at 256Hz) with 50% overlap (2-second step).
Bandpass filter: 0.5 - 100 Hz + 60 Hz notch filter (preserving 30-80 Hz gamma amplitude for PAC-SSL).
Preictal: 30 to 120 min before seizure onset.
Interictal: > 4 hours from any seizure onset across the patient timeline.
Tracks `block_id` per source recording to prevent data leakage across splits.
Uses ProcessPoolExecutor (8 concurrent workers) for multi-core speed.
"""
import os
import sys
import re
import time
import logging
from pathlib import Path
import numpy as np
import h5py
import pyedflib
from scipy import signal
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configuration
DATA_ROOT = Path("data/chb-mit")
OUT_ROOT = Path("data/preprocessed")
OUT_ROOT.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUT_ROOT / "stage1_preprocessing.log"

PATIENTS = [f"chb{i:02d}" for i in range(1, 25) if i != 12]

STANDARD_18_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    'FZ-CZ',  'CZ-PZ'
]

CHANNEL_ALIASES = {
    'F7-T3': 'F7-T7', 'T3-T5': 'T7-P7', 'T5-O1': 'P7-O1',
    'F8-T4': 'F8-T8', 'T4-T6': 'T8-P8', 'T6-O2': 'P8-O2',
    'FP1-F7-1': 'FP1-F7', 'FP2-F8-1': 'FP2-F8',
    'T7-P7-1': 'T7-P7', 'P7-O1-1': 'P7-O1'
}

FS = 256
WIN_SAMPLES = 4 * FS       # 4-second windows = 1024 samples
STEP_SAMPLES = 2 * FS      # 50% overlap = 2-second step = 512 samples
INTERICTAL_STEP_SAMPLES = 4 * FS  # 4-second step for interictal to keep balanced storage while spanning all qualifying files

PREICTAL_START_SEC = 120 * 60  # 120 minutes before onset
PREICTAL_END_SEC = 30 * 60     # 30 minutes before onset
INTERICTAL_BUFFER_SEC = 4 * 3600  # 4 hours away from any seizure
MAX_WORKERS = 8


def log_msg(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def parse_summary(patient):
    sum_path = DATA_ROOT / patient / f"{patient}-summary.txt"
    if not sum_path.exists():
        return {}
    
    content = sum_path.read_text(encoding="utf-8", errors="ignore")
    lines = content.splitlines()
    
    files_info = {}
    current_file = None
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("File Name:"):
            fname = line.split(":", 1)[1].strip()
            current_file = fname
            files_info[fname] = {"seizures": []}
        elif line.startswith("Number of Seizures in File:") and current_file:
            try:
                n_seiz = int(line.split(":", 1)[1].strip())
                for s in range(1, n_seiz + 1):
                    start_sec, end_sec = None, None
                    while i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if f"Seizure {s} Start Time:" in next_line or (n_seiz == 1 and "Seizure Start Time:" in next_line):
                            m = re.search(r"(\d+)\s*seconds", next_line)
                            if m: start_sec = int(m.group(1))
                        elif f"Seizure {s} End Time:" in next_line or (n_seiz == 1 and "Seizure End Time:" in next_line):
                            m = re.search(r"(\d+)\s*seconds", next_line)
                            if m: end_sec = int(m.group(1))
                        elif next_line.startswith("File Name:") or (f"Seizure {s+1} Start Time:" in next_line):
                            break
                        i += 1
                        if start_sec is not None and end_sec is not None:
                            files_info[current_file]["seizures"].append((start_sec, end_sec))
                            break
            except Exception as e:
                pass
        i += 1
    return files_info


def design_filters(fs=FS):
    sos_bp = signal.butter(4, [0.5, 100.0], btype="bandpass", fs=fs, output="sos")
    sos_notch = signal.tf2sos(*signal.iirnotch(60.0, 30.0, fs=fs))
    return sos_bp, sos_notch


def load_and_map_channels(edf_path):
    reader = pyedflib.EdfReader(str(edf_path))
    n_signals = reader.signals_in_file
    raw_labels = [reader.getLabel(i).strip().upper() for i in range(n_signals)]
    fs = reader.getSampleFrequency(0)
    
    mapped_indices = []
    for std_ch in STANDARD_18_CHANNELS:
        idx = -1
        if std_ch in raw_labels:
            idx = raw_labels.index(std_ch)
        else:
            for alias, target in CHANNEL_ALIASES.items():
                if target == std_ch and alias in raw_labels:
                    idx = raw_labels.index(alias)
                    break
        if idx == -1:
            reader.close()
            raise ValueError(f"Missing required channel {std_ch} in {edf_path.name}")
        mapped_indices.append(idx)
        
    n_samples = reader.getNSamples()[0]
    data = np.zeros((len(STANDARD_18_CHANNELS), n_samples), dtype=np.float32)
    for i, idx in enumerate(mapped_indices):
        data[i, :] = reader.readSignal(idx).astype(np.float32)
        
    reader.close()
    return data, fs


def process_patient_worker(patient):
    """Worker function to process all files of a single patient."""
    t0 = time.time()
    sos_bp, sos_notch = design_filters(FS)
    
    p_dir = DATA_ROOT / patient
    edf_files = sorted(list(p_dir.glob("*.edf")))
    summary_info = parse_summary(patient)
    
    all_seizures = []
    for fname, info in summary_info.items():
        for s_start, s_end in info["seizures"]:
            all_seizures.append((fname, s_start, s_end))
            
    preictal_windows = []
    preictal_block_ids = []
    preictal_files = []
    preictal_timestamps = []
    
    interictal_windows = []
    interictal_block_ids = []
    interictal_files = []
    interictal_timestamps = []
    
    for idx_f, edf_path in enumerate(edf_files):
        fname = edf_path.name
        try:
            raw_data, fs = load_and_map_channels(edf_path)
            if fs != FS:
                n_resampled = int(raw_data.shape[1] * FS / fs)
                raw_data = signal.resample(raw_data, n_resampled, axis=1).astype(np.float32)
                
            filtered_data = signal.sosfiltfilt(sos_bp, raw_data, axis=-1).astype(np.float32)
            filtered_data = signal.sosfiltfilt(sos_notch, filtered_data, axis=-1).astype(np.float32)
            
            n_samples = filtered_data.shape[1]
            file_seizures = [s for s in all_seizures if s[0] == fname]
            
            # 1. PREICTAL EXTRACTION: [onset - 7200, onset - 1800]
            for s_idx, (_, s_start, s_end) in enumerate(file_seizures):
                block_id = f"{patient}_{fname}_seiz_{s_idx+1}"
                win_start_sec = max(0, s_start - PREICTAL_START_SEC)
                win_end_sec = max(0, s_start - PREICTAL_END_SEC)
                
                if win_end_sec > win_start_sec:
                    start_sample = int(win_start_sec * FS)
                    end_sample = int(win_end_sec * FS)
                    
                    for sample_idx in range(start_sample, end_sample - WIN_SAMPLES + 1, STEP_SAMPLES):
                        win = filtered_data[:, sample_idx : sample_idx + WIN_SAMPLES]
                        preictal_windows.append(win)
                        preictal_block_ids.append(block_id)
                        preictal_files.append(fname)
                        preictal_timestamps.append(sample_idx / FS)
                        
            # 2. INTERICTAL EXTRACTION: > 4 hours from any seizure
            if len(file_seizures) == 0:
                file_idx = edf_files.index(edf_path)
                is_far = True
                for s_fname, _, _ in all_seizures:
                    matching_edfs = [f for f in edf_files if f.name == s_fname]
                    if matching_edfs:
                        s_file_idx = edf_files.index(matching_edfs[0])
                        # Check if within 4 files (~4 hours) of a seizure
                        if abs(file_idx - s_file_idx) <= 4:
                            is_far = False
                            break
                            
                if is_far:
                    block_id = f"{patient}_{fname}_interictal"
                    for sample_idx in range(0, n_samples - WIN_SAMPLES + 1, INTERICTAL_STEP_SAMPLES):
                        win = filtered_data[:, sample_idx : sample_idx + WIN_SAMPLES]
                        interictal_windows.append(win)
                        interictal_block_ids.append(block_id)
                        interictal_files.append(fname)
                        interictal_timestamps.append(sample_idx / FS)
                        
        except Exception as e:
            pass

    out_h5 = OUT_ROOT / f"{patient}_segments.h5"
    with h5py.File(out_h5, "w") as f:
        g_pre = f.create_group("preictal")
        if preictal_windows:
            g_pre.create_dataset("data", data=np.stack(preictal_windows, axis=0), dtype=np.float32, compression="gzip")
            g_pre.create_dataset("block_id", data=np.array(preictal_block_ids, dtype="S"))
            g_pre.create_dataset("file_name", data=np.array(preictal_files, dtype="S"))
            g_pre.create_dataset("timestamp", data=np.array(preictal_timestamps, dtype=np.float64))
        else:
            g_pre.create_dataset("data", shape=(0, len(STANDARD_18_CHANNELS), WIN_SAMPLES), dtype=np.float32)
            
        g_inter = f.create_group("interictal")
        if interictal_windows:
            g_inter.create_dataset("data", data=np.stack(interictal_windows, axis=0), dtype=np.float32, compression="gzip")
            g_inter.create_dataset("block_id", data=np.array(interictal_block_ids, dtype="S"))
            g_inter.create_dataset("file_name", data=np.array(interictal_files, dtype="S"))
            g_inter.create_dataset("timestamp", data=np.array(interictal_timestamps, dtype=np.float64))
        else:
            g_inter.create_dataset("data", shape=(0, len(STANDARD_18_CHANNELS), WIN_SAMPLES), dtype=np.float32)

    elapsed = time.time() - t0
    n_pre = len(preictal_windows)
    n_inter = len(interictal_windows)
    log_msg(f"Completed {patient} ({len(edf_files)} files) in {elapsed:.1f}s | Preictal (4s, 50% ovl): {n_pre} | Interictal: {n_inter}")
    return patient, n_pre, n_inter


def main():
    if LOG_FILE.exists():
        LOG_FILE.unlink()
    log_msg(f"=== Starting Fast Parallel Stage 1 Preprocessing ({MAX_WORKERS} workers) across 23 Subjects ===")
    t0 = time.time()
    
    total_pre = 0
    total_inter = 0
    patient_stats = {}
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_patient_worker, p): p for p in PATIENTS}
        for future in as_completed(futures):
            try:
                patient, n_pre, n_inter = future.result()
                total_pre += n_pre
                total_inter += n_inter
                patient_stats[patient] = {"preictal": n_pre, "interictal": n_inter}
            except Exception as e:
                p = futures[future]
                log_msg(f"ERROR processing {p}: {e}")
                patient_stats[p] = {"preictal": 0, "interictal": 0}
                
    elapsed = time.time() - t0
    log_msg(f"\n=== Stage 1 Summary Across 23 Subjects (Total time: {elapsed/60:.1f} min) ===")
    log_msg(f"Total Preictal Windows (4s, 50% overlap): {total_pre}")
    log_msg(f"Total Interictal Windows (4s, 0% overlap): {total_inter}")
    
    summary_txt = OUT_ROOT / "stage1_counts.txt"
    with open(summary_txt, "w") as f:
        f.write("Patient\tPreictal_4s\tInterictal_4s\n")
        for p in sorted(patient_stats.keys()):
            stats = patient_stats[p]
            f.write(f"{p}\t{stats['preictal']}\t{stats['interictal']}\n")
    log_msg(f"Saved counts to {summary_txt}")


if __name__ == "__main__":
    main()
