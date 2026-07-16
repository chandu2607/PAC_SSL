"""
convert_h5_to_npy.py — One-time conversion from gzip-compressed HDF5 to uncompressed numpy files.
Reads each patient's HDF5 sequentially (fast contiguous reads, no random access overhead),
saves as uncompressed .npy files that support O(1) memory-mapped random access.

Data integrity: writes the exact same float32 values — only the storage format changes.
"""
import time
import h5py
import numpy as np
from pathlib import Path

DATA_ROOT = Path("data/preprocessed")
NPY_ROOT = DATA_ROOT / "npy"
PATIENTS_ALL = [f"chb{i:02d}" for i in range(1, 25) if i != 12]


def convert_all_patients():
    NPY_ROOT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    total_windows = 0

    for p in PATIENTS_ALL:
        h5_path = DATA_ROOT / f"{p}_segments.h5"
        if not h5_path.exists():
            print(f"[{p}] HDF5 not found, skipping.", flush=True)
            continue

        pre_npy = NPY_ROOT / f"{p}_preictal.npy"
        inter_npy = NPY_ROOT / f"{p}_interictal.npy"

        if pre_npy.exists() and inter_npy.exists():
            print(f"[{p}] .npy files already exist, skipping.", flush=True)
            # Still count for total
            pre = np.load(pre_npy, mmap_mode='r')
            inter = np.load(inter_npy, mmap_mode='r')
            total_windows += len(pre) + len(inter)
            continue

        t1 = time.time()
        with h5py.File(h5_path, "r") as f:
            # Sequential read of entire arrays (fast even on gzip — contiguous chunk reads)
            pre_data = f["preictal"]["data"][:]     # shape (N_pre, 18, 1024) float32
            inter_data = f["interictal"]["data"][:]  # shape (N_inter, 18, 1024) float32

        # Save as uncompressed numpy arrays
        np.save(pre_npy, pre_data)
        np.save(inter_npy, inter_data)

        n_pre = len(pre_data)
        n_inter = len(inter_data)
        total_windows += n_pre + n_inter
        h5_size_mb = h5_path.stat().st_size / 1024 / 1024
        npy_size_mb = (pre_npy.stat().st_size + inter_npy.stat().st_size) / 1024 / 1024

        print(f"[{p}] Converted: {n_pre:,} preictal + {n_inter:,} interictal = {n_pre+n_inter:,} windows | "
              f"H5: {h5_size_mb:.0f} MB -> NPY: {npy_size_mb:.0f} MB | {time.time()-t1:.1f}s", flush=True)

        # Free RAM immediately
        del pre_data, inter_data

    elapsed = time.time() - t0
    print(f"\nConversion complete: {total_windows:,} total windows across {len(PATIENTS_ALL)} patients in {elapsed:.1f}s", flush=True)
    print(f"NPY directory: {NPY_ROOT.resolve()}", flush=True)


if __name__ == "__main__":
    convert_all_patients()
