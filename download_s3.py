"""
download_s3.py — Download CHB-MIT dataset from PhysioNet's S3 mirror.
Uses boto3 with unsigned requests (no AWS account needed).
Downloads patient-by-patient, skipping chb12 and already-downloaded files.
Uses concurrent ThreadPoolExecutor for parallel downloads.
"""
import os
import sys
import time
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore import UNSIGNED
from botocore.config import Config

# Configuration
BUCKET = "physionet-open"
PREFIX = "chbmit/1.0.0/"
DEST_ROOT = Path(__file__).parent / "data" / "chb-mit"
LOG_FILE = Path(__file__).parent / "data" / "download_log.txt"
MAX_WORKERS = 12  # parallel download threads
SKIP_PATIENTS = {"chb12"}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

def get_s3_client():
    """Create an unsigned S3 client (no credentials needed)."""
    return boto3.client("s3", config=Config(signature_version=UNSIGNED), region_name="us-east-1")

def list_all_objects(s3, bucket, prefix):
    """List all objects under a prefix using pagination."""
    objects = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            objects.append(obj)
    return objects

def download_file(s3, bucket, key, local_path):
    """Download a single file from S3."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, str(local_path))
    return local_path

def main():
    s3 = get_s3_client()
    
    log.info("=== CHB-MIT S3 Download Started ===")
    log.info(f"Source: s3://{BUCKET}/{PREFIX}")
    log.info(f"Destination: {DEST_ROOT}")
    log.info(f"Parallel workers: {MAX_WORKERS}")
    
    # List all objects in the dataset
    log.info("Listing files on S3...")
    all_objects = list_all_objects(s3, BUCKET, PREFIX)
    log.info(f"Total objects on S3: {len(all_objects)}")
    
    # Filter: skip chb12, build download list
    to_download = []
    skipped_existing = 0
    skipped_chb12 = 0
    total_size = 0
    
    for obj in all_objects:
        key = obj["Key"]
        rel_path = key[len(PREFIX):]  # Remove prefix to get relative path
        
        if not rel_path:
            continue
            
        # Check if this belongs to a skipped patient
        parts = rel_path.split("/")
        if parts[0] in SKIP_PATIENTS:
            skipped_chb12 += 1
            continue
        
        local_path = DEST_ROOT / rel_path.replace("/", os.sep)
        
        # Skip if already downloaded with correct size
        if local_path.exists() and local_path.stat().st_size == obj["Size"]:
            skipped_existing += 1
            continue
        
        to_download.append((key, local_path, obj["Size"]))
        total_size += obj["Size"]
    
    log.info(f"Files to download: {len(to_download)} ({total_size / (1024**3):.2f} GB)")
    log.info(f"Already downloaded (skipped): {skipped_existing}")
    log.info(f"Skipped chb12 files: {skipped_chb12}")
    
    if not to_download:
        log.info("All files already downloaded!")
        return
    
    # Download with thread pool
    completed = 0
    failed = []
    start_time = time.time()
    last_report_time = start_time
    bytes_downloaded = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for key, local_path, size in to_download:
            future = executor.submit(download_file, get_s3_client(), BUCKET, key, local_path)
            futures[future] = (key, local_path, size)
        
        for future in as_completed(futures):
            key, local_path, size = futures[future]
            try:
                future.result()
                completed += 1
                bytes_downloaded += size
                
                # Progress report every 30 seconds or every 20 files
                now = time.time()
                if now - last_report_time > 30 or completed % 20 == 0:
                    elapsed = now - start_time
                    speed_mbps = (bytes_downloaded / (1024**2)) / elapsed if elapsed > 0 else 0
                    pct = (completed / len(to_download)) * 100
                    remaining_bytes = total_size - bytes_downloaded
                    eta_sec = remaining_bytes / (bytes_downloaded / elapsed) if bytes_downloaded > 0 else 0
                    eta_min = eta_sec / 60
                    
                    rel = str(local_path.relative_to(DEST_ROOT))
                    patient = rel.split(os.sep)[0] if os.sep in rel else rel
                    log.info(
                        f"  Progress: {completed}/{len(to_download)} ({pct:.1f}%) | "
                        f"{bytes_downloaded/(1024**3):.2f}/{total_size/(1024**3):.2f} GB | "
                        f"{speed_mbps:.1f} MB/s | ETA: {eta_min:.0f} min | "
                        f"Current: {patient}"
                    )
                    last_report_time = now
                    
            except Exception as e:
                failed.append((key, str(e)))
                log.error(f"  FAILED: {key} — {e}")
    
    # Retry failed files
    if failed:
        log.info(f"Retrying {len(failed)} failed files...")
        still_failed = []
        retry_s3 = get_s3_client()
        for key, error in failed:
            rel_path = key[len(PREFIX):]
            local_path = DEST_ROOT / rel_path.replace("/", os.sep)
            try:
                download_file(retry_s3, BUCKET, key, local_path)
                completed += 1
                log.info(f"  Retry OK: {key}")
            except Exception as e:
                still_failed.append(key)
                log.error(f"  Retry FAILED: {key} — {e}")
        failed = still_failed
    
    # Final summary
    elapsed = time.time() - start_time
    log.info("=== Download Complete ===")
    log.info(f"Time: {elapsed/60:.1f} minutes")
    log.info(f"Downloaded: {completed} files ({bytes_downloaded/(1024**3):.2f} GB)")
    log.info(f"Previously existed: {skipped_existing}")
    log.info(f"Failed: {len(failed)}")
    
    if failed:
        for f in failed:
            log.info(f"  MISSING: {f}")
    
    # Per-patient summary
    log.info("")
    log.info("--- Per-Patient Summary ---")
    patients = sorted([d.name for d in DEST_ROOT.iterdir() if d.is_dir() and d.name.startswith("chb")])
    total_dataset_size = 0
    for patient in patients:
        patient_dir = DEST_ROOT / patient
        edf_files = list(patient_dir.glob("*.edf"))
        patient_size = sum(f.stat().st_size for f in patient_dir.iterdir() if f.is_file())
        total_dataset_size += patient_size
        log.info(f"  {patient}: {len(edf_files)} .edf files, {patient_size/(1024**3):.2f} GB")
    
    log.info(f"\nTotal dataset size: {total_dataset_size/(1024**3):.2f} GB")
    log.info("=========================")

if __name__ == "__main__":
    main()
