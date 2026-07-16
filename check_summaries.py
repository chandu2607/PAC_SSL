from pathlib import Path
import re

DATA_ROOT = Path("data/chb-mit")
PATIENTS = [f"chb{i:02d}" for i in range(1, 25) if i != 12]

def parse_summary_file(patient):
    sum_path = DATA_ROOT / patient / f"{patient}-summary.txt"
    if not sum_path.exists():
        return []
    
    content = sum_path.read_text(encoding="utf-8", errors="ignore")
    
    # We want to extract for each file: File Name, Number of Seizures, Seizure Start Time, Seizure End Time
    # Example format:
    # File Name: chb01_03.edf
    # File Start Time: 13:43:04
    # File End Time: 14:43:04
    # Number of Seizures in File: 1
    # Seizure 1 Start Time: 2996 seconds
    # Seizure 1 End Time: 3036 seconds
    
    file_records = []
    current_file = None
    
    lines = content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("File Name:"):
            fname = line.split(":", 1)[1].strip()
            current_file = {
                "file_name": fname,
                "seizures": []
            }
            file_records.append(current_file)
        elif line.startswith("Number of Seizures in File:") and current_file is not None:
            try:
                n_seiz = int(line.split(":", 1)[1].strip())
                for s in range(1, n_seiz + 1):
                    # Look ahead for Seizure s Start Time and End Time
                    start_sec = None
                    end_sec = None
                    while i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if f"Seizure {s} Start Time:" in next_line or (n_seiz == 1 and "Seizure Start Time:" in next_line):
                            match = re.search(r"(\d+)\s*seconds", next_line)
                            if match:
                                start_sec = int(match.group(1))
                        elif f"Seizure {s} End Time:" in next_line or (n_seiz == 1 and "Seizure End Time:" in next_line):
                            match = re.search(r"(\d+)\s*seconds", next_line)
                            if match:
                                end_sec = int(match.group(1))
                        elif next_line.startswith("File Name:") or (f"Seizure {s+1} Start Time:" in next_line):
                            break
                        i += 1
                        if start_sec is not None and end_sec is not None:
                            current_file["seizures"].append((start_sec, end_sec))
                            break
            except Exception as e:
                pass
        i += 1
        
    return file_records

def audit_all_summaries():
    total_seizures = 0
    total_files = 0
    files_with_seizures = 0
    
    print("=== Summary File Audit Across 23 Patients ===")
    for p in PATIENTS:
        records = parse_summary_file(p)
        p_files = len(records)
        p_seiz = sum(len(r["seizures"]) for r in records)
        p_seiz_files = sum(1 for r in records if len(r["seizures"]) > 0)
        
        total_files += p_files
        total_seizures += p_seiz
        files_with_seizures += p_seiz_files
        print(f"  {p}: {p_files} files | {p_seiz_files} files with seizures | Total seizures: {p_seiz}")
        if p_seiz > 0:
            # Show first seizure example
            for r in records:
                if r["seizures"]:
                    print(f"      Example: {r['file_name']} -> {r['seizures']}")
                    break
                    
    print(f"\nTOTAL across 23 patients: {total_files} files | {files_with_seizures} files with seizures | {total_seizures} total seizures")

if __name__ == "__main__":
    audit_all_summaries()
