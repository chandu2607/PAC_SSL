from pathlib import Path
import pyedflib

DATA_ROOT = Path("data/chb-mit")
PATIENTS = [f"chb{i:02d}" for i in range(1, 25) if i != 12]

def find_common_channels():
    all_channel_sets = []
    
    for p in PATIENTS:
        p_dir = DATA_ROOT / p
        if not p_dir.exists():
            continue
        edf_files = sorted(list(p_dir.glob("*.edf")))
        
        # Check every file for this patient to ensure we catch variations
        patient_common = None
        for f in edf_files:
            try:
                reader = pyedflib.EdfReader(str(f))
                chs = set([reader.getLabel(i).strip().upper() for i in range(reader.signals_in_file)])
                reader.close()
                if patient_common is None:
                    patient_common = chs
                else:
                    patient_common = patient_common.intersection(chs)
            except Exception as e:
                pass
        
        all_channel_sets.append((p, patient_common))
    
    # Find overall intersection across all 23 subjects
    overall_common = all_channel_sets[0][1]
    for p, chs in all_channel_sets[1:]:
        overall_common = overall_common.intersection(chs)
        
    print(f"=== Common Intersection Across ALL 23 Subjects (all files) ===")
    print(f"Total common channels: {len(overall_common)}")
    print(f"Common channels: {sorted(list(overall_common))}")
    
    # Let's check what channels are present in say 21 or 22 patients
    from collections import Counter
    ch_counts = Counter()
    for p, chs in all_channel_sets:
        for ch in chs:
            if ch not in ['-', '.', 'VNS', 'ECG', 'LOC', 'ROC']:
                ch_counts[ch] += 1
                
    print("\n=== Channel Frequency Across 23 Subjects ===")
    for ch, count in ch_counts.most_common():
        print(f"  {ch}: present in {count}/23 subjects")

if __name__ == "__main__":
    find_common_channels()
