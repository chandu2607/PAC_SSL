import os
from pathlib import Path
import pyedflib

DATA_ROOT = Path("data/chb-mit")
PATIENTS = [
    f"chb{i:02d}" for i in range(1, 25) if i != 12
]

def check_all_patients():
    print("=== Channel Audit across 23 Patients ===")
    channel_sets = {}
    
    for p in PATIENTS:
        p_dir = DATA_ROOT / p
        if not p_dir.exists():
            continue
        edf_files = sorted(list(p_dir.glob("*.edf")))
        if not edf_files:
            continue
        
        # Check first EDF file of the patient
        f0 = edf_files[0]
        try:
            reader = pyedflib.EdfReader(str(f0))
            ch_names = [reader.getLabel(i).strip() for i in range(reader.signals_in_file)]
            fs = reader.getSampleFrequency(0)
            reader.close()
            
            # Check if all EDF files in this patient have the same channel names
            diff_files = []
            for f in edf_files[1:]:
                try:
                    r = pyedflib.EdfReader(str(f))
                    chs = [r.getLabel(i).strip() for i in range(r.signals_in_file)]
                    r.close()
                    if chs != ch_names:
                        diff_files.append((f.name, chs))
                except Exception as e:
                    pass
            
            print(f"{p}: {len(ch_names)} channels | Fs={fs}Hz | e.g. {ch_names[:4]} ... | Diff within patient: {len(diff_files)} files")
            if diff_files:
                print(f"   -> Example diff in {diff_files[0][0]}: {diff_files[0][1][:4]} (total {len(diff_files[0][1])})")
                
            key = tuple(ch_names)
            if key not in channel_sets:
                channel_sets[key] = []
            channel_sets[key].append(p)
        except Exception as e:
            print(f"Error reading {f0}: {e}")

    print("\n=== Unique Channel Configurations Across Patients ===")
    for idx, (chs, plist) in enumerate(channel_sets.items()):
        print(f"\nConfig #{idx+1} (used by {len(plist)} patients: {plist}):")
        print(f"  Count: {len(chs)} channels")
        print(f"  List: {chs}")

if __name__ == "__main__":
    check_all_patients()
