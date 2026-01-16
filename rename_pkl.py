import os
import pickle
import pandas as pd
import sys

# --- CONFIGURATION ---
MODELS_DIR = "models"
FIGURES_DIR = "figures"
LOG_FILE = "logs/experiments_summary.xlsx"  # Supports .xlsx or .csv

# Short codes for filenames
ANSATZ_MAP = {
    'realamplitudes': 'realamp',
    'ugates': 'ugates',
    'efficientsu2': 'effsu2',
    'unknown': 'unk'
}

ENTANGLE_MAP = {
    'linear': 'lin',
    'reverse_linear': 'rev',
    'circular': 'circ',
    'full': 'full',
    'pairwise': 'pair',
    'sca': 'sca'
}

def get_timestamp_from_name(filename):
    """Extracts 'MM-DD_HH-MM-SS' from filename."""
    parts = filename.split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return None

def rename_experiments_robustly():
    # 1. Load the Log File
    df = None
    file_type = None
    
    if os.path.exists(LOG_FILE):
        print(f"Reading log: {LOG_FILE}...")
        df = pd.read_excel(LOG_FILE)
        file_type = 'xlsx'
    elif os.path.exists(LOG_FILE.replace('.xlsx', '.csv')):
        csv_path = LOG_FILE.replace('.xlsx', '.csv')
        print(f"Reading log: {csv_path}...")
        df = pd.read_csv(csv_path)
        file_type = 'csv'
    else:
        print("CRITICAL WARNING: No log file found. Files will be renamed but log cannot be updated.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y': sys.exit()

    # 2. Check Permissions (Try to open log for writing immediately to fail early)
    if df is not None:
        try:
            if file_type == 'xlsx':
                with open(LOG_FILE, 'a'): pass
            else:
                with open(LOG_FILE.replace('.xlsx', '.csv'), 'a'): pass
        except PermissionError:
            print("\n!!! ERROR: Permission Denied !!!")
            print(f"Please CLOSE '{LOG_FILE}' in Excel and try again.")
            return

    print(f"\nScanning '{MODELS_DIR}'...")
    updates_count = 0
    errors = []

    # 3. Process Files
    for filename in os.listdir(MODELS_DIR):
        if not filename.endswith(".pkl"):
            continue

        filepath = os.path.join(MODELS_DIR, filename)
        timestamp = get_timestamp_from_name(filename)

        try:
            # --- A. Load Config ---
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            config = data.get('config', {})
            
            # Skip Classical Models (no ansatz)
            if 'ansatz' not in config:
                continue

            # --- B. Determine Target Name ---
            ansatz = config.get('ansatz', 'unknown')
            entangle = config.get('entangle', 'unknown')
            reps = config.get('reps', 0)

            a_short = ANSATZ_MAP.get(ansatz, ansatz)
            e_short = ENTANGLE_MAP.get(entangle, entangle)
            
            # Reconstruct base name from config + timestamp to ensure cleanliness
            # Format: TIMESTAMP_model_f#_w#_h#...
            # But simpler: assume current filename starts with the correct base or use timestamp
            # We stick to the user's convention: Append suffix to the existing 'base' 
            # (removing old suffixes if re-running)
            
            # Heuristic: split by underscore. 
            # Standard parts: Date, Time, Model, F, W, H. (6 parts)
            parts = filename.replace('.pkl', '').split('_')
            
            # Base is usually the first 6 parts (01-14_14-28-07_vanilla_f2_w5_h5)
            # If the filename is longer, it already has suffixes.
            if len(parts) >= 6:
                base_name_str = "_".join(parts[:6]) 
            else:
                base_name_str = filename.replace('.pkl', '')

            suffix = f"_{a_short}_{e_short}_r{reps}"
            target_filename = f"{base_name_str}{suffix}.pkl"
            target_filepath = os.path.join(MODELS_DIR, target_filename)

            # --- C. Rename Disk Files (if needed) ---
            file_renamed = False
            if filename != target_filename:
                # 1. Rename .pkl
                os.rename(filepath, target_filepath)
                
                # 2. Rename Figure Folder
                old_fig_dir = os.path.join(FIGURES_DIR, filename.replace('.pkl', ''))
                new_fig_dir = os.path.join(FIGURES_DIR, target_filename.replace('.pkl', ''))
                
                if os.path.exists(old_fig_dir):
                    # Handle case where target folder exists (merge/overwrite) or rename
                    if os.path.exists(new_fig_dir):
                        print(f"   [Warn] Target figure folder exists: {new_fig_dir}. Skipping folder rename.")
                    else:
                        os.rename(old_fig_dir, new_fig_dir)
                
                print(f"Renamed: {filename} -> {target_filename}")
                file_renamed = True
            else:
                # File is already named correctly
                pass

            # --- D. Update Log (Sync) ---
            if df is not None and timestamp:
                # Find row by Timestamp (Robust matching)
                # match rows where 'model_id' contains the timestamp
                mask = df['model_id'].astype(str).str.contains(timestamp)
                
                if mask.any():
                    current_log_name = df.loc[mask, 'model_id'].values[0]
                    
                    if current_log_name != target_filename:
                        df.loc[mask, 'model_id'] = target_filename
                        updates_count += 1
                        if not file_renamed:
                            print(f"Fixed Log: {current_log_name} -> {target_filename}")
                else:
                    print(f"   [Warn] Timestamp {timestamp} not found in log file.")

        except Exception as e:
            errors.append(f"{filename}: {str(e)}")

    # 4. Save Changes
    if updates_count > 0:
        print(f"\nSaving {updates_count} updates to log file...")
        try:
            if file_type == 'xlsx':
                df.to_excel(LOG_FILE, index=False)
            else:
                df.to_csv(LOG_FILE.replace('.xlsx', '.csv'), index=False)
            print("Log file updated successfully.")
        except PermissionError:
            print("!!! FATAL: Could not save Excel file. It is likely open.")
            print("Files have been renamed, but log is not updated.")
    else:
        print("\nNo log updates needed.")

    if errors:
        print("\n--- Errors encountered ---")
        for err in errors:
            print(err)

if __name__ == "__main__":
    rename_experiments_robustly()