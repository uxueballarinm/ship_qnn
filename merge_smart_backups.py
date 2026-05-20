import pandas as pd
import glob
import os

def get_abs_path(path):
    """Converts relative path to absolute and adds Windows Long Path prefix."""
    abs_p = os.path.abspath(path)
    if os.name == 'nt' and not abs_p.startswith('\\\\?\\'):
        return '\\\\?\\' + abs_p
    return abs_p

def save_to_target(data_list, target_path):
    """Helper function to merge and save data to a specific Excel file."""
    if not data_list:
        return

    new_data = pd.concat(data_list, ignore_index=True)
    target_abs = os.path.abspath(target_path)
    
    print(f"\nProcessing {os.path.basename(target_path)}...")
    print(f"Total new rows to merge: {len(new_data)}")

    if os.path.exists(target_abs):
        try:
            existing_df = pd.read_excel(target_abs, engine='openpyxl')
            final_df = pd.concat([existing_df, new_data], ignore_index=True)
            print(f"Appending to existing file ({len(existing_df)} rows already present).")
        except Exception as e:
            print(f"Could not read existing file, starting fresh. Error: {e}")
            final_df = new_data
    else:
        final_df = new_data

    os.makedirs(os.path.dirname(target_abs), exist_ok=True)
    
    try:
        final_df.to_excel(target_abs, index=False, engine='openpyxl')
        print(f"SUCCESS: {target_path} saved with {len(final_df)} total rows.")
    except PermissionError:
        print(f"CRITICAL: Close '{target_path}' and run again!")

def smart_merge_backups_final():
    # --- 1. SETTINGS ---
    # Simplified search dir to avoid the 'double folder' issue
    base = r"logs\experiments_systematic\correlation_study"
    base_search = r"logs\experiments_systematic\correlation_study\study_qnn\backups"
    base_search_abs = get_abs_path(base_search)
    target_identity = os.path.join(base, "correlation_qnn_identity_2.xlsx")
    target_uniform = os.path.join(base, "correlation_qelm_uniform_2.xlsx")

    head_num = 2

    # --- 2. FIND ALL FILES ---
    # We search from the parent folder to catch everything
    search_pattern = os.path.join(base_search_abs, "**", "*.[xc][ls][sv]*")
    files = glob.glob(search_pattern, recursive=True)
    
    targets = [os.path.abspath(target_identity), os.path.abspath(target_uniform)]
    backup_files = [f for f in files if os.path.abspath(f) not in targets]

    if not backup_files:
        print(f"No files found in: {base_search}")
        return

    print(f"Found {len(backup_files)} potential backup files.")

    identity_list = []
    uniform_list = []
    files_processed = []

    # --- 3. PROCESS ---
    for f in backup_files:
        # Use the Long Path Fix for every file open attempt
        f_abs = get_abs_path(f)
        
        if not os.path.exists(f_abs):
            continue

        try:
            if f.lower().endswith('.csv'):
                df = pd.read_csv(f_abs)
            else:
                df = pd.read_excel(f_abs, engine='openpyxl')

            if df.empty:
                continue

            col_head = 'head_number' if 'head_number' in df.columns else 'num_heads'
            
            if col_head in df.columns:
                val_head = pd.to_numeric(df[col_head].iloc[0], errors='coerce')
                
                if val_head == head_num:
                    if 'initialization' in df.columns:
                        init_type = str(df['initialization'].iloc[0]).lower().strip()
                        
                        # Date cleaning
                        date_col = next((c for c in df.columns if c.lower() == 'date'), None)
                        if date_col:
                            df[date_col] = pd.to_datetime(df[date_col], errors='coerce').dt.strftime('%d/%m/%Y %H:%M:%S')

                        if init_type == 'identity':
                            identity_list.append(df)
                            files_processed.append(f)
                        elif init_type == 'uniform':
                            uniform_list.append(df)
                            files_processed.append(f)
                    else:
                        print(f"  [Skipped] {os.path.basename(f)} (No 'initialization' column)")
        except Exception as e:
            # Silencing the error noise unless it's not a path issue
            pass

    # --- 4. SAVE ---
    save_to_target(identity_list, target_identity)
    save_to_target(uniform_list, target_uniform)

    # --- 5. CLEANUP ---
    if files_processed:
        print(f"\nSuccessfully processed {len(files_processed)} files.")
        choice = input(f"Delete processed backup files? (y/n): ")
        if choice.lower() == 'y':
            for f in files_processed:
                try:
                    os.remove(get_abs_path(f))
                except:
                    pass
            print("Files deleted.")
    else:
        print("\nNo data matched the criteria. Nothing saved.")

if __name__ == "__main__":
    smart_merge_backups_final()