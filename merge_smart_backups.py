import pandas as pd
import glob
import os

def smart_merge_backups_final():
    # --- 1. PATH DEFINITIONS ---
    path_1_heads = r"logs/after_analysing_features/experiments_summary.xlsx"
    path_2_heads = r"logs/after_analysing_features/experiments_summary.xlsx"
    path_3_heads = r"logs/after_analysing_features/experiments_summary.xlsx"
    path_4_heads = r"logs/after_analysing_features/experiments_summary.xlsx"
    # --- 2. SEARCH LOGIC ---
    # Use recursive search to find ALL backups in any subfolder of 'logs'
    backup_files = glob.glob("logs/after_analysing_features/*.csv", recursive=True)
    
    # Alternatively, if they aren't in folders named 'backups', use:
    # backup_files = glob.glob("logs/**/*.csv", recursive=True)

    if not backup_files:
        print(f"No CSV backup files found in the 'logs/' directory tree.")
        return

    print(f"Found {len(backup_files)} file(s). Processing in memory...")

    list_1_heads = []
    list_2_heads = []
    list_3_heads = []
    list_4_heads = []

    files_processed = []

    # --- 3. Create the parent directories if they are missing ---
    for p in [path_1_heads, path_2_heads, path_3_heads, path_4_heads]:
        folder = os.path.dirname(p)
        if folder and not os.path.exists(folder):
            print(f"Creating missing directory: {folder}")
            os.makedirs(folder, exist_ok=True)

    # --- 4. Process CSVs ---
    for csv_file in backup_files:
        try:
            df = pd.read_csv(csv_file)
            if df.empty: continue
            
            # Check for head count column
            col = 'head_number' if 'head_number' in df.columns else 'num_heads'
            if col in df.columns:
                num_heads = df[col].iloc[0]
                
                # Date cleaning
                date_col = next((c for c in df.columns if c.lower() == 'date'), None)
                if date_col:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce').dt.strftime('%d/%m/%Y %H:%M:%S')

                # Sort into correct list
                if num_heads == 1:
                    list_1_heads.append(df)
                elif num_heads == 2:
                    list_2_heads.append(df)
                elif num_heads == 3:
                    list_3_heads.append(df)
                elif num_heads == 4:
                    list_4_heads.append(df)
                
                files_processed.append(csv_file)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    # --- 5. Final Merge and Save ---
    # Logic remains identical, handles all 3 lists
    merge_tasks = [
        (list_1_heads, path_1_heads, "1-HEADS"),
        (list_2_heads, path_2_heads, "2-HEADS"),
        (list_3_heads, path_3_heads, "3-HEADS"),
        (list_4_heads, path_4_heads, "4-HEADS")
    ]

    for current_list, target_path, label in merge_tasks:
        if current_list:
            print(f"\nFinalizing {label} merge...")
            df_new_data = pd.concat(current_list, ignore_index=True)
            
            if os.path.exists(target_path):
                try:
                    df_existing = pd.read_excel(target_path)
                    df_final = pd.concat([df_existing, df_new_data], ignore_index=True)
                except Exception as e:
                    print(f"Could not read existing Excel {target_path}, creating new. Error: {e}")
                    df_final = df_new_data
            else:
                df_final = df_new_data
                
            try:
                df_final.to_excel(target_path, index=False)
                print(f"SUCCESS: {target_path} updated.")
            except PermissionError:
                print(f"CRITICAL ERROR: Please close '{target_path}' and run again!")

    # --- 6. Cleanup ---
    if files_processed:
        choice = input(f"\nSuccessfully processed {len(files_processed)} files. Delete CSVs? (y/n): ")
        if choice.lower() == 'y':
            for f in files_processed:
                os.remove(f)
            print("CSV files deleted.")

if __name__ == "__main__":
    smart_merge_backups_final()