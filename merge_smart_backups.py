import pandas as pd
import glob
import os

def smart_merge_backups_final():
    # --- 1. SETTINGS ---
    # We search ONLY in your specific subfolder to avoid touching other head counts
    search_dir = r"logs/experiments_systematic/qrc/ansatz_study/3heads/3heads"
    target_path = r"logs/experiments_systematic/qrc/ansatz_study/3heads/exp2.xlsx"
    
    # --- 2. FIND ALL FILES (.csv and .xlsx) ---
    # This pattern finds both types at once
    files = glob.glob(os.path.join(search_dir, "**/*.[xc][ls][sv]*"), recursive=True)
    
    # Filter out the target file itself if it happens to be in the search path
    backup_files = [f for f in files if os.path.abspath(f) != os.path.abspath(target_path)]

    if not backup_files:
        print(f"No files found in: {search_dir}")
        return

    print(f"Found {len(backup_files)} potential backup files.")

    merged_list = []
    files_processed = []

    # --- 3. PROCESS ---
    for f in backup_files:
        try:
            # Determine format by extension
            if f.lower().endswith('.csv'):
                df = pd.read_csv(f)
            else:
                df = pd.read_excel(f, engine='openpyxl')

            if df.empty:
                continue

            # Identify the head count column
            col = 'head_number' if 'head_number' in df.columns else 'num_heads'
            
            if col in df.columns:
                # FORCE to integer to prevent "3" vs 3.0 mismatches
                val = pd.to_numeric(df[col].iloc[0], errors='coerce')
                
                if val == 3:
                    print(f"  -> Adding {len(df)} rows from: {os.path.basename(f)}")
                    
                    # Date cleaning
                    date_col = next((c for c in df.columns if c.lower() == 'date'), None)
                    if date_col:
                        df[date_col] = pd.to_datetime(df[date_col], errors='coerce').dt.strftime('%d/%m/%Y %H:%M:%S')
                    
                    merged_list.append(df)
                    files_processed.append(f)
                else:
                    print(f"  [Skipped] {os.path.basename(f)} (Head count is {val}, not 3)")
            else:
                print(f"  [Skipped] {os.path.basename(f)} (Column '{col}' not found)")

        except Exception as e:
            print(f"Error reading {f}: {e}")

    # --- 4. SAVE ---
    if merged_list:
        new_data = pd.concat(merged_list, ignore_index=True)
        print(f"\nTotal new rows to merge: {len(new_data)}")

        if os.path.exists(target_path):
            try:
                existing_df = pd.read_excel(target_path, engine='openpyxl')
                final_df = pd.concat([existing_df, new_data], ignore_index=True)
                print(f"Appending to existing file ({len(existing_df)} rows already present).")
            except Exception as e:
                print(f"Could not read existing file, starting fresh. Error: {e}")
                final_df = new_data
        else:
            final_df = new_data

        # Ensure directory exists
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        try:
            final_df.to_excel(target_path, index=False, engine='openpyxl')
            print(f"SUCCESS: {target_path} saved with {len(final_df)} total rows.")
        except PermissionError:
            print(f"CRITICAL: Close '{target_path}' and run again!")

        # --- 5. CLEANUP ---
        if files_processed:
            choice = input(f"\nDelete {len(files_processed)} processed backup files? (y/n): ")
            if choice.lower() == 'y':
                for f in files_processed:
                    os.remove(f)
                print("Files deleted.")
    else:
        print("\nNo data matched the 'head_number == 3' criteria. Nothing saved.")

if __name__ == "__main__":
    smart_merge_backups_final()