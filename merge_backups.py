import pandas as pd
import glob
import os

def merge_backups():
    excel_path = "logs/experiments_summary.xlsx"
    
    # 1. Find all backup CSV files
    backup_files = glob.glob("logs/backup_01-23_14-07-33.csv")
    
    if not backup_files:
        print("No backup files found in 'logs/'.")
        return

    print(f"Found {len(backup_files)} backup file(s). Merging...")

    # 2. Load the main Excel file
    if os.path.exists(excel_path):
        try:
            df_main = pd.read_excel(excel_path)
        except PermissionError:
            print(f"ERROR: Please close '{excel_path}' before running this script!")
            return
    else:
        print("Main Excel file not found. Creating a new one.")
        df_main = pd.DataFrame()

    # 3. Iterate and Merge
    files_merged = []
    
    for csv_file in backup_files:
        try:
            df_backup = pd.read_csv(csv_file)
            date_col = None
            if 'date' in df_backup.columns: date_col = 'date'
            elif 'Date' in df_backup.columns: date_col = 'Date'
            
            if date_col:
                df_backup[date_col] = pd.to_datetime(df_backup[date_col], errors='coerce').dt.strftime('%d/%m/%Y %H:%M:%S')

            # Concatenate
            df_main = pd.concat([df_main, df_backup], ignore_index=True)
            files_merged.append(csv_file)
            print(f" -> Merged: {csv_file}")
            
        except Exception as e:
            print(f" -> Error reading {csv_file}: {e}")

    # 4. Save back to Excel
    try:
        df_main.to_excel(excel_path, index=False)
        print("\nSUCCESS! Main Excel file updated.")
        
        # 5. Clean up (Optional - Ask user)
        choice = input("Do you want to delete the merged backup CSV files? (y/n): ")
        if choice.lower() == 'y':
            for f in files_merged:
                os.remove(f)
            print("Backup files deleted.")
        else:
            print("Backup files kept.")
            
    except PermissionError:
        print(f"\nCRITICAL ERROR: Could not save to '{excel_path}'. Is it still open?")

if __name__ == "__main__":
    merge_backups()