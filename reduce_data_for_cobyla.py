import os
import pandas as pd

def downsample_structure():
    # 1. Define Paths
    base_dir = 'data'
    source_folder = os.path.join(base_dir, 'reduce_row_number_absolutes')
    target_folder = os.path.join(base_dir, 'reduce_row_number_cobyla')
    
    subfolders = ['train', 'test', 'validation']

    print(f"Starting downsampling from: {source_folder}")
    print(f"Targeting new structure at: {target_folder}")
    print("-" * 50)

    # 2. Check if source exists
    if not os.path.exists(source_folder):
        print(f"ERROR: Source folder '{source_folder}' not found.")
        return

    # 3. Process each subfolder
    for sub in subfolders:
        src_path = os.path.join(source_folder, sub)
        dst_path = os.path.join(target_folder, sub)

        # Create the target subfolder (train/test/val) if it doesn't exist
        os.makedirs(dst_path, exist_ok=True)

        if not os.path.exists(src_path):
            print(f"Skipping '{sub}': Folder not found in source.")
            continue

        files = [f for f in os.listdir(src_path) if f.endswith('.csv')]
        print(f"Processing folder: {sub} ({len(files)} files found)")

        for filename in files:
            file_src = os.path.join(src_path, filename)
            file_dst = os.path.join(dst_path, filename)

            try:
                # Load original data
                df = pd.read_csv(file_src)

                # THE LOGIC: 4s -> 12s (Jump of 3 rows)
                # Picks indices: 0, 3, 6, 9... (which correspond to 0s, 12s, 24s, 36s...)
                df_reduced = df.iloc[::3].reset_index(drop=True)

                # Save to the new mirrored folder
                df_reduced.to_csv(file_dst, index=False)
                
            except Exception as e:
                print(f"  Error processing {filename} in {sub}: {e}")

    print("-" * 50)
    print("Downsampling complete!")
    print(f"Total Rows per file reduced by ~66%.")
    print(f"Structure mirrored in: {target_folder}")

if __name__ == '__main__':
    downsample_structure()