import pandas as pd
import os

def add_features(input_dir, output_dir):
    print(f"--- Processing Physics Features ---")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print("-" * 30)
    
    count = 0
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv'):
                # Safety check: skip files that might have been processed already
                if "_adding_absolutes" in file:
                    continue

                # 1. Build Source Path
                src_path = os.path.join(root, file)
                
                # 2. Build Destination Path
                # Calculate relative path (e.g. "train/zigzag_10.csv")
                rel_path = os.path.relpath(src_path, input_dir)
                
                # Create the new filename with suffix
                name_root, ext = os.path.splitext(os.path.basename(rel_path))
                new_filename = f"{name_root}_adding_absolutes{ext}"
                
                # Keep the subdirectory structure (train/test/val)
                sub_dir = os.path.dirname(rel_path)
                
                # Final output path
                dest_dir = os.path.join(output_dir, sub_dir)
                dest_path = os.path.join(dest_dir, new_filename)
                
                # Ensure destination folder exists
                os.makedirs(dest_dir, exist_ok=True)

                # 3. Load & Process
                df = pd.read_csv(src_path)
                
                # --- ADD PHYSICS FEATURES ---
                # Absolute Sway
                if 'Sway Velocity' in df.columns:
                    df['Abs Sway'] = df['Sway Velocity'].abs()
                    
                # Absolute Rudder
                rudder_col = 'Rudder Angle (rad)' if 'Rudder Angle (rad)' in df.columns else 'Rudder Angle'
                if rudder_col in df.columns:
                    df['Abs Rudder'] = df[rudder_col].abs()
                
                # 4. Save
                df.to_csv(dest_path, index=False)
                # print(f"Saved: {dest_path}")
                count += 1
                
    print("-" * 30)
    print(f"Done! Processed {count} files.")
    print(f"All saved to: {output_dir}")

if __name__ == "__main__":
    # 1. The folder where your data is NOW (The 800-row version)
    INPUT_FOLDER = "data/reduce_row_number_2"
    
    # 2. The NEW folder where you want the result
    OUTPUT_FOLDER = "data/reduce_row_number_absolutes"
    
    add_features(INPUT_FOLDER, OUTPUT_FOLDER)