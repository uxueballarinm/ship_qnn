import pandas as pd
import numpy as np
import os

def reduce_dataset(input_path, output_path=None, mode='total_points', value=1000, time_col='timestamp'):
    """
    Reduces a CSV/Excel dataset.
    """
    
    # 1. Load Data
    # print(f"Processing: {os.path.basename(input_path)}...", end=" ")
    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    elif input_path.endswith('.xlsx'):
        df = pd.read_excel(input_path)
    else:
        print(f"Skipping {input_path} (not .csv or .xlsx)")
        return
        
    original_len = len(df)
    
    # 2. Reduction Logic
    if mode == 'cutoff':
        # "Take first X points"
        limit = int(value)
        df_new = df.head(limit)
        # print(f"-> Cutoff ({limit})")

    elif mode == 'total_points':
        # "Take X points spread evenly" (Stride)
        target_n = int(value)
        if target_n >= original_len:
            df_new = df.copy()
        else:
            # Calculate stride to get approximately target_n
            step = max(1, original_len // target_n)
            df_new = df.iloc[::step, :]  # Take every nth row
            df_new = df_new.head(target_n) # Ensure exact count if slight mismatch
        # print(f"-> Total Points ({len(df_new)})")

    elif mode == 'time_interval':
        # "Take point every X seconds"
        seconds = float(value)
        
        # Ensure time column exists
        if time_col not in df.columns:
            print(f"Error: Column '{time_col}' not found.")
            return
        
        # Convert timestamp to datetime objects for resampling
        temp_time = pd.to_datetime(df[time_col], unit='s')
        
        df_temp = df.copy()
        df_temp['temp_index'] = temp_time
        df_temp = df_temp.set_index('temp_index')
        
        # Resample and take first valid value
        df_new = df_temp.resample(f'{seconds}S').first().dropna().reset_index(drop=True)
        # print(f"-> Interval ({seconds}s)")

    else:
        raise ValueError("Unknown mode. Use: 'total_points', 'cutoff', or 'time_interval'")

    # 3. Save
    if output_path is None:
        name, ext = os.path.splitext(input_path)
        output_path = f"{name}_reduced_{mode}_{value}{ext}"
        
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if input_path.endswith('.csv'):
        df_new.to_csv(output_path, index=False)
    else:
        df_new.to_excel(output_path, index=False)
        
    print(f"[{mode}] {os.path.basename(input_path)}: {original_len} -> {len(df_new)} rows. Saved.")
    
    return df_new

def process_directory(input_dir, output_dir, mode, value, time_col='timestamp'):
    """
    Walks through input_dir, replicates structure in output_dir, and reduces files.
    """
    print(f"\n--- Starting Batch Processing ---")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Mode:   {mode} = {value}")
    print("-" * 30)

    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    count = 0
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.csv', '.xlsx')):
                # 1. Get full path of source
                src_path = os.path.join(root, file)
                
                # 2. Compute relative path (e.g., 'train/file.csv')
                rel_path = os.path.relpath(src_path, input_dir)
                
                # 3. Compute full path of destination
                dest_path = os.path.join(output_dir, rel_path)
                
                # 4. Process
                reduce_dataset(src_path, dest_path, mode=mode, value=value, time_col=time_col)
                count += 1
    
    print("-" * 30)
    print(f"Batch processing complete. Processed {count} files.")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    
    # 1. CONFIGURATION
    INPUT_FOLDER = "data\\dataset"
    OUTPUT_FOLDER = "data\\reduce_row_number_2"
    
    # Modes: 'total_points' (row count), 'time_interval' (seconds), 'cutoff' (first N)
    REDUCTION_MODE = 'total_points' 
    
    # If mode is 'total_points', this is the target number of rows per file.
    # If mode is 'time_interval', this is the seconds between points.
    TARGET_VALUE = 800
    
    # Only needed if using 'time_interval'
    TIME_COLUMN = 'Time (s)' 

    # 2. RUN BATCH PROCESS
    process_directory(
        input_dir=INPUT_FOLDER, 
        output_dir=OUTPUT_FOLDER, 
        mode=REDUCTION_MODE, 
        value=TARGET_VALUE, 
        time_col=TIME_COLUMN
    )