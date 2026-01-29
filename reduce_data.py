import pandas as pd
import numpy as np
import os

def reduce_dataset(input_path, output_path=None, mode='total_points', value=1000, time_col='timestamp'):
    """
    Reduces a CSV/Excel dataset.
    
    Args:
        input_path (str): Path to original file.
        output_path (str): Path to save reduced file.
        mode (str): 
            - 'total_points': Downsamples to have exactly 'value' rows (evenly spaced).
            - 'cutoff': Keeps only the first 'value' rows.
            - 'time_interval': Keeps 1 point every 'value' seconds.
        value (float/int): The parameter for the mode.
        time_col (str): The name of the time column (required for time_interval).
    """
    
    # 1. Load Data
    print(f"Loading: {input_path}...")
    if input_path.endswith('.csv'):
        df = pd.read_csv(input_path)
    elif input_path.endswith('.xlsx'):
        df = pd.read_excel(input_path)
    else:
        raise ValueError("File must be .csv or .xlsx")
        
    original_len = len(df)
    
    # 2. Reduction Logic
    if mode == 'cutoff':
        # "Take first X points"
        limit = int(value)
        df_new = df.head(limit)
        print(f"Mode: Cutoff (First {limit} rows)")

    elif mode == 'total_points':
        # "Take X points spread evenly" (Stride)
        target_n = int(value)
        if target_n >= original_len:
            df_new = df.copy()
        else:
            step = original_len // target_n
            df_new = df.iloc[::step, :]  # Take every nth row
            df_new = df_new.head(target_n) # Ensure exact count
        print(f"Mode: Total Points (Target: {target_n}, Stride: {step})")

    elif mode == 'time_interval':
        # "Take point every X seconds"
        seconds = float(value)
        
        # Ensure time column exists
        if time_col not in df.columns:
            raise ValueError(f"Column '{time_col}' not found. Available: {list(df.columns)}")
        
        # Convert timestamp to datetime objects for resampling
        # Assuming timestamp is in seconds (UNIX or relative float)
        temp_time = pd.to_datetime(df[time_col], unit='s')
        
        # Set temp index, resample, and take the first valid value in that bin
        df_temp = df.copy()
        df_temp['temp_index'] = temp_time
        df_temp = df_temp.set_index('temp_index')
        
        # Resample (e.g., '2S' for 2 seconds)
        # .first() takes the actual data point at the start of the bin (no interpolation)
        df_new = df_temp.resample(f'{seconds}S').first().dropna().reset_index(drop=True)
        print(f"Mode: Time Interval (Every {seconds} seconds)")

    else:
        raise ValueError("Unknown mode. Use: 'total_points', 'cutoff', or 'time_interval'")

    # 3. Save
    if output_path is None:
        name, ext = os.path.splitext(input_path)
        output_path = f"{name}_reduced_{mode}_{value}{ext}"
        
    if input_path.endswith('.csv'):
        df_new.to_csv(output_path, index=False)
    else:
        df_new.to_excel(output_path, index=False)
        
    print(f"Done! Reduced from {original_len} to {len(df_new)} rows.")
    print(f"Saved to: {output_path}")
    
    return df_new

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    
    # CHANGE THIS to your file path
    file_path = "dataset/validation/zigzag_18_18_ind.csv"
    
    # # OPTION 1: Keep exactly 500 points (spread across the whole file)
    # reduce_dataset(file_path, mode='total_points', value=500)
    
    # OPTION 2: Take 1 point every 2.0 seconds
    reduce_dataset(file_path, mode='time_interval', output_path="reduce_dataset_cobyla/validation/zigzag_18_18_ind_reduced_2_s.csv", value=2.0, time_col='Time (s)')
    
    # # OPTION 3: Just take the first 1000 rows
    # reduce_dataset(file_path, mode='cutoff', value=1000)