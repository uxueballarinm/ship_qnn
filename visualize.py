import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def process_and_plot(input_root, output_root):
    # Features to plot
    features = ['Rudder Angle (deg)', 'Yaw Rate', 'Yaw Angle', 'Surge Velocity', 'Sway Velocity']
    time_col = 'Time (s)'

    # Convert to Path objects and resolve absolute paths to prevent errors
    input_root_path = Path(input_root).resolve()
    output_root_path = Path(output_root).resolve()

    if not input_root_path.exists():
        print(f"ERROR: Folder not found! Checked: {input_root_path}")
        print("Please check if the folder name is 'reduce_row_number_cobyla' or 'reduce_row_number_absolute_cobyla'")
        return

    print(f"Looking for CSVs in: {input_root_path}")

    files_processed = 0
    # Walk through the input directory
    for root, dirs, files in os.walk(input_root_path):
        for file in files:
            if file.endswith('.csv'):
                # 1. Construct input path
                input_path = Path(root) / file
                
                # 2. Construct output path
                # This part was likely failing locally if paths weren't resolved
                relative_path = input_path.relative_to(input_root_path)
                output_dir = output_root_path / relative_path.parent
                output_dir.mkdir(parents=True, exist_ok=True)
                
                output_file = output_dir / f"{input_path.stem}.png"

                print(f"Processing: {relative_path}")

                try:
                    df = pd.read_csv(input_path)

                    # Create the plot
                    fig, axes = plt.subplots(len(features), 1, figsize=(10, 15), sharex=True)
                    
                    # If only one feature, axes isn't a list, so we fix that
                    if len(features) == 1: axes = [axes]

                    for i, feature in enumerate(features):
                        if feature in df.columns:
                            axes[i].plot(df[time_col], df[feature], label=feature)
                            axes[i].set_ylabel(feature)
                            axes[i].legend(loc='upper right')
                            axes[i].grid(True, linestyle='--', alpha=0.7)
                        else:
                            axes[i].set_title(f"Feature '{feature}' not found")

                    axes[-1].set_xlabel('Time (s)')
                    plt.suptitle(f"Features vs Time: {file}", fontsize=16)
                    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

                    # Save and close
                    plt.savefig(output_file)
                    plt.close(fig) 
                    files_processed += 1
                    
                except Exception as e:
                    print(f"  Error processing {file}: {e}")

    print(f"\nDone! Processed {files_processed} files.")

# --- CHECK YOUR FOLDER NAME HERE ---
# Based on your previous message, the folder is likely:
input_directory = r'data/reduce_row_number_cobyla' 
output_directory = r'data_plots_cobyla'

process_and_plot(input_directory, output_directory)