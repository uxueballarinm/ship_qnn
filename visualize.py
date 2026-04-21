import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def process_and_plot(input_root, output_root):
    # Features to plot
    features = ['Rudder Angle (deg)', 'Yaw Rate', 'Yaw Angle', 'Surge Velocity', 'Sway Velocity']
    time_col = 'Time (s)'

    # Walk through the input directory
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.endswith('.csv'):
                # 1. Construct input path
                input_path = Path(root) / file
                
                # 2. Construct output path (replicate subfolder structure)
                relative_path = input_path.relative_to(input_root)
                output_dir = Path(output_root) / relative_path.parent
                output_dir.mkdir(parents=True, exist_ok=True)
                
                output_file = output_dir / f"{input_path.stem}.png"

                print(f"Processing: {input_path} -> {output_file}")

                try:
                    # 3. Read the data
                    df = pd.read_csv(input_path)

                    # 4. Create the plot
                    fig, axes = plt.subplots(len(features), 1, figsize=(10, 15), sharex=True)
                    
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

                    # 5. Save the plot and close to free memory
                    plt.savefig(output_file)
                    plt.close(fig) 
                    
                except Exception as e:
                    print(f"Error processing {file}: {e}")

# Define your paths here
# Use r'' for windows paths to handle backslashes correctly
input_directory = r'data\reduce_row_number_absolutes'
output_directory = r'data_plots'

process_and_plot(input_directory, output_directory)
print("Done!")