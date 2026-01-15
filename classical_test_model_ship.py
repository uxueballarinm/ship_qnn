import argparse
import os
import pickle
import torch
import numpy as np
import pandas as pd
import sys

# Import utilities
try:
    from utils import *
except ImportError:
    print("Error: Could not import necessary modules. Ensure 'qnn_utils.py' and 'classical_run.py' are in the same directory.")
    sys.exit(1)

class Args:
    """Helper to convert dict to object for compatibility with utils"""
    def __init__(self, **entries):
        self.__dict__.update(entries)

def run_test_classical(cli_args):
    
    # --- 2. PRESERVE CLI ARGUMENTS ---
    # We save these before overwriting 'args' with the training config
    model_path = cli_args.model
    data_path = cli_args.data
    use_final_weights = cli_args.final
    show_plot_flag = cli_args.show_plot
    save_plot_flag = cli_args.save_plot
    
    if not os.path.exists(model_path): return print(f"Error: Model not found at {model_path}")
    if not os.path.exists(data_path): return print(f"Error: Data not found at {data_path}")

    # 1. LOAD PICKLE
    print(f"Loading classical model from: {model_path}")
    with open(model_path, 'rb') as f:
        saved_data = pickle.load(f)

    # 2. MERGE CONFIG
    # Load training config
    args_dict = saved_data['config']
    
    # Inject runtime flags into the config so the Utils functions work
    args_dict['show_plot'] = show_plot_flag
    args_dict['save_plot'] = save_plot_flag
    
    # Convert to object
    args = Args(**args_dict)
    
    # Pre-fitted scalers
    x_scaler = saved_data['x_scaler'] 
    y_scaler = saved_data['y_scaler'] 
    
    # Retrieve weights
    weights = saved_data['final_weights'] if use_final_weights else saved_data['best_weights']
    
    print(f"Loaded Config: F={len(args.features)}, W={args.window_size}, H={args.horizon}")

    # 3. PREPARE TEST DATA
    df = pd.read_csv(data_path, index_col=0)
    df['delta_x'] = df['position_x'].diff().fillna(0)
    df['delta_y'] = df['position_y'].diff().fillna(0)

    # Feature Mapping (Robustness)
    cols = map_features(args.features)
    pred_cols = ["delta_x", "delta_y"] if args.predict == "delta" else ["position_x", "position_y"]
    
    # Extract and Scale
    feature_seqs = df[cols].to_numpy()
    prediction_seqs = df[pred_cols].to_numpy()
    
    feat_norm = x_scaler.transform(feature_seqs) 
    
    if y_scaler: 
        pred_norm = y_scaler.transform(prediction_seqs)
    else: 
        pred_norm = prediction_seqs

    # Fold Logic
    x_folds, y_folds = make_sliding_window_ycustom_folds(feat_norm, pred_norm, args.window_size, args.horizon)
    x_test = x_folds[args.testing_fold]
    y_test = y_folds[args.testing_fold]

    # 4. INITIALIZE CLASSICAL MODEL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference Device: {device}")

    input_dim = x_test.shape[2] 
    output_dim = args.horizon * 2 

    model = ClassicalLSTM(
        input_size=input_dim,
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        output_size=output_dim,
        seed=args.run
    )
    
    wrapper = ClassicalWrapper(model, device)
    
    # 5. EVALUATE
    print("Running Evaluation...")
    results = evaluate_model(args, wrapper, weights, x_test, y_test, x_scaler, y_scaler)

    # 6. PLOT
    base_name = os.path.basename(model_path).replace('.pkl', '')
    
    if not os.path.exists("figures"):
        os.makedirs("figures")
    
    save_dir = f"figures/{base_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print(f"Saving plots to {save_dir}...")
    plot_horizon_branches(args, results, filename=f"{save_dir}/plot_branches.png")
    plot_horizon_euclidean_boxplots(args, results, mode='local', filename=f"{save_dir}/plot_horizon_errors")
    plot_trajectory_components(args, results, filename=f"{save_dir}/plot_trajectory.png")
    plot_errors_and_position_time(args, results, filename=f"{save_dir}/plot_error_vs_time")

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, choices=['old', 'new'], default='old')
    parser.add_argument('--model', type=str, required=True, help="Path to classical .pkl file")
    parser.add_argument('--data', type=str, default="datasets/zigzag_10_40_ood_reduced_2_s.csv")
    parser.add_argument('--final', action='store_true', help="Use final weights instead of best weights")
    parser.add_argument('--show_plot', type=str2bool, default=False, help="Show plots interactively")
    parser.add_argument('--save_plot', type=str2bool, default=True, help="Save plots to files")
    
    args = parser.parse_args()
    
    if args.version == 'old': 
        run_test_classical(args)
    else: 
        # Ensure load_experiment_results is imported from utils
        load_experiment_results(args.model)