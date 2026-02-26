import argparse
import os
import pickle
import torch
import numpy as np
import pandas as pd
import sys
from copy import deepcopy

# Import utilities
try:
    from qnn_utils import *
except ImportError:
    print("Error: Could not import necessary modules.")
    sys.exit(1)

class Args:
    """Helper to convert dict to object for compatibility with utils"""
    def __init__(self, **entries):
        self.__dict__.update(entries)

def run_test_classical(cli_args):
    
    # --- 1. PRESERVE CLI ARGUMENTS ---
    model_path = cli_args.model
    data_path = cli_args.data # This should now be a directory containing 'test/'
    force_final = cli_args.final
    show_plot_flag = cli_args.show_plot
    save_plot_flag = cli_args.save_plot
    
    if not os.path.exists(model_path): return print(f"Error: Model not found at {model_path}")
    if not os.path.exists(data_path): return print(f"Error: Data not found at {data_path}")

    # 2. LOAD PICKLE
    print(f"Loading classical model from: {model_path}")
    with open(model_path, 'rb') as f:
        saved_data = pickle.load(f)

    # 3. MERGE CONFIG
    args_dict = saved_data['config']
    args_dict['show_plot'] = show_plot_flag
    args_dict['save_plot'] = save_plot_flag
    args = Args(**args_dict)
    
    x_scaler = saved_data['x_scaler'] 
    y_scaler = saved_data['y_scaler'] 
    
    # --- SMART WEIGHT SELECTION (Validation Logic) ---
    if force_final:
        print("[Info] Forcing usage of FINAL weights (User Request).")
        weights = saved_data.get('final_weights')
    elif 'selected_weights' in saved_data:
        # Respect the "Best vs Final" winner chosen during training
        method = saved_data.get('weight_selection_method', 'Unknown')
        print(f"[Info] Using automatically SELECTED weights (Method: {method})")
        weights = saved_data['selected_weights']
    else:
        # Fallback for old classical models
        print("[Warning] 'selected_weights' not found. Falling back to 'best_weights'.")
        weights = saved_data.get('best_weights', saved_data.get('final_weights'))
    
    print(f"Loaded Config: F={len(args.features)}, W={args.window_size}, H={args.horizon}")

    # --- 4. PREPARE TEST DATA (Directory Mode) ---
    # We now load from the 'test/' subfolder to match QNN logic
    test_dir = os.path.join(data_path, "test") if os.path.isdir(data_path) else data_path
    print(f"Loading Test Data from: {test_dir}")

    # We use the unified prepare_dataset_from_directory to ensure deltas/normalization match exactly
    x_test, y_test, _, _ = prepare_dataset_from_directory(
        test_dir, args, x_scaler=x_scaler, y_scaler=y_scaler, fit_scalers=False
    )
    print(f"Test Data Shape: X={x_test.shape}, Y={y_test.shape}")

    # --- 5. INITIALIZE CLASSICAL MODEL ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = x_test.shape[2] 
    output_dim = y_test.shape[2] # Target dim (usually 4)

    model = ClassicalLSTM(
        input_size=input_dim,
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        output_size=args.horizon * output_dim, # Flattened output size
        seed=args.run
    )

    wrapper = ClassicalWrapper(model, device, output_shape=y_test.shape)
    
    # --- 6. EVALUATE ---
    print("Running Evaluation...")
    results = evaluate_model(args, wrapper, weights, x_test, y_test, x_scaler, y_scaler)

    # --- 7. PLOT ---
    base_name = os.path.basename(model_path).replace('.pkl', '')
    save_dir = f"figures/{base_name}"
    os.makedirs(save_dir, exist_ok=True)
        
    print(f"Saving plots to {save_dir}...")
    
    plot_kinematics_branches(args, results, filename=f"{save_dir}/plot_branches_local.png")
    plot_kinematics_boxplots(args, results, mode='local', filename=f"{save_dir}/plot_horizon_errors")
    plot_kinematics_time_series(args, results, loop='open', filename=f"{save_dir}/plot_kinematics_open.png")
    plot_kinematics_errors(args, results, loop='open', filename=f"{save_dir}/compare_error/plot_error_vs_time")
    plot_kinematics_time_series(args, results, loop='closed', filename=f"{save_dir}/plot_kinematics_closed.png")
    plot_kinematics_errors(args, results, loop='closed', filename=f"{save_dir}/compare_error/plot_error_vs_time")

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, choices=['old', 'new'], default='old')
    parser.add_argument('--model', type=str, required=True, help="Path to classical .pkl file")
    parser.add_argument('--data', type=str, default="dataset", help="Path to root dataset folder")
    parser.add_argument('--final', action='store_true', help="Force use of final weights")
    parser.add_argument('--show_plot', type=str2bool, default=False)
    parser.add_argument('--save_plot', type=str2bool, default=True)
    
    args = parser.parse_args()
    
    if args.version == 'old': 
        run_test_classical(args)
    else: 
        load_experiment_results(args.model)