import os
import yaml
import torch
import torch.nn as nn
import datetime
import argparse
import numpy as np
from copy import deepcopy
from torch.utils.data import TensorDataset, DataLoader

# Import everything from your central utilities
from qnn_utils_adam import *

def train_classical(args):
    timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not hasattr(args, 'save_dir'):
        setattr(args, 'save_dir', args.save_folder)

    # Reproducibility
    seed = getattr(args, 'run', 1)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. DATA LOADING
    print(f"{C_BLUE}[Data] Loading from: {args.data}{C_RESET}")
    train_dir = os.path.join(args.data, "train")
    val_dir = os.path.join(args.data, "validation")
    test_dir = os.path.join(args.data, "test")

    x_train, y_train, x_scaler, y_scaler = prepare_dataset_from_directory(train_dir, args, fit_scalers=True)
    x_val, y_val, _, _ = prepare_dataset_from_directory(val_dir, args, x_scaler=x_scaler, y_scaler=y_scaler)
    x_test, y_test, _, _ = prepare_dataset_from_directory(test_dir, args, x_scaler=x_scaler, y_scaler=y_scaler)

    # 2. MODEL INITIALIZATION
    args.features = map_names(args.select_features)
    input_dim = x_train.shape[2]
    num_target_features = y_train.shape[2] 
    output_dim_flat = args.horizon * num_target_features 

    print(f"{C_BLUE}[Model] MLP Hidden: {args.hidden_size} | Layers: {args.layers} | Output: {output_dim_flat}{C_RESET}")
    if getattr(args, 'model', 'vanilla') == 'multihead':
        print(f"{C_BLUE}[Model] Constructing Classical Multi-Head MLP...{C_RESET}")
        model = ClassicalMultiHeadMLP(
            heads_config=args.heads_config,
            features=args.features,
            window_size=args.window_size,
            horizon=args.horizon,
            seed=seed
        ).to(device)
    else:
        print(f"{C_BLUE}[Model] Constructing Standard Single-Head MLP...{C_RESET}")
        model = ClassicalMLP(
            input_size=input_dim,
            window_size=args.window_size,
            hidden_size=args.hidden_size,
            num_layers=args.layers,
            output_size=output_dim_flat,
            seed=seed,
            horizon=args.horizon,             # NEW
            num_targets=num_target_features   # NEW
        ).to(device)
    model_wrapper = ClassicalWrapper(model, device, output_shape=y_train.shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  > [Efficiency] Total Classical Parameters: {total_params}")
    

    # 3. TRAINING
    if args.optimizer.lower() == 'adam':
        results = train_classical_model(args, model, x_train, y_train, x_val, y_val, y_scaler, device)
        model.load_state_dict(results['best_weights'])
        best_w = model_wrapper.get_weights()
    else:
        results = train_model(args, model_wrapper, x_train, y_train, x_val, y_val, y_scaler)
        best_w = results.get('best_weights')
    
    # 4. PREPARE WEIGHTS FOR EVALUATION
    # Capture final state before overwriting with best weights
    final_state_dict = deepcopy(model.state_dict())
    

    
    print(f"{C_YELLOW}Running Final Evaluation (Open & Closed Loop)...{C_RESET}")
    # Evaluation must use the weights intended for the benchmark (the best ones)
    val_eval = evaluate_model(args, model_wrapper, best_w, x_val, y_val, x_scaler, y_scaler)
    test_eval = evaluate_model(args, model_wrapper, best_w, x_test, y_test, x_scaler, y_scaler)

    # Convert best weights to state_dict for saving
    model_wrapper.set_weights(best_w)
    best_state_dict = deepcopy(model.state_dict())
    
    # 5. ORGANIZE AND SAVE
    train_results = {
        'train_history': results.get('train_history', []), 
        'val_history': results.get('val_history', []), 
        'best_weights': best_state_dict,  
        'final_weights': final_state_dict
    }
    
    # Read excel_path from args if it exists (passed from YAML)
    custom_excel = getattr(args, 'excel_path', None)
    save_classical_results(args, train_results, val_eval, test_eval, (x_scaler, y_scaler), timestamp, excel_path=custom_excel)    
    print(f"{C_GREEN}[Success] Results saved to {args.save_folder}{C_RESET}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classical LSTM Ship Dynamics Trainer")
    parser.add_argument('--config_file', type=str, default=None)
    parser.add_argument('--indices', type=int, nargs='+', default=None, help="Specific experiment indices to run from the config file (1-based)")
    parser.add_argument('--save_folder', type=str, default="classical_baselines")
    parser.add_argument('--data', type=str, default="data/reduce_row_number_absolutes")
    parser.add_argument('--select_features', nargs='+', default=['sv','wv','yr','ya','rarad'])
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--predict', type=str, default='motion')
    parser.add_argument('--hidden_size', type=int, default=4)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--maxiter', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, nargs='+', default=[0.05, 0.005], help="Initial and final learning rate for dynamic decay")
    parser.add_argument('--run', type=int, default=1)
    parser.add_argument('--norm', type=str2bool, default=True)
    parser.add_argument('--reconstruct_train', type=str2bool, default=False)
    parser.add_argument('--reconstruct_val', type=str2bool, default=False)
    parser.add_argument('--weights', type=str, default="[1.0, 1.0, 1.0, 1.0]")
    parser.add_argument('--optimizer', type=str, default='spsa', choices=['spsa', 'cobyla'])
    parser.add_argument('--perturbation', type=float, default=0.05)
    parser.add_argument('--tolerance', type=float, default=None)
    parser.add_argument('-init', '--initialization', type=str, default='uniform', choices=['uniform', 'identity'])# uniform
    parser.add_argument('--save_plot', type=str2bool, default=True)
    parser.add_argument('--show_plot', type=str2bool, default=False)
    
    args = parser.parse_args()

    if args.config_file:
        with open(args.config_file, 'r') as f:
            config_content = yaml.safe_load(f)
            
        config_list = config_content if isinstance(config_content, list) else [config_content]
        
        # If we have specific indices (like from your Slurm script)
        if args.indices:
            # We iterate through whatever indices were passed. 
            # In your Slurm case, this will usually just be ONE number.
            for idx_to_run in range (args.indices[0], args.indices[1]):
                target_idx = idx_to_run - 1 # Convert 1-based to 0-based
                if 0 <= target_idx < len(config_list):
                    print(f"\n{C_BLUE}--- [Targeted Run] Experiment {idx_to_run}/{len(config_list)} ---{C_RESET}")
                    config_dict = config_list[target_idx]
                    current_args = argparse.Namespace(**vars(args))
                    for k, v in config_dict.items():
                        setattr(current_args, k, v)
                    train_classical(current_args)
                else:
                    print(f"{C_RED}Error: Index {idx_to_run} is out of range.{C_RESET}")
        else:
            # Traditional behavior: Run everything in the file
            print(f"No indices specified. Running all {len(config_list)} experiments.")
            for i, config_dict in enumerate(config_list):
                print(f"\n--- Running Experiment {i+1}/{len(config_list)} ---")
                current_args = argparse.Namespace(**vars(args))
                for k, v in config_dict.items():
                    setattr(current_args, k, v)
                train_classical(current_args)