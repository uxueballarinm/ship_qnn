from utils import *
import torch
import os
import datetime
import argparse
from copy import deepcopy

def run_classical(args):
    # Setup
    timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Fix seed
    seed = args.run
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    print(f"Running Classical Experiment on: {device} | Seed: {seed}")
    
    # 1. SETUP PATHS (Assuming standard subfolders like QNN)
    print(f"\n[Data Loading] Root Directory: {args.data}")
    train_dir = os.path.join(args.data, "train")
    val_dir = os.path.join(args.data, "validation")
    test_dir = os.path.join(args.data, "test")

    # 2. LOAD DATASETS (Using the new directory-based prepare_dataset)
    print("\nProcessing TRAINING Set...")
    x_train, y_train, x_scaler, y_scaler = prepare_dataset_from_directory(
        train_dir, args, fit_scalers=True
    )
    
    print("Processing VALIDATION Set...")
    x_val, y_val, _, _ = prepare_dataset_from_directory(
        val_dir, args, x_scaler=x_scaler, y_scaler=y_scaler, fit_scalers=False
    )
    
    print("Processing TEST Set...")
    x_test, y_test, _, _ = prepare_dataset_from_directory(
        test_dir, args, x_scaler=x_scaler, y_scaler=y_scaler, fit_scalers=False
    )

    print(f"\nData Shapes:")
    print(f"  Train: X={x_train.shape}, Y={y_train.shape}")
    print(f"  Val:   X={x_val.shape},   Y={y_val.shape}")
    print(f"  Test:  X={x_test.shape},  Y={y_test.shape}")

    # Set metadata for logging
    args.data_n = x_train.shape[0] + x_val.shape[0] + x_test.shape[0]
    args.data_dt = 4.5 # Standard ship resolution

    # 3. MODEL SETUP
    input_dim = x_train.shape[2] 
    output_dim = y_train.shape[2] 
    
    # Target flattening for training (N, Horizon, T) -> (N, Horizon*T)
    # PyTorch logic requires flattened targets for basic MSELoss
    y_train_flat = y_train.reshape(y_train.shape[0], -1)

    raw_model = ClassicalLSTM(
        input_size=input_dim,
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        output_size=y_train.shape[1] * output_dim, # Total flattened outputs
        seed=seed
    )
    
    # Compatibility Wrapper for evaluate_model
    wrapper = ClassicalWrapper(raw_model, device, output_shape=y_train.shape)
    
    num_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    print(f"\nTotal Trainable Parameters: {num_params}")

    # 4. TRAINING
    results = train_classical_model(args, raw_model, x_train, y_train_flat, x_val, y_val, y_scaler=y_scaler, device=device)

    # ==============================================================================
    # 5. MODEL SELECTION (VALIDATION SET) - Unified Logic
    # ==============================================================================
    print("\n[Model Selection] Comparing 'Best' vs 'Final' weights on VALIDATION set...")
    
    val_eval_best = evaluate_model(args, wrapper, results['best_weights'], x_val, y_val, x_scaler, y_scaler)
    val_eval_final = evaluate_model(args, wrapper, results['final_weights'], x_val, y_val, x_scaler, y_scaler)
    
    metric_key = 'Global_open_MSE'
    score_best = val_eval_best['metrics'][metric_key]
    score_final = val_eval_final['metrics'][metric_key]
    
    print(f"  > Validation {metric_key}: Best Weights={score_best:.5f} | Final Weights={score_final:.5f}")

    if score_best < score_final:
        selected_weights_type = "Best (Lowest Val Loss)"
        selected_weights = results['best_weights']
        selected_val_metrics = val_eval_best
        print(f"  > Selected: BEST weights")
    else:
        selected_weights_type = "Final (Last Iteration)"
        selected_weights = results['final_weights']
        selected_val_metrics = val_eval_final
        print(f"  > Selected: FINAL weights")

    # ==============================================================================
    # 6. FINAL TESTING (TEST SET)
    # ==============================================================================
    print("\n[Testing] Evaluating SELECTED weights on TEST set...")
    test_eval = evaluate_model(args, wrapper, selected_weights, x_test, y_test, x_scaler, y_scaler)
    
    # Inject selected weights into results dict for saving
    results['selected_weights'] = selected_weights

    # 7. SAVE RESULTS
    scalers = [x_scaler, y_scaler]
    saved_file = save_classical_results(
        args, 
        results, 
        selected_val_metrics, 
        test_eval, 
        scalers=scalers, 
        timestamp=timestamp,
        selection_type=selected_weights_type
    )
    
    # Load and Print Summary
    load_experiment_results(saved_file)

    # 8. PLOTTING
    if args.save_plot or args.show_plot:
        fig_dir = saved_file.replace("models", "figures").replace(".pkl", "")
        os.makedirs(fig_dir, exist_ok=True)

        plot_convergence(args, results, filename=f"{fig_dir}/plot_convergence.png")
        plot_kinematics_branches(args, test_eval, filename=f"{fig_dir}/plot_branches_local.png")
        plot_kinematics_boxplots(args, test_eval, mode='local', filename=f"{fig_dir}/plot_horizon_errors")
        plot_kinematics_time_series(args, test_eval, loop='open', filename=f"{fig_dir}/plot_trajectory_open.png")
        plot_kinematics_errors(args, test_eval, loop='open', filename=f"{fig_dir}/compare_error/plot_error_vs_time")
        plot_kinematics_time_series(args, test_eval, loop='closed', filename=f"{fig_dir}/plot_trajectory_closed.png")
        plot_kinematics_errors(args, test_eval, loop='closed', filename=f"{fig_dir}/compare_error/plot_error_vs_time")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Data & Task (Matches your QNN parser)
    parser.add_argument('--data', type=str, default="dataset")
    parser.add_argument('-select', '--select_features', type=str, nargs='+', help="Explicitly select features")
    parser.add_argument('-drop', '--drop_features', type=str, nargs='+', help="Drop features")
    parser.add_argument('-ws', '--window_size', type=int, default=5)
    parser.add_argument('-y', '--horizon', type=int, default=5)
    parser.add_argument('--predict', type=str, default='delta', choices=['delta', 'motion'])
    parser.add_argument('--norm', type=str2bool, default=True, choices=[True, False])
    parser.add_argument('--weights', type=str, default="[1.0, 1.0, 1.0, 1.0]")
    
    # Classical Model Hyperparameters
    parser.add_argument('--model', type=str, default='classical')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--hidden_size', type=int, default=16)
    parser.add_argument('--layers', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--maxiter', type=int, default=100, help="Epochs")
    parser.add_argument('--patience', type=int, default=50)
    
    # Flags
    parser.add_argument('-rt', '--reconstruct_train', type=str2bool, default=False)
    parser.add_argument('-rv', '--reconstruct_val', type=str2bool, default=False) 
    parser.add_argument('--show_plot', type=str2bool, default=False)
    parser.add_argument('--save_plot', type=str2bool, default=True)
    parser.add_argument('--run', type=int, default=0)
    
    args = parser.parse_args()
    run_classical(args)