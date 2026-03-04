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
from qnn_utils import (
    prepare_dataset_from_directory, 
    evaluate_model, 
    save_classical_results,
    str2bool,
    map_names,
    ClassicalLSTM, 
    ClassicalWrapper,
    C_GREEN, C_BLUE, C_YELLOW, C_RED, C_RESET
)

def train_classical(args):
    timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure save_dir attribute exists for compatibility with save_classical_results
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

    train_loader = DataLoader(TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train)), 
                              batch_size=args.batch_size, shuffle=True)

    # 2. MODEL INITIALIZATION
    input_dim = x_train.shape[2]
    num_target_features = y_train.shape[2] 
    # Output dim is flat (Horizon * Features) per your ClassicalLSTM in qnn_utils
    output_dim_flat = args.horizon * num_target_features 

    print(f"{C_BLUE}[Model] LSTM Hidden: {args.hidden_size} | Layers: {args.layers} | Output: {output_dim_flat}{C_RESET}")
    
    model = ClassicalLSTM(
        input_size=input_dim, 
        hidden_size=args.hidden_size, 
        num_layers=args.layers, 
        output_size=output_dim_flat, 
        seed=seed
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()

    # 3. TRAINING LOOP
    best_val_loss = float('inf')
    train_hist, val_hist = [], []

    print(f"{C_YELLOW}Training for {args.maxiter} epochs...{C_RESET}")
    for epoch in range(args.maxiter):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            
            pred = model(xb)
            target_flat = yb.view(yb.size(0), -1) # Match linear output
            
            loss = criterion(pred, target_flat)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        
        avg_train = np.mean(epoch_losses)
        train_hist.append(avg_train)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.Tensor(x_val).to(device))
            val_target_flat = torch.Tensor(y_val).to(device).view(y_val.shape[0], -1)
            val_loss = criterion(val_pred, val_target_flat).item()
            val_hist.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = deepcopy(model.state_dict())

        if epoch % 20 == 0 or epoch == args.maxiter - 1:
            print(f"  Epoch {epoch:4d} | Train MSE: {avg_train:.6f} | Val MSE: {val_loss:.6f}")

    # 4. EVALUATION (Matches Quantum Logic)
    model.load_state_dict(best_weights)
    wrapper = ClassicalWrapper(model, device, output_shape=y_test.shape)
    
    print(f"{C_YELLOW}Running Final Evaluation (Open & Closed Loop)...{C_RESET}")
    val_eval = evaluate_model(args, wrapper, best_weights, x_val, y_val, x_scaler, y_scaler)
    test_eval = evaluate_model(args, wrapper, best_weights, x_test, y_test, x_scaler, y_scaler)

    # 5. SAVING (Uses your specific save_folder)
    train_results = {
        'train_history': train_hist, 
        'val_history': val_hist, 
        'best_weights': best_weights, 
        'final_weights': model.state_dict()
    }
    
    # This utility now uses args.save_dir (which we set to args.save_folder) 
    # to organize models, logs, and figures.
    save_classical_results(args, train_results, val_eval, test_eval, (x_scaler, y_scaler), timestamp)
    print(f"{C_GREEN}[Success] Results saved to {args.save_folder}{C_RESET}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classical LSTM Ship Dynamics Trainer")
    parser.add_argument('--config_file', type=str, default=None)
    parser.add_argument('--save_folder', type=str, default="classical_baselines")
    parser.add_argument('--data', type=str, default="data/reduce_row_number_absolutes")
    parser.add_argument('--select_features', nargs='+', default=['sv','wv','yr','ya','rarad'])
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--predict', type=str, default='motion')
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--maxiter', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--run', type=int, default=1)
    parser.add_argument('--norm', type=str2bool, default=True)
    parser.add_argument('--reconstruct_train', type=str2bool, default=False)
    parser.add_argument('--reconstruct_val', type=str2bool, default=False)
    parser.add_argument('--weights', type=str, default="[1.0, 5.0, 5.0, 2.0]")
    
    args = parser.parse_args()

    if args.config_file:
        with open(args.config_file, 'r') as f:
            experiments = yaml.safe_load(f)
        for i, exp_config in enumerate(experiments):
            print(f"\n{C_GREEN}--- Experiment {i+1}/{len(experiments)} ---{C_RESET}")
            exp_args = deepcopy(args)
            for k, v in exp_config.items():
                setattr(exp_args, k, v)
            train_classical(exp_args)
    else:
        train_classical(args)