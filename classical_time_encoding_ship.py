from utils import *

def run_classical(args):

    # Setup
    timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Classical Experiment on: {device}")
    
    # --- DATA LOADING (Same as QNN) ---
    df = pd.read_csv(args.data, index_col=0)
    
    # Extract dataset info
    args.data_n = len(df)
    timestamps = pd.Series(df.index)
    dt_avg = timestamps.diff().median()
    args.data_dt = float(f"{dt_avg:.4f}")
    print(f"\nDataset: N = {args.data_n} | dt = {args.data_dt} s")

    df['delta x'] = df['Position X'].diff().fillna(0)
    df['delta y'] = df['Position Y'].diff().fillna(0)

    # Feature selection
    select_list = getattr(args, 'select_features', None)
    drop_list = getattr(args, 'drop_features', None)
    if select_list:
        args.features = map_features(args.select_features)
    elif drop_list:
        args.features = [f for f in full_feature_set if f not in map_features(args.drop_features)]
    else:
        args.features = full_feature_set

    print(f"{len(args.features)} features selected: {args.features}")

    # Feature Map
    args.targets = ["delta x", "delta y"] if args.predict == "delta" else ["Position X", "Position Y"]
    
    print(f"Features: {args.features}")
    feature_seqs, prediction_seqs = get_seqs(df, args.features, args.targets)

    # Split
    num_folds = 4
    split_indices = get_fold_indices(len(feature_seqs), num_folds)
    test_start, test_end = split_indices[args.testing_fold], split_indices[args.testing_fold + 1]
    train_val_indices = np.delete(np.arange(len(feature_seqs)), np.arange(test_start, test_end))
    train_mask = train_val_indices[:int(len(train_val_indices)*(1 - 0.15))]

    # Scaling
    x_scaler = MinMaxScaler(feature_range=(0, 1)) # Standard [0,1] for classical
    x_scaler.fit(feature_seqs[train_mask])
    x_norm = x_scaler.transform(feature_seqs)

    if args.norm:
        y_scaler = MinMaxScaler(feature_range=(-1, 1))
        y_scaler.fit(prediction_seqs[train_mask])
        y_norm = y_scaler.transform(prediction_seqs)
    else:
        y_scaler = None
        y_norm = prediction_seqs

    # Windowing
    x_folds, y_folds = make_sliding_window_ycustom_folds(x_norm, y_norm, args.window_size, args.horizon, num_folds)
    
    # Datasets
    x_test = x_folds[args.testing_fold]
    y_test = y_folds[args.testing_fold]
    
    train_val_indices_list = [i for i in range(num_folds) if i != args.testing_fold]
    x_train_val = np.concatenate([x_folds[i] for i in train_val_indices_list], axis=0)
    y_train_val = np.concatenate([y_folds[i] for i in train_val_indices_list], axis=0)
    
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.15, shuffle=False, random_state=args.run)
    print(f"Train: {x_train.shape} | Val: {x_val.shape} | Test: {x_test.shape}")

    # --- MODEL SETUP ---
    input_dim = x_train.shape[2] # Number of features
    output_dim = y_train.shape[1] * y_train.shape[2] # Horizon * 2 (flattened output usually)
    
    # NOTE: PyTorch LSTM expects output_dim to be features per step, usually 2 for us.
    # But y_train is (N, Horizon, 2).
    # If Horizon > 1, we need to flatten targets or adjust model output.
    # For simplicity, let's flatten targets for training: (N, Horizon*2)
    y_train_flat = y_train.reshape(y_train.shape[0], -1)
    y_val_flat = y_val.reshape(y_val.shape[0], -1)
    
    model = ClassicalLSTM(
        input_size=input_dim,
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        output_size=output_dim, # Flattened output size
        seed=args.run
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Trainable Parameters: {num_params}")
    # --- TRAINING ---
    results = train_classical_model(args, model, x_train, y_train_flat, x_val, y_val_flat, y_scaler=y_scaler,device=device)

    # --- EVALUATION ---
    # We need to wrap the evaluation because 'evaluate_model' expects a QNN-style 'model.forward(x, params)' interface.
    # We will create a dummy wrapper class that behaves like the QNN wrapper.
    


    wrapper = ClassicalWrapper(model, device)

    # Save directories
    fig_dir = f"figures/{timestamp}_classical_f{len(args.features)}_w{args.window_size}_h{args.horizon}"
    os.makedirs(fig_dir, exist_ok=True)

    # Plot Convergence
    plot_convergence(args, results, filename=f"{fig_dir}/{timestamp}_convergence_classical.png")

    print(f"\n--- Evaluation (Best Weights) ---")
    best_eval = evaluate_model(args, wrapper, results['best_weights'], x_test, y_test, x_scaler, y_scaler)

    print(f"\n--- Evaluation (Final Weights) ---")
    final_eval = evaluate_model(args, wrapper, results['final_weights'], x_test, y_test, x_scaler, y_scaler)

    # Local plots
    plot_horizon_branches(args, final_eval, filename=f"{fig_dir}/plot_branches.png")
    plot_horizon_euclidean_boxplots(args, final_eval, mode='local', filename=f"{fig_dir}/plot_horizon_errors")
    #Global Plots
    plot_trajectory_components(args, final_eval, filename=f"{fig_dir}/plot_trajectory.png")
    plot_errors_and_position_time(args, final_eval, filename=f"{fig_dir}/plot_error_vs_time")

    scalers = [x_scaler, y_scaler]
    save_classical_results(args, results, best_eval, final_eval, scalers=scalers, timestamp=timestamp)
    load_experiment_results(f"models/{timestamp}_classical_f{len(args.features)}_w{args.window_size}_h{args.horizon}.pkl")

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    # Data & Task
    parser.add_argument('--data', type=str, default="datasets/zigzag_10_40_ood_reduced_2_s.csv")
    parser.add_argument('-select', '--select_features', type=str, nargs='+', help="Explicitly select features")
    parser.add_argument('-drop', '--drop_features', type=str, nargs='+', help="Drop features")
    parser.add_argument('-ws', '--window_size', type=int, default=5)
    parser.add_argument('-y', '--horizon', type=int, default=5) # FIXED: Was 1, now 5
    parser.add_argument('-t', '--testing_fold', type=int, default=3)
    parser.add_argument('--predict', type=str, default='delta', choices=['delta', 'pos'])
    parser.add_argument('--norm', type=str2bool, default=True, choices=[True, False])
    
    # Classical Model Hyperparameters
    parser.add_argument('--optimizer', type=str, default='Adam', help="Optimizer name (for logging)")
    parser.add_argument('--hidden_size', type=int, default=16, help="LSTM hidden units")
    parser.add_argument('--layers', type=int, default=1, help="Number of LSTM layers")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--maxiter', type=int, default=100, help="Epochs")
    parser.add_argument('--patience', type=int, default=50)
    
    # Flags (Synchronized)
    parser.add_argument('-rt', '--reconstruct_train', type=str2bool, choices=[True, False], default=False)
    parser.add_argument('-rv', '--reconstruct_val', type=str2bool, choices=[True, False], default=False) # FIXED: Was True, now False
    parser.add_argument('--show_plot', type=str2bool, choices=[True, False], default=False)
    parser.add_argument('--save_plot', type=str2bool, choices=[True, False], default=True) # ADDED
    parser.add_argument('--run', type=int, default=0)
    
    args = parser.parse_args()
    run_classical(args)