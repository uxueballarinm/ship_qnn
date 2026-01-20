from utils import *

def run(args):
    timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    
    # Fix seed
    seed = args.run
    random.seed(seed)
    np.random.seed(seed)

    # Load data
    num_folds = 4
    df = pd.read_csv(args.data, index_col=0)

    # Extract dataset info
    args.data_n = len(df)
    timestamps = pd.Series(df.index)
    dt_avg = timestamps.diff().median()
    args.data_dt = float(f"{dt_avg:.4f}")
    print(f"\nDataset: N = {args.data_n} | dt = {args.data_dt} s")


    df['delta Surge Velocity'] = df['Surge Velocity'].diff().fillna(0)
    df['delta Sway Velocity'] = df['Sway Velocity'].diff().fillna(0)
    df['delta Yaw Rate'] = df['Yaw Rate'].diff().fillna(0)
    df['delta Yaw Angle'] = df['Yaw Angle'].diff().fillna(0)

    control_cols = ["Rudder Angle (deg)", "Rudder Angle (rad)"] # Add whatever names you use
    
    for col in control_cols:
        if col in df.columns:
            # Shift (-1 means move next row's value to current row)
            df[col] = df[col].shift(-1) 
            print(f"Shifted feature '{col}' by -1 step (Predicting t using Action t)")

    # 3. Drop the last row (which is now NaN because of the shift)
    df.dropna(inplace=True)

    select_list = getattr(args, 'select_features', None)
    drop_list = getattr(args, 'drop_features', None)
    if select_list:
        args.features = map_names(args.select_features)
    elif drop_list:
        args.features = [f for f in full_feature_set if f not in map_names(args.drop_features)]
    else:
        args.features = full_feature_set

    print(f"{len(args.features)} features selected: {args.features}")

    if args.predict == "motion": args.targets = ["Surge Velocity","Sway Velocity","Yaw Rate","Yaw Angle"]
    elif args.predict == "delta": args.targets = ["delta Surge Velocity", "delta Sway Velocity", "delta Yaw Rate", "delta Yaw Angle"]
    feature_seqs, prediction_seqs = get_seqs(df, args.features, args.targets)

    # Train/test folds split indices
    split_indices = get_fold_indices(len(feature_seqs), num_folds)
    test_start, test_end = split_indices[args.testing_fold], split_indices[args.testing_fold + 1]
    train_val_indices = np.delete(np.arange(len(feature_seqs)), np.arange(test_start, test_end))

    # Train/val split for train mask
    val_size = 0.15 #TODO: Maybe increase?
    train_mask = train_val_indices[:int(len(train_val_indices)*(1 - val_size))]

    x_scaler = MinMaxScaler(feature_range=(0, np.pi))
    x_scaler.fit(feature_seqs[train_mask])
    feature_seqs_norm = x_scaler.transform(feature_seqs)
    feature_seqs_norm = np.clip(feature_seqs_norm, 0, np.pi)

    # Normalization
    if args.norm:
        y_scaler = MinMaxScaler(feature_range=(-1, 1)) # for (N, 2) data
        y_scaler.fit(prediction_seqs[train_mask])
        prediction_seqs_norm = y_scaler.transform(prediction_seqs)
    else:
        y_scaler = None
        prediction_seqs_norm = prediction_seqs
    
    # Windowing
    x_data_folds, y_data_folds = make_sliding_window_ycustom_folds(feature_seqs_norm, prediction_seqs_norm, args.window_size, args.horizon, num_folds=num_folds)

    # Split already windowed data
    x_test = x_data_folds[args.testing_fold]
    y_test = y_data_folds[args.testing_fold]
    train_val_fold_indices = [i for i in range(num_folds) if i != args.testing_fold]
    x_train_val = np.concatenate([x_data_folds[i] for i in train_val_fold_indices], axis=0)
    y_train_val = np.concatenate([y_data_folds[i] for i in train_val_fold_indices], axis=0)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=val_size, shuffle=False, random_state=seed)

    # Construct QNN
    qc,input_params, weight_params = create_multivariate_circuit(args)
    backend = AerSimulator(seed_simulator=seed)
    estimator_options = {"run_options": {"shots": None, "seed": seed}, "backend_options": {"seed_simulator": seed}}
    estimator = Estimator(options=estimator_options)
    obsvs = [SparsePauliOp('I' * (qc.num_qubits - 1 - i) + 'Z' + 'I' * i) for i in range(qc.num_qubits)]
    estimator_qnn = EstimatorQNN(
        circuit=qc, 
        input_params=input_params, 
        weight_params=weight_params, 
        observables=obsvs, 
        estimator=estimator, 
        input_gradients=False,  
        pass_manager=generate_preset_pass_manager(backend=backend, optimization_level=1, seed_transpiler=seed), #TODO: Change optimization level?
        default_precision=0.0
    )

    # Instantiate model
    model = WindowEncodingQNN(estimator_qnn, y_train.shape, seed)
    
    # Training
    results = train_model(args, model, x_train, y_train, x_val, y_val, y_scaler)

    # Plot Convergence with Dual Axes
    short_args = map_names([args.ansatz, args.entangle], reverse=True)
    fig_dir = f"figures/{timestamp}_{args.model}_f{len(args.features)}_w{args.window_size}_h{args.horizon}_{short_args[0]}_{short_args[1]}_r{args.reps}"
    os.makedirs(fig_dir, exist_ok=True)
    plot_convergence(args, results, filename=f"{fig_dir}/plot_convergence.png")

    # Evaluation
    print("\nEvaluating on Test Set...")
    
    # --- BEST WEIGHTS ---
    print("...with best weights")
    best_eval = evaluate_model(args, model, results['best_weights'], x_test, y_test, x_scaler, y_scaler)

    # --- FINAL WEIGHTS ---
    print("...with final weights")
    final_eval = evaluate_model(args, model, results['final_weights'], x_test, y_test, x_scaler, y_scaler)

    
    # Local plots
    plot_kinematics_branches(args, final_eval, filename=f"{fig_dir}/plot_branches_local.png")
    plot_kinematics_boxplots(args, final_eval, mode='local', filename=f"{fig_dir}/plot_horizon_errors")

    # Global Open Plots
    plot_kinematics_time_series(args, final_eval, loop = 'open', filename=f"{fig_dir}/plot_kinematics_open.png")
    plot_kinematics_errors(args, final_eval, loop = 'open', filename=f"{fig_dir}/compare_error/plot_error_vs_time")

    #Global Closed Plots
    plot_kinematics_time_series(args, final_eval, loop = 'closed', filename=f"{fig_dir}/plot_kinematics_closed.png")
    plot_kinematics_errors(args, final_eval, loop = 'closed', filename=f"{fig_dir}/compare_error/plot_error_vs_time")
    
    scalers = [x_scaler, y_scaler]
    qnn_dict = {"qc": qc, "input_params": input_params, "weight_params": weight_params}
    save_experiment_results(args, results, best_eval, final_eval, scalers, qnn_dict, timestamp)
    load_experiment_results(f"models/{timestamp}_{args.model}_f{len(args.features)}_w{args.window_size}_h{args.horizon}_{short_args[0]}_{short_args[1]}_r{args.reps}.pkl")

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    # Data structure
    parser.add_argument('--data', type=str, default="datasets\zigzag_11_11_ind_reduced_2_s.csv") #TODO: Try full dataset
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-select', '--select_features', type=str, default=['dwv','dsvdyr','dya','rarad'], nargs='+', help="Explicitly select features (e.g. --select_features wv sv yr ya)")
    group.add_argument('-drop', '--drop_features', type=str, nargs='+', help="Drop features from default set (e.g. --drop_features rarad)")
    parser.add_argument('-ws', '--window_size', type=int, default=5, help="Window size = num qubits")
    parser.add_argument('-y', '--horizon', type=int, default=5) # 1,3,5
    parser.add_argument('-t', '--testing_fold', type=int, default=3, help="Testing fold number")
    
    # Target Options
    parser.add_argument('--predict', type=str, default='delta', choices=['delta', 'motion'], help="Target: 'delta' (simple steps) or 'motion' (kinematic variables)")
    parser.add_argument('--norm', type = str2bool, default=True, choices = [True, False], help="Normalize targets to [-1, 1]") # Don't normalize NOTE maybe normalize for a classical layer that goes before the redout if we add 
    parser.add_argument('-rt', '--reconstruct_train', type = str2bool, choices=[True, False], default=False, help="If True, calculates loss on the reconstructed trajectory (meters)")
    parser.add_argument('-rv', '--reconstruct_val', type = str2bool, choices=[True, False], default=False, help="If True, calculates loss on the reconstructed trajectory (meters)")
    # parser.add_argument('--plot', type=str, default='short', choices=['short', 'long'], help="Plot 'short' (based on previous true kinematic variables) or 'long' (cumulative) prediction")

    # QNN model
    parser.add_argument('--map', type = str,nargs='+', help="Specific order of feature indices (e.g. 2 0 1)")  
    parser.add_argument('--reorder', type=str2bool, default=True, choices=[True, False])
    parser.add_argument('--encoding', type=str, default='compact', choices=['compact', 'serial', 'parallel'], help="Strategy for multi-feature encoding")
    parser.add_argument('--entangle', type=str, default='reverse_linear', choices=['full', 'linear', 'reverse_linear', 'circular', 'sca']) # reverse_linear
    parser.add_argument('--ansatz', type=str, default='ugates', choices=['ugates', 'efficientsu2', 'realamplitudes']) # ugates
    parser.add_argument('--reps', type=int, default=3) # 1,3,5,7
    parser.add_argument('-init', '--initialization', type=str, default='uniform', choices=['uniform', 'identity'])# uniform
    parser.add_argument('--model', type=str, default='vanilla', choices=['vanilla', 'hybrid']) # vanilla

    # Optimization
    parser.add_argument('-opt','--optimizer', type=str, default='cobyla', choices=['cobyla', 'spsa']) # cobyla
    parser.add_argument('--maxiter', type=int, default=10000)
    parser.add_argument('-tol', '--tolerance', type=float, default=None)
    # parser.add_argument('-lr','--learning_rate', type = float, default = 0.01)
    # parser.add_argument('-p','--perturbation',type = float, default = 0.01)

    parser.add_argument('--show_plot', type = str2bool, default=False)
    parser.add_argument('--save_plot', type = str2bool, default=True)
    parser.add_argument('--run', type=int, default=0)
    args = parser.parse_args()
    run(args)

    #NOTE: Add two metrics in validation
    # - 