from utils import *

def run(args):
    timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    
    # Fix seed
    seed = args.run
    random.seed(seed)
    np.random.seed(seed)
    qiskit_algorithms.utils.algorithm_globals.random_seed = seed
# 1. SETUP PATHS (Assume subfolders 'train', 'validation', 'test' inside args.data)
    print(f"\n[Data Loading] Root Directory: {args.data}")
    train_dir = os.path.join(args.data, "train")
    val_dir = os.path.join(args.data, "validation")
    test_dir = os.path.join(args.data, "test")

    # 2. LOAD DATASETS (Load -> FeatureEng -> Window -> Stack)
    # Train: Fit Scalers = True
    print("\nProcessing TRAINING Set...")
    x_train, y_train, x_scaler, y_scaler = prepare_dataset_from_directory(
        train_dir, args, fit_scalers=True
    )
    
    # Val/Test: Fit Scalers = False (Use training scalers)
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

    total_samples = x_train.shape[0] + x_val.shape[0] + x_test.shape[0]
    args.data_n = total_samples
    args.data_dt = 4.5  # You can set this if you have a specific time resolution

    # Construct QNN
    if args.model == 'multihead':
        if not hasattr(args, 'heads_config') or not args.heads_config:
            raise ValueError("Model is 'multihead' but no 'heads_config' found in arguments/YAML.")

        models_list = []
        input_indices_list = []
        combined_params = {'input_params': [], 'weight_params': [], 'qc': None}
        
        print(f"\n[Builder] Constructing Dynamic Multi-Head QNN ({len(args.heads_config)} heads)...")
        
        # 2. Iterate through the configuration list
        for i, head_cfg in enumerate(args.heads_config):
            
            # --- A. Resolve Input Indices ---
            # We look for the feature names in the global 'args.select_features' list
           # --- A. Resolve Input Indices ---
            # We look for the feature names in the global 'args.select_features' list
            head_feature_names = map_names(head_cfg.get('features', []))
            current_head_indices = []
            
            for feat in head_feature_names:
                # Find index of this feature in the actual dataset columns we selected
                # We use string matching. 
                try:
                    # Note: args.features usually holds the full names or codes depending on your map_names logic
                    # We try to match exactly what is in args.features
                    idx = args.features.index(feat)
                    current_head_indices.append(idx)
                except ValueError:
                    raise ValueError(f"Head {i+1} Config Error: Feature '{feat}' not found in global select_features: {args.features}")
            
            input_indices_list.append(current_head_indices)
            
            # --- B. Create Head-Specific Args ---
            head_args = deepcopy(args)
            head_args.map = None
            # Override global args with head-specific ones (reps, encoding, etc.)
            for key, val in head_cfg.items():
                if key not in ['features', 'outputs']: # Skip non-arg keys
                    setattr(head_args, key, val)
            
            # Important: Set the 'features' attribute for the circuit builder to match this head's inputs
            # The circuit builder uses len(args.features) to determine qubit count
            head_args.features = head_feature_names 
            
            # --- C. Build Circuit ---
            qc, in_p, w_p = create_multivariate_circuit(head_args)
            backend = AerSimulator(seed_simulator=seed)
            estimator_options = {"run_options": {"shots": None, "seed": seed}, "backend_options": {"seed_simulator": seed}}
            estimator = Estimator(options=estimator_options)
            obsvs = [SparsePauliOp('I' * (qc.num_qubits - 1 - i) + 'Z' + 'I' * i) for i in range(qc.num_qubits)]
            estimator_qnn = EstimatorQNN(
                circuit=qc, input_params=in_p, weight_params=w_p, observables=obsvs, estimator=estimator, input_gradients=False,  pass_manager=generate_preset_pass_manager(backend=backend, optimization_level=1, seed_transpiler=seed))
            
            # --- D. Determine Output Size ---
            # The head output size = number of targets assigned to this head.
            # You can explicitly pass 'output_dim' in config, or we infer it.
            # Assuming 'outputs' key in config lists the target names for logging/verification
            num_outputs = head_cfg.get('output_dim', 1) 
            
            # Create Wrapper
            head_model = WindowEncodingQNN(estimator_qnn, (0, args.horizon, num_outputs), seed=args.run)
            models_list.append(head_model)
            
            # Store structure info (for saving)
            combined_params['input_params'].extend(list(in_p))
            combined_params['weight_params'].extend(list(w_p))
            print(f"  > Built Head {i+1}: Input {head_feature_names} -> {num_outputs} Outputs | Reps={head_args.reps}, Enc={head_args.encoding}")

        # --- E. Combine ---
        model = MultiHeadQNN(models_list, input_indices_list)
        qnn_dict = combined_params
    elif args.model == 'vanilla':
        qc, input_params, weight_params = create_multivariate_circuit(args)
        backend = AerSimulator(seed_simulator=seed)
        estimator_options = {"run_options": {"shots": None, "seed": seed}, "backend_options": {"seed_simulator": seed}}
        estimator = Estimator(options=estimator_options)
        obsvs = [SparsePauliOp('I' * (qc.num_qubits - 1 - i) + 'Z' + 'I' * i) for i in range(qc.num_qubits)]
        estimator_qnn = EstimatorQNN(circuit=qc,input_params=input_params, weight_params=weight_params, observables=obsvs, estimator=estimator, input_gradients=False,  pass_manager=generate_preset_pass_manager(backend=backend, optimization_level=1, seed_transpiler=seed))

        # Instantiate model
        model = WindowEncodingQNN(estimator_qnn, y_train.shape, seed)
        qnn_dict = {'input_params': input_params, 'weight_params': weight_params, 'qc': qc}
    # Training
    results = train_model(args, model, x_train, y_train, x_val, y_val, y_scaler)

    # Evaluation
    print("\nEvaluating on Test Set...")
    
    # --- BEST WEIGHTS ---
    print("...with best weights")
    best_eval = evaluate_model(args, model, results['best_weights'], x_test, y_test, x_scaler, y_scaler)

    # --- FINAL WEIGHTS ---
    print("...with final weights")
    final_eval = evaluate_model(args, model, results['final_weights'], x_test, y_test, x_scaler, y_scaler)

    scalers = [x_scaler, y_scaler]
    model_dir = save_experiment_results(args, results, best_eval, final_eval, scalers, qnn_dict, timestamp)
    load_experiment_results(model_dir)

    fig_dir = model_dir.replace("models", "figures")[:-4]
    os.makedirs(fig_dir, exist_ok=True)

    plot_convergence(args, results, filename=f"{fig_dir}/plot_convergence.png")
    # Local plots
    plot_kinematics_branches(args, final_eval, filename=f"{fig_dir}/plot_branches_local.png")
    plot_kinematics_boxplots(args, final_eval, mode='local', filename=f"{fig_dir}/plot_horizon_errors")

    # Global Open Plots
    plot_kinematics_time_series(args, final_eval, loop = 'open', filename=f"{fig_dir}/plot_kinematics_open.png")
    plot_kinematics_errors(args, final_eval, loop = 'open', filename=f"{fig_dir}/compare_error/plot_error_vs_time")

    #Global Closed Plots
    plot_kinematics_time_series(args, final_eval, loop = 'closed', filename=f"{fig_dir}/plot_kinematics_closed.png")
    plot_kinematics_errors(args, final_eval, loop = 'closed', filename=f"{fig_dir}/compare_error/plot_error_vs_time")
    
    
    

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    # Data structure
    parser.add_argument('--data', type=str, default="dataset") #TODO: Try full dataset
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-select', '--select_features', type=str, default=['wv','sv', 'yr','ya','rarad'], nargs='+', help="Explicitly select features (e.g. --select_features wv sv yr ya)")
    group.add_argument('-drop', '--drop_features', type=str, nargs='+', help="Drop features from default set (e.g. --drop_features rarad)")
    parser.add_argument('-ws', '--window_size', type=int, default=5, help="Window size = num qubits")
    parser.add_argument('-y', '--horizon', type=int, default=5) # 1,3,5
    
    # Target Options
    parser.add_argument('--predict', type=str, default='delta', choices=['delta', 'motion','motion_without_surge'], help="Target: 'delta' (simple steps) or 'motion' (kinematic variables)")
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
    parser.add_argument('--model', type=str, default='vanilla', choices=['vanilla', 'multihead']) # vanilla
    parser.add_argument('--heads_config', default=None, help="Config for multi-head model (loaded via YAML)")
    # Optimization
    parser.add_argument('-opt','--optimizer', type=str, default='cobyla', choices=['cobyla', 'spsa']) # cobyla
    parser.add_argument('--maxiter', type=int, default=10000)
    parser.add_argument('-tol', '--tolerance', type=float, default=None)
    parser.add_argument('-lr','--learning_rate', type = float, default = 0.01)
    parser.add_argument('-p','--perturbation',type = float, default = 0.1)
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for optimizers that support mini-batching (e.g., SPSA). Ignored for full-batch optimizers like COBYLA.")
    parser.add_argument('--weights', type = str, default="[1.0, 1.0, 1.0, 1.0]", help="Weights for the loss function.")
    
    parser.add_argument('--show_plot', type = str2bool, default=False)
    parser.add_argument('--save_plot', type = str2bool, default=True)
    parser.add_argument('--run', type=int, default=0)
    args = parser.parse_args()
    run(args)