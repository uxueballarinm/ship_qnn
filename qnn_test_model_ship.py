# Works for models after 12-11_14-00-00
# For models after 12-18_12-30-00 use load_experiment_results()

import argparse
from utils import *
from copy import deepcopy

class Args:
    """Helper to convert dict to object for compatibility with utils"""
    def __init__(self, **entries):
        self.__dict__.update(entries)

def run_test(cli_args):

    model_path = cli_args.model
    data_path = cli_args.data
    use_final_weights = cli_args.final
    show_plot_flag = cli_args.show_plot
    save_plot_flag = cli_args.save_plot

    if not os.path.exists(model_path): return print(f"Error: Model not found at {model_path}")
    if not os.path.exists(data_path): return print(f"Error: Data not found at {data_path}")


    # 1. LOAD PICKLE
    print(f"Loading model from: {model_path}")
    with open(model_path, 'rb') as f:
        saved_data = pickle.load(f)

    # 2. UNPACK SAVED ARTIFACTS
    args_dict = saved_data['config']
    args_dict['show_plot'] = show_plot_flag
    args_dict['save_plot'] = save_plot_flag
    args = Args(**args_dict)
    
    x_scaler = saved_data['x_scaler'] # Pre-fitted scaler
    y_scaler = saved_data['y_scaler'] # Pre-fitted scaler
    weights = saved_data['final_weights'] if use_final_weights else saved_data['best_weights']
    
    print(f"Loaded Config: Model={args.model}, F={len(args.features)}, W={args.window_size}, H={args.horizon}")

    # 3. PREPARE TEST DATA
    # We still need to load the CSV to get the actual test values
    df = pd.read_csv(data_path, index_col=0)
    
    df['delta Surge Velocity'] = df['Surge Velocity'].diff().fillna(0)
    df['delta Sway Velocity'] = df['Sway Velocity'].diff().fillna(0)
    df['delta Yaw Rate'] = df['Yaw Rate'].diff().fillna(0)
    df['delta Yaw Angle'] = df['Yaw Angle'].diff().fillna(0)
    
    # Reconstruct column lists based on config string
    cols = args.features
    pred_cols = ["delta Surge Velocity", "delta Sway Velocity", "delta Yaw Rate", "delta Yaw Angle"] if args.predict == "delta" else ["Surge Velocity","Sway Velocity","Yaw Rate","Yaw Angle"]
    
    # Extract and Scale
    feature_seqs = df[cols].to_numpy()
    prediction_seqs = df[pred_cols].to_numpy()
    
    feat_norm = x_scaler.transform(feature_seqs) # Use saved scaler
    feat_norm = np.clip(feat_norm, 0, np.pi)
    
    if y_scaler: pred_norm = y_scaler.transform(prediction_seqs) # Use saved scaler
    else: pred_norm = prediction_seqs

    # Fold Logic
    x_folds, y_folds = make_sliding_window_ycustom_folds(feat_norm, pred_norm, args.window_size, args.horizon)
    x_test = x_folds[args.testing_fold]
    y_test = y_folds[args.testing_fold]

    # 4. INITIALIZE & RECONSTRUCT MODEL
    
    # --- BRANCH A: MULTI-HEAD RECONSTRUCTION ---
    if getattr(args, 'model', 'vanilla') == 'multihead':
        print("[Builder] Reconstructing Multi-Head Model...")
        if not hasattr(args, 'heads_config') or not args.heads_config:
            raise ValueError("Model is marked 'multihead' but 'heads_config' is missing in args.")

        models = []
        input_indices_list = []
        
        # We assume the user wants to run on the same seed as training for circuit consistency
        backend = AerSimulator(seed_simulator=args.run)
        
        for i, head_cfg in enumerate(args.heads_config):
            # 4a. Create Config for this Head
            head_args = deepcopy(args)
            for k, v in head_cfg.items():
                setattr(head_args, k, v)

            # 4b. Rebuild Circuit
            qc_head, in_p_head, w_p_head = create_multivariate_circuit(head_args)
            
            # 4c. Rebuild Estimator
            estimator = Estimator(options={"run_options": {"shots": None}, "backend_options": {"seed_simulator": args.run}})
            pass_manager = generate_preset_pass_manager(backend=backend, optimization_level=1, seed_transpiler=args.run)
            
            # Observables (Z on each qubit)
            obsvs = [SparsePauliOp('I' * (qc_head.num_qubits - 1 - j) + 'Z' + 'I' * j) for j in range(qc_head.num_qubits)]
            
            qnn_head = EstimatorQNN(
                circuit=qc_head, 
                input_params=in_p_head, 
                weight_params=w_p_head, 
                observables=obsvs, 
                estimator=estimator, 
                pass_manager=pass_manager,
                default_precision=0.0
            )
            
            # 4d. Wrap in WindowEncodingQNN
            # Calculate output shape: (Batch, Horizon, Head_Output_Dim)
            h_out_dim = head_cfg['output_dim']
            dummy_shape = (y_test.shape[0], args.horizon, h_out_dim)
            
            model_head = WindowEncodingQNN(qnn_head, dummy_shape, args.run)
            models.append(model_head)
            
            # 4e. Resolve Input Indices for this Head
            # Convert short codes in head_cfg['features'] to full names, then find index in args.features
            head_feat_names_full = map_names(head_cfg['features']) # e.g. ['Surge Velocity']
            
            # args.features should be the full list of names used in training (e.g. ['Surge Velocity', 'Sway...', ...])
            # We map the head features to indices in the global input vector
            indices = []
            for feat in head_feat_names_full:
                try:
                    idx = args.features.index(feat)
                    indices.append(idx)
                except ValueError:
                    raise ValueError(f"Head feature '{feat}' not found in global feature list {args.features}")
            input_indices_list.append(indices)
            print(f"  > Head {i+1} Rebuilt: {len(indices)} Inputs -> {h_out_dim} Outputs")

        # 4f. Create the Wrapper
        model = MultiHeadQNN(models, input_indices_list)

    # --- BRANCH B: VANILLA RECONSTRUCTION (Old Logic) ---
    else:
        print("[Builder] Reconstructing Vanilla Model...")
        # Recover saved structure
        qc = saved_data['qnn_structure']['qc']
        input_params = saved_data['qnn_structure']['input_params']
        weight_params = saved_data['qnn_structure']['weight_params']
        
        backend = AerSimulator(seed_simulator=args.run)
        estimator = Estimator(options={"run_options": {"shots": None}, "backend_options": {"seed_simulator": args.run}})
        obsvs = [SparsePauliOp('I' * (qc.num_qubits - 1 - i) + 'Z' + 'I' * i) for i in range(qc.num_qubits)]
        
        estimator_qnn = EstimatorQNN(
            circuit=qc, 
            input_params=input_params, 
            weight_params=weight_params, 
            observables=obsvs, 
            estimator=estimator, 
            pass_manager=generate_preset_pass_manager(backend=backend, optimization_level=1, seed_transpiler=args.run),
            default_precision=0.0
        )
            
        dummy_shape = (y_test.shape[0], args.horizon, y_test.shape[-1]) 
        model = WindowEncodingQNN(estimator_qnn, dummy_shape, args.run)


    # 6. EVALUATE & PLOT
    print("Running Evaluation...")
    results = evaluate_model(args, model, weights, x_test, y_test, x_scaler, y_scaler)
    base_name = os.path.basename(model_path).replace('.pkl', '')
    if cli_args.version == 'old':
        # Create Figures Directory
        if not os.path.exists("figures"):
            os.makedirs("figures")
        
        save_dir = f"figures/{base_name}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        print(f"Saving plots to {save_dir}...")
        #plot_convergence(args, saved_data, filename=f"{save_dir}/plot_convergence.png")
        plot_kinematics_branches(args, results, filename=f"{save_dir}/plot_branches_local.png")
        plot_kinematics_boxplots(args, results, mode='local', filename=f"{save_dir}/plot_horizon_errors")

        # Global Open Plots
        plot_kinematics_time_series(args, results, loop = 'open', filename=f"{save_dir}/plot_trajectory_open.png")
        plot_kinematics_errors(args, results, loop = 'open', filename=f"{save_dir}/compare_error/plot_error_vs_time")

        #Global Closed Plots
        plot_kinematics_time_series(args, results, loop = 'closed', filename=f"{save_dir}/plot_trajectory_closed.png")
        plot_kinematics_errors(args, results, loop = 'closed', filename=f"{save_dir}/compare_error/plot_error_vs_time")
    elif cli_args.version == 'save':
        print("\n[Info] Saving Test Results to Excel/Logs...")
        
        try:
            parts = base_name.split('_')
            timestamp = f"{parts[0]}_{parts[1]}"
            print(f"[Info] Using original timestamp: {timestamp}")
        except IndexError:
            # Fallback if filename format is unexpected
            timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
            print(f"[Warning] Could not extract timestamp. Using current time: {timestamp}")
        
        train_results = {
            "best_weights": saved_data.get('best_weights'),
            "final_weights": saved_data.get('final_weights'),
            "train_history": saved_data.get('train_history', []),
            "val_history": saved_data.get('val_history', [])
        }
        
        # We reuse the training 'best' metrics just to fill the slot, 
        # as we are only testing one set of weights here.
        best_eval = {"metrics": saved_data.get('final_eval_metrics', {})}
        
        # 'results' contains the metrics for THIS test run (on the new data/fold)
        final_eval = results 
        
        scalers = [x_scaler, y_scaler]
        qnn_dict = saved_data['qnn_structure']
        
        # Call the utility function
        save_experiment_results(args, train_results, best_eval, final_eval, scalers, qnn_dict, timestamp)

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, choices=['old', 'new','save'],default='old', help="Choose 'old' to run test and generate plots, 'new' to load results, 'save' to save test results")
    parser.add_argument('--model', type=str, required=True, help="Path to .pkl")
    parser.add_argument('--data', type=str, default="datasets\zigzag_11_11_ind_reduced_2_s.csv")
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--show_plot', type=str2bool, default=False, choices=[True, False], help="Show plots interactively")
    parser.add_argument('--save_plot', type=str2bool, default=True, choices=[True, False], help="Save plots to files")
    args = parser.parse_args()
    
    if args.version in ['old', 'save']:  run_test(args)
    else: load_experiment_results(args.model)