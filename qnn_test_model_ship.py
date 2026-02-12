from utils import *

class Args:
    """Helper to convert dict to object for compatibility with utils"""
    def __init__(self, **entries):
        self.__dict__.update(entries)

def run_test(cli_args):

    model_path = cli_args.model
    data_path = cli_args.data
    force_final = cli_args.final
    show_plot_flag = cli_args.show_plot
    save_plot_flag = cli_args.save_plot

    if not os.path.exists(model_path): return print(f"Error: Model not found at {model_path}")
    if not os.path.exists(data_path): return print(f"Error: Data not found at {data_path}")

    # 1. LOAD MODEL & CONFIG
    print(f"Loading model from: {model_path}")
    with open(model_path, 'rb') as f:
        saved_data = pickle.load(f)

    args_dict = saved_data['config']
    args_dict['show_plot'] = show_plot_flag
    args_dict['save_plot'] = save_plot_flag
    args = Args(**args_dict)
    
    x_scaler = saved_data['x_scaler'] 
    y_scaler = saved_data['y_scaler'] 
    if force_final:
        # User explicitly asked for final weights (e.g. for debugging)
        print("[Info] Forcing usage of FINAL weights (User Request).")
        weights = saved_data.get('final_weights')
    
    elif 'selected_weights' in saved_data:
        # PERFECT CASE: Use the weights that WON the validation comparison during training
        method = saved_data.get('weight_selection_method', 'Unknown')
        print(f"[Info] Using automatically SELECTED weights (Method: {method})")
        weights = saved_data['selected_weights']
        
    else:
        # FALLBACK (Old models): Default to 'best_weights' (Lowest Validation Loss)
        print("[Warning] 'selected_weights' not found (Old Model). Falling back to 'best_weights'.")
        weights = saved_data.get('best_weights', saved_data.get('final_weights'))
    
    print(f"Loaded Config: Model={args.model}, F={len(args.features)}, W={args.window_size}, H={args.horizon}")

    # 2. PREPARE TEST DATA
    if os.path.isfile(data_path):
        file_list = [data_path]
        print(f"Test Mode: Single File ({os.path.basename(data_path)})")
    elif os.path.isdir(data_path):
        test_subdir = os.path.join(data_path, 'test')
        search_path = os.path.join(test_subdir, "*.csv") if os.path.exists(test_subdir) else os.path.join(data_path, "*.csv")
        file_list = glob.glob(search_path)
        print(f"Test Mode: Directory ({len(file_list)} files)")
    else:
        raise ValueError("Data path is invalid.")

    if not file_list: raise ValueError("No CSV files found.")

    all_x, all_y = [], []
    for fpath in file_list:
        df = pd.read_csv(fpath, index_col=None)
        if 'timestamp' not in df.columns and 'Time (s)' in df.columns: df['timestamp'] = df['Time (s)']
        df = process_single_df(df)
        
        cols = args.features
        pred_cols = ["delta Surge Velocity", "delta Sway Velocity", "delta Yaw Rate", "delta Yaw Angle"] if args.predict == "delta" else ["Surge Velocity","Sway Velocity","Yaw Rate","Yaw Angle"]
        if any(c not in df.columns for c in cols + pred_cols): continue

        feature_seqs = df[cols].to_numpy()
        prediction_seqs = df[pred_cols].to_numpy()
        
        feat_norm = np.clip(x_scaler.transform(feature_seqs), 0, np.pi)
        pred_norm = y_scaler.transform(prediction_seqs) if y_scaler else prediction_seqs

        x_w, y_w = sliding_window(feat_norm, pred_norm, args.window_size, args.horizon)
        all_x.append(x_w); all_y.append(y_w)

    if not all_x: raise ValueError("No valid data generated.")
    x_test = np.concatenate(all_x, axis=0)
    y_test = np.concatenate(all_y, axis=0)
    print(f"Test Data Shape: X={x_test.shape}, Y={y_test.shape}")

    # 3. RECONSTRUCT MODEL
    if getattr(args, 'model', 'vanilla') == 'multihead':
        models, input_indices_list = [], []
        backend = AerSimulator(seed_simulator=args.run)
        for head_cfg in args.heads_config:
            head_args = deepcopy(args)
            for k, v in head_cfg.items(): setattr(head_args, k, v)
            head_args.features = map_names(head_cfg['features'])
            qc, in_p, w_p = create_multivariate_circuit(head_args)
            est = Estimator(options={"run_options": {"shots": None}, "backend_options": {"seed_simulator": args.run}})
            pm = generate_preset_pass_manager(backend=backend, optimization_level=1, seed_transpiler=args.run)
            obsvs = [SparsePauliOp('I'*(qc.num_qubits-1-j)+'Z'+'I'*j) for j in range(qc.num_qubits)]
            qnn = EstimatorQNN(circuit=qc, input_params=in_p, weight_params=w_p, observables=obsvs, estimator=est, pass_manager=pm, default_precision=0.0)
            models.append(WindowEncodingQNN(qnn, (y_test.shape[0], args.horizon, head_cfg['output_dim']), args.run))
            input_indices_list.append([args.features.index(f) for f in map_names(head_cfg['features'])])
        model = MultiHeadQNN(models, input_indices_list)
    else:
        qc = saved_data['qnn_structure']['qc']
        est_qnn = EstimatorQNN(circuit=qc, input_params=saved_data['qnn_structure']['input_params'], weight_params=saved_data['qnn_structure']['weight_params'], observables=[SparsePauliOp('I'*(qc.num_qubits-1-i)+'Z'+'I'*i) for i in range(qc.num_qubits)], estimator=Estimator(), pass_manager=generate_preset_pass_manager(backend=AerSimulator(), optimization_level=1))
        model = WindowEncodingQNN(est_qnn, (y_test.shape[0], args.horizon, y_test.shape[-1]), args.run)

    # 4. EVALUATE
    print("Running Evaluation... (This may take a moment)")
    results = evaluate_model(args, model, weights, x_test, y_test, x_scaler, y_scaler)
    m_final = results['metrics'] # Extract metrics for printing

    # 5. PRINT DETAILED SUMMARY (Just like 'new' version)
    def get_fmt(metrics, key):
        val = metrics.get(key)
        if val is None: return "N/A"
        return f"{val:.5f}" if isinstance(val, (int, float)) else str(val)

    print("\n" + "="*120)
    print(f"       TEST REPORT: {os.path.basename(data_path)}")
    print("="*120)
    
    # Aggregate Row
    print(f"Step MSE: {get_fmt(m_final, 'Step_MSE'):<12} | Step R2: {get_fmt(m_final, 'Step_R2'):<12} | Local MSE: {get_fmt(m_final, 'Local_MSE'):<12} | Local ADE: {get_fmt(m_final, 'Local_ADE')}")
    print("-" * 120)
    print(f"Global OPEN   -> MSE: {get_fmt(m_final, 'Global_open_MSE'):<10} | R2: {get_fmt(m_final, 'Global_open_R2'):<10} | ADE: {get_fmt(m_final, 'Global_open_ADE'):<10} | FDE: {get_fmt(m_final, 'Global_open_FDE')}")
    print(f"Global CLOSED -> MSE: {get_fmt(m_final, 'Global_closed_MSE'):<10} | R2: {get_fmt(m_final, 'Global_closed_R2'):<10} | ADE: {get_fmt(m_final, 'Global_closed_ADE'):<10} | FDE: {get_fmt(m_final, 'Global_closed_FDE')}")

    print("\n--- Detailed Breakdown per Target ---")
    headers = ["TARGET", "Step MSE", "Step R2", "Loc MSE", "Loc ADE", "Open MSE", "Open R2", "Open ADE", "Clos MSE", "Clos R2", "Clos ADE"]
    header_str = "{:<16} | {:<9} {:<9} | {:<9} {:<9} | {:<9} {:<9} {:<9} | {:<9} {:<9} {:<9}".format(*headers)
    print("-" * len(header_str))
    print(header_str)
    print("-" * len(header_str))

    target_names = ["Surge_Velocity", "Sway_Velocity", "Yaw_Rate", "Yaw_Angle"]
    for tgt in target_names:
        def t_get(suffix): return get_fmt(m_final, f"{tgt}_{suffix}")
        row_vals = [
            tgt.replace("_", " "), t_get("Step_MSE"), t_get("Step_R2"),
            t_get("Local_MSE"), t_get("Local_ADE"),
            t_get("Global_open_MSE"), t_get("Global_open_R2"), t_get("Global_open_ADE"),
            t_get("Global_closed_MSE"), t_get("Global_closed_R2"), t_get("Global_closed_ADE")
        ]
        print("{:<16} | {:<9} {:<9} | {:<9} {:<9} | {:<9} {:<9} {:<9} | {:<9} {:<9} {:<9}".format(*row_vals))
    print("="*120 + "\n")

    # 6. PLOTTING (SHOW ONLY, NO SAVE)
    base_name = os.path.basename(model_path).replace('.pkl', '')
    
    # Define Filename generator that returns None if save_plot is False
    def get_fname(name):
        if not save_plot_flag: return None
        save_dir = f"figures/{base_name}"
        os.makedirs(save_dir, exist_ok=True)
        return f"{save_dir}/{name}"

    if not save_plot_flag:
        print("[Info] Interactive Mode: Plots will be shown but NOT saved.")

    # Generate Plots (If show_plot=True, they will appear. If save_plot=False, filename is None)
    plot_kinematics_branches(args, results, filename=get_fname("plot_branches_local.png"))
    plot_kinematics_boxplots(args, results, mode='local', filename=get_fname("plot_horizon_errors"))
    plot_kinematics_errors(args, results, loop = 'open', filename=get_fname("compare_error_plot_error_vs_time"))
    plot_kinematics_errors(args, results, loop = 'closed', filename=get_fname("compare_error_plot_error_vs_time"))

    if len(file_list) == 1:
        plot_kinematics_time_series(args, results, loop = 'open', filename=get_fname("plot_trajectory_open.png"))
        plot_kinematics_time_series(args, results, loop = 'closed', filename=get_fname("plot_trajectory_closed.png"))
    else:
        print("[Info] Trajectory plots skipped for folder input (Metrics only).")

    print("Done.")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, choices=['old', 'new','save'],default='old', help="Choose 'old' to run test and generate plots, 'new' to load results, 'save' to save test results")
    parser.add_argument('--model', type=str, required=True, help="Path to .pkl")
    parser.add_argument('--data', type=str, default="data/reduce_row_number_2", help="Path to test .csv data")
    parser.add_argument('--final',type=str2bool, default=False, choices=[True, False])
    parser.add_argument('--show_plot', type=str2bool, default=False, choices=[True, False], help="Show plots interactively")
    parser.add_argument('--save_plot', type=str2bool, default=True, choices=[True, False], help="Save plots to files")
    args = parser.parse_args()
    
    if args.version in ['old', 'save']:  run_test(args)
    else: load_experiment_results(args.model, final=args.final)