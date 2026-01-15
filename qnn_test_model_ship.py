# Works for models after 12-11_14-00-00
# For models after 12-18_12-30-00 use load_experiment_results()

import argparse
from utils import *

class Args:
    """Helper to convert dict to object for compatibility with utils"""
    def __init__(self, **entries):
        self.__dict__.update(entries)

def run_test(cli_args): #TODO: Ordenar y elegir plots

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
    # 3. RECONSTRUCT CIRCUIT FROM SAVED STRUCTURE
    # No need to call create_multivariate_circuit()!
    qc = saved_data['qnn_structure']['qc']
    input_params = saved_data['qnn_structure']['input_params']
    weight_params = saved_data['qnn_structure']['weight_params']
    
    print(f"Loaded Config: F={len(args.features)}, W={args.window_size}, H={args.horizon}")

    # 4. PREPARE TEST DATA
    # We still need to load the CSV to get the actual test values
    df = pd.read_csv(data_path, index_col=0)
    df['delta x'] = df['Position X'].diff().fillna(0)
    df['delta y'] = df['Position Y'].diff().fillna(0)

    # Reconstruct column lists based on config string
    cols = args.features
    pred_cols = ["delta x", "delta y"] if args.predict == "delta" else ["Position X", "Position Y"]
    
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

    # 5. INITIALIZE QNN
    # Minimal Setup - no circuit logic required
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
    
    dummy_shape = (10, args.horizon, 2)
    model = WindowEncodingQNN(estimator_qnn, dummy_shape, args.run)

    # 6. EVALUATE & PLOT
    print("Running Evaluation...")
    results = evaluate_model(args, model, weights, x_test, y_test, x_scaler, y_scaler)
    base_name = os.path.basename(model_path).replace('.pkl', '')
    
    # Create Figures Directory
    if not os.path.exists("figures"):
        os.makedirs("figures")
    
    save_dir = f"figures/{base_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    print(f"Saving plots to {save_dir}...")
    # plot_convergence(args, saved_data, filename=f"{save_dir}/plot_convergence.png")
    plot_horizon_branches(args, results, filename=f"{save_dir}/plot_branches.png")
    plot_horizon_euclidean_boxplots(args, results, mode='local', filename=f"{save_dir}/plot_horizon_errors")
    plot_trajectory_components(args, results, filename=f"{save_dir}/plot_trajectory.png")
    plot_errors_and_position_time(args, results, filename=f"{save_dir}/plot_error_vs_time")

    print("Done.")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=str, choices=['old', 'new'])
    parser.add_argument('--model', type=str, required=True, help="Path to .pkl")
    parser.add_argument('--data', type=str, default="datasets\zigzag_10_40_ood_reduced_2_s.csv")
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--show_plot', type=str2bool, default=False, choices=[True, False], help="Show plots interactively")
    parser.add_argument('--save_plot', type=str2bool, default=True, choices=[True, False], help="Save plots to files")
    args = parser.parse_args()
    
    if args.version == 'old': run_test(args)
    else: load_experiment_results(args.model)