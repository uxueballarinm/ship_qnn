import argparse
import os
import pickle
import glob
import datetime
from copy import deepcopy
from qnn_utils import *

class Args:
    """Helper to convert dict to object for compatibility with utils"""
    def __init__(self, **entries):
        self.__dict__.update(entries)

def run_evaluation(cli_args):
    # 1. LOAD MODEL & CONFIG
    print(f"{C_BLUE}Loading model from: {cli_args.model}...{C_RESET}")
    if not os.path.exists(cli_args.model):
        return print(f"{C_RED}Error: Model not found at {cli_args.model}{C_RESET}")

    with open(cli_args.model, 'rb') as f:
        saved_data = pickle.load(f)

    args_dict = saved_data['config']
    # Overwrite saved plotting flags with current CLI choices
    args_dict['show_plot'] = cli_args.show_plot
    args_dict['save_plot'] = cli_args.save_plot
    args = Args(**args_dict)
    
    # 2. DECIDE: RE-EVALUATE OR USE SAVED RESULTS
    # Trajectory plots (plot_kinematics_branches, etc) ALWAYS need a forward pass 
    # because raw arrays are not stored in the .pkl.
    needs_forward_pass = cli_args.reevaluate or 'plot' in cli_args.mode

    if needs_forward_pass:
        print(f"{C_YELLOW}Preparing data and QNN for inference...{C_RESET}")
        x_scaler, y_scaler = saved_data['x_scaler'], saved_data['y_scaler']
        data_path = args.data
        
        # Prepare Test Data
        test_subdir = os.path.join(data_path, 'test')
        search_path = test_subdir if os.path.exists(test_subdir) else data_path
        file_list = glob.glob(os.path.join(search_path, "*.csv"))
        
        if not file_list: raise ValueError(f"No CSV files found in {search_path}")

        all_x, all_y = [], []
        for fpath in file_list:
            df = process_single_df(pd.read_csv(fpath))
            feat_norm = np.clip(x_scaler.transform(df[args.features].values), 0, np.pi)
            pred_vals = df[args.targets].values
            pred_norm = y_scaler.transform(pred_vals) if y_scaler else pred_vals
            xw, yw = sliding_window(feat_norm, pred_norm, args.window_size, args.horizon)
            all_x.append(xw); all_y.append(yw)

        x_test, y_test = np.concatenate(all_x, axis=0), np.concatenate(all_y, axis=0)
        weights = saved_data.get('selected_weights', saved_data.get('best_weights', saved_data.get('final_weights')))

        # 3. RECONSTRUCT ARCHITECTURE
        if getattr(args, 'model', 'vanilla') == 'multihead':
            models, input_indices_list = [], []
            backend = AerSimulator(seed_simulator=args.run)
            for head_cfg in args.heads_config:
                # Merge global params into head config
                full_h_cfg = vars(deepcopy(args))
                full_h_cfg.update(head_cfg)
                head_args = Args(**full_h_cfg)
                head_args.features = map_names(head_cfg['features'])
                
                # Circuit and QNN Logic exactly as requested
                qc, in_p, w_p = create_multivariate_circuit(head_args)
                est = Estimator(options={"run_options": {"shots": None}, "backend_options": {"seed_simulator": args.run}})
                pm = generate_preset_pass_manager(backend=backend, optimization_level=1, seed_transpiler=args.run)
                obsvs = [SparsePauliOp('I'*(qc.num_qubits-1-j)+'Z'+'I'*j) for j in range(qc.num_qubits)]
                qnn = EstimatorQNN(circuit=qc, input_params=in_p, weight_params=w_p, observables=obsvs, 
                                   estimator=est, pass_manager=pm, default_precision=0.0)
                
                models.append(WindowEncodingQNN(qnn, (y_test.shape[0], args.horizon, head_cfg['output_dim']), args.run))
                input_indices_list.append([args.features.index(f) for f in head_args.features])
            
            model = MultiHeadQNN(models, input_indices_list)
        else:
            # Vanilla Logic
            qc, in_p, w_p = create_multivariate_circuit(args)
            est = Estimator(options={"run_options": {"shots": None}, "backend_options": {"seed_simulator": args.run}})
            pm = generate_preset_pass_manager(backend=AerSimulator(), optimization_level=1, seed_transpiler=args.run)#TODO: Passmanager inside the function? isnt it better to be outside?
            obsvs = [SparsePauliOp('I'*(qc.num_qubits-1-i)+'Z'+'I'*i) for i in range(qc.num_qubits)]
            qnn = EstimatorQNN(circuit=qc, input_params=in_p, weight_params=w_p, observables=obsvs, 
                               estimator=est, pass_manager=pm, default_precision=0.0)
            model = WindowEncodingQNN(qnn, (y_test.shape[0], args.horizon, y_test.shape[-1]), args.run)

        # Execute evaluation
        results = evaluate_model(args, model, weights, x_test, y_test, x_scaler, y_scaler)
        selection_type = "Inference-Evaluation"
    else:
        # Load-only mode for terminal/Excel logs
        print(f"{C_GREEN}Mode: Summary Only. Using pre-saved metrics...{C_RESET}")
        results = {
            'metrics': saved_data.get('test_metrics', saved_data.get('val_metrics')),
            'train_history': saved_data.get('train_history', []),
            'val_history': saved_data.get('val_history', [])
        }
        selection_type = "Loaded-Results"

    # 4. OUTPUT HANDLING
    if 'log' in cli_args.mode:
        load_experiment_results(cli_args.model) 
        if cli_args.save_excel:
            ts = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
            trainer_res = {'train_history': results.get('train_history', []), 
                           'val_history': results.get('val_history', []),
                           'final_weights': saved_data.get('final_weights')}
            save_experiment_results(args, trainer_res, results, results, (saved_data['x_scaler'], saved_data['y_scaler']), 
                                    saved_data['qnn_structure'], ts, selection_type=selection_type)

    if 'plot' in cli_args.mode:
        print(f"{C_GREEN}Generating Visualizations...{C_RESET}")
        base_name = os.path.basename(cli_args.model).replace('.pkl', '')
        fig_prefix = f"figures/{base_name}/test_"
        if cli_args.save_plot: os.makedirs(os.path.dirname(fig_prefix), exist_ok=True)

        plot_kinematics_branches(args, results, filename=f"{fig_prefix}branches.png")
        # plot_kinematics_boxplots(args, results, mode='local', filename=fig_prefix)
        plot_kinematics_time_series(args, results, loop='open', filename=f"{fig_prefix}trajectory_open.png")

        plot_kinematics_time_series(args, results, loop='closed', filename=f"{fig_prefix}trajectory_closed.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QNN Ship Dynamics: Tester & Visualizer")
    parser.add_argument('--model', type=str, required=True, help="Path to .pkl model")
    parser.add_argument('--mode', type=str, choices=['log', 'plot', 'log+plot'], default='log')
    parser.add_argument('--reevaluate', type=str2bool, default=False, help="Forces fresh inference on data")
    parser.add_argument('--save_excel', type=str2bool, default=False, help="Update Excel summary")
    parser.add_argument('--show_plot', type=str2bool, default=True)
    parser.add_argument('--save_plot', type=str2bool, default=False)
    
    run_evaluation(parser.parse_args())