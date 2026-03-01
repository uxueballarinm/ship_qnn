import argparse
import pickle
import torch
import os
from qnn_utils import *
class Args:
    """Helper to convert dict to object for compatibility with utils"""
    def __init__(self, **entries):
        self.__dict__.update(entries)

def run_test_classical(cli_args):
    # 1. LOAD MODEL
    print(f"{C_BLUE}Loading classical model: {cli_args.model}{C_RESET}")
    if not os.path.exists(cli_args.model):
        return print(f"{C_RED}Error: Model not found at {cli_args.model}{C_RESET}")

    with open(cli_args.model, 'rb') as f:
        saved_data = pickle.load(f)

    args_dict = saved_data['config']
    args_dict['show_plot'], args_dict['save_plot'] = cli_args.show_plot, cli_args.save_plot
    args = Args(**args_dict)
    
    x_scaler, y_scaler = saved_data['x_scaler'], saved_data['y_scaler']
    
    # 2. DECIDE: INFERENCE OR LOAD
    needs_inference = cli_args.reevaluate or 'plot' in cli_args.mode

    if needs_inference:
        print(f"{C_YELLOW}Performing inference to generate trajectory arrays...{C_RESET}")
        
        # Load Data
        test_dir = os.path.join(args.data, "test")
        search_path = test_dir if os.path.exists(test_dir) else args.data
        file_list = glob.glob(os.path.join(search_path, "*.csv"))
        
        if not file_list: raise ValueError(f"No CSVs found in {search_path}")

        all_x, all_y = [], []
        for fpath in file_list:
            df = process_single_df(pd.read_csv(fpath))
            feat_norm = np.clip(x_scaler.transform(df[args.features].values), 0, np.pi)
            pred_norm = y_scaler.transform(df[args.targets].values) if y_scaler else df[args.targets].values
            xw, yw = sliding_window(feat_norm, pred_norm, args.window_size, args.horizon)
            all_x.append(xw); all_y.append(yw)

        x_test, y_test = np.concatenate(all_x, axis=0), np.concatenate(all_y, axis=0)

        # Reconstruct Architecture
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = x_test.shape[2]
        num_targets = y_test.shape[2]
        output_dim_flat = args.horizon * num_targets
        
        # Load Weights
        weights = saved_data.get('selected_weights', saved_data.get('best_weights', saved_data.get('final_weights')))
        
        model = ClassicalLSTM(input_dim, args.hidden_size, args.layers, output_dim_flat)
        model.to(device)
        
        # Wrap for evaluate_model compatibility
        wrapper = ClassicalWrapper(model, device, output_shape=y_test.shape)

        # FIX: Pass 'weights' instead of 'None'
        results = evaluate_model(args, wrapper, weights, x_test, y_test, x_scaler, y_scaler)
    else:
        print(f"{C_GREEN}Loading stored metrics...{C_RESET}")
        results = {'metrics': saved_data.get('test_metrics', saved_data['val_metrics'])}

    # 3. LOGGING & PLOTTING
    if 'log' in cli_args.mode:
        load_experiment_results(cli_args.model)

    if 'plot' in cli_args.mode:
        print(f"{C_GREEN}Generating Visualizations...{C_RESET}")
        # Standardizing folder path logic to match quantum figure organization
        folder_path = getattr(args, 'save_dir', getattr(args, 'save_folder', 'classical_baselines'))
        base_name = os.path.basename(cli_args.model).replace('.pkl','')
        fig_prefix = f"figures/{folder_path}/{base_name}_test_"
        
        if args.save_plot: os.makedirs(os.path.dirname(fig_prefix), exist_ok=True)
        
        plot_kinematics_branches(args, results, filename=f"{fig_prefix}branches.png")
        plot_kinematics_time_series(args, results, loop='closed', filename=f"{fig_prefix}trajectory_closed.png")
        plot_kinematics_boxplots(args, results, mode='local', filename=fig_prefix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classical LSTM Ship Dynamics: Tester")
    parser.add_argument('--model', type=str, required=True, help="Path to .pkl")
    parser.add_argument('--mode', type=str, choices=['log', 'plot', 'log+plot'], default='log+plot')
    parser.add_argument('--reevaluate', type=str2bool, default=False)
    parser.add_argument('--show_plot', type=str2bool, default=True)
    parser.add_argument('--save_plot', type=str2bool, default=False)
    
    run_test_classical(parser.parse_args())