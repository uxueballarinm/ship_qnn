from qnn_utils import *
import yaml
import re
from copy import deepcopy

class FrozenQNNWrapper:
    """
    Wraps a QNN (Vanilla or MultiHead) to freeze quantum weights.
    SPSA will only 'see' and optimize the classical readout parameters.
    """
    def __init__(self, original_model, initial_params):
        self.model = original_model
        # We keep a copy of the FULL initial parameter vector
        self.full_params = np.copy(initial_params)
        
        # Identify which indices are Quantum (Frozen) and Classical (Active)
        self.q_indices = []
        self.c_indices = []
        
        if hasattr(original_model, 'models'): # It's a MultiHead Model
            offset = 0
            for m in original_model.models:
                # Quantum weights are at the start of each head's chunk
                self.q_indices.extend(range(offset, offset + m.num_q_params))
                # Classical weights are at the end of each head's chunk
                self.c_indices.extend(range(offset + m.num_q_params, offset + m.total_params))
                offset += m.total_params
        else: # It's a Vanilla Model
            self.q_indices = list(range(original_model.num_q_params))
            self.c_indices = list(range(original_model.num_q_params, original_model.total_params))
            
        self.total_params = len(self.c_indices)
        print(f"{C_BLUE}[FrozenQNN] Quantum weights FROZEN ({len(self.q_indices)} params).{C_RESET}")
        print(f"{C_BLUE}[FrozenQNN] Optimizer will tune {self.total_params} classical weights.{C_RESET}")

    def forward(self, x, classical_params):
        # Inject the active parameters into the correct slots of the full vector
        self.full_params[self.c_indices] = classical_params
        return self.model.forward(x, self.full_params)

    def initialize_parameters(self, strategy):
        # We only return the classical part of a fresh initialization for the optimizer
        full_init = self.model.initialize_parameters(strategy)
        return full_init[self.c_indices]

def run(args):
    timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    
    # 1. SEARCH FOR EXISTING EXPERIMENT (.pkl)
    found_path, saved_data = find_existing_experiment(args)
    
    if found_path is not None and args.check_existing:
        print(f"\n{C_GREEN}>> MATCH FOUND: {found_path}{C_RESET}")
        if not getattr(args, 'save_in_excel', True):
            print(f"{C_YELLOW}>> Flag '--save_in_excel' is False. Skipping.{C_RESET}")
            return 
            
        print(f"{C_BLUE}>> Syncing results to Excel and models folder...{C_RESET}")
        saved_config = saved_data.get('config', {})
        for attr in ['features', 'targets', 'data_n', 'data_dt']:
            if not hasattr(args, attr):
                setattr(args, attr, saved_config.get(attr))
        
        train_res = {
            'selected_weights': saved_data.get('selected_weights', saved_data.get('best_weights')),
            'final_weights': saved_data.get('final_weights'),
            'train_history': saved_data.get('train_history', []),
            'val_history': saved_data.get('val_history', [])
        }
        
        save_experiment_results(args, train_res, 
            {'metrics': saved_data.get('val_metrics', {})}, 
            {'metrics': saved_data.get('test_metrics', {})}, 
            [saved_data['x_scaler'], saved_data['y_scaler']], 
            saved_data['qnn_structure'], timestamp, 
            selection_type=saved_data.get('weight_selection_method', 'Unknown'))
        return 

    # 2. START TRAINING
    print(f"\n{C_BLUE}--- Starting New Training ---{C_RESET}")

    seed = args.run
    random.seed(seed); np.random.seed(seed)
    qiskit_algorithms.utils.algorithm_globals.random_seed = seed

    train_dir = os.path.join(args.data, "train")
    val_dir = os.path.join(args.data, "validation")
    test_dir = os.path.join(args.data, "test")

    x_train, y_train, x_scaler, y_scaler = prepare_dataset_from_directory(train_dir, args, fit_scalers=True)
    x_val, y_val, _, _ = prepare_dataset_from_directory(val_dir, args, x_scaler=x_scaler, y_scaler=y_scaler, fit_scalers=False)
    x_test, y_test, _, _ = prepare_dataset_from_directory(test_dir, args, x_scaler=x_scaler, y_scaler=y_scaler, fit_scalers=False)

    args.data_n = x_train.shape[0] + x_val.shape[0] + x_test.shape[0]
    args.data_dt = 4.5 

    # Construct QNN
    if args.model == 'multihead':
        models_list = []; input_indices_list = []
        combined_params = {'input_params': [], 'weight_params': [], 'qc': None}
        print(f"\n[Builder] Constructing Dynamic Multi-Head QNN ({len(args.heads_config)} heads)...")
        
        for i, head_cfg in enumerate(args.heads_config):
            head_feature_names = map_names(head_cfg.get('features', []))
            current_head_indices = [args.features.index(feat) for feat in head_feature_names]
            input_indices_list.append(current_head_indices)
            
            head_args = deepcopy(args)
            head_args.map = head_cfg.get('map', None) # Pass map if exists, else None
            for key, val in head_cfg.items():
                if key not in ['features', 'outputs']: setattr(head_args, key, val)
            head_args.features = head_feature_names 
            
            qc, in_p, w_p = create_multivariate_circuit(head_args)
            backend = AerSimulator(seed_simulator=seed)
            estimator = Estimator(options={"run_options": {"shots": None, "seed": seed}})
            obsvs = [SparsePauliOp('I'*(qc.num_qubits-1-j)+'Z'+'I'*j) for j in range(qc.num_qubits)]
            estimator_qnn = EstimatorQNN(circuit=qc, input_params=in_p, weight_params=w_p, observables=obsvs, estimator=estimator,  pass_manager=generate_preset_pass_manager(backend=backend, optimization_level=1, seed_transpiler=seed))
            
            num_outputs = head_cfg.get('output_dim', 1) 
            head_model = WindowEncodingQNN(estimator_qnn, (0, args.horizon, num_outputs), seed=args.run)
            models_list.append(head_model)
            combined_params['weight_params'].extend(list(w_p))
            print(f"  > Head {i+1}: {head_feature_names} -> {num_outputs} Out | Reps={head_args.reps}")

        model = MultiHeadQNN(models_list, input_indices_list)
        qnn_dict = combined_params

    elif args.model == 'vanilla':
        qc, in_p, w_p = create_multivariate_circuit(args)
        estimator = Estimator(options={"run_options": {"shots": None, "seed": seed}})
        obsvs = [SparsePauliOp('I'*(qc.num_qubits-1-j)+'Z'+'I'*j) for j in range(qc.num_qubits)]
        estimator_qnn = EstimatorQNN(circuit=qc, input_params=in_p, weight_params=w_p, observables=obsvs, estimator=estimator,  pass_manager=generate_preset_pass_manager(backend=backend, optimization_level=1, seed_transpiler=seed))
        model = WindowEncodingQNN(estimator_qnn, y_train.shape, seed)
        qnn_dict = {'input_params': in_p, 'weight_params': w_p, 'qc': qc}

    # --- WRAPPER LOGIC ---
    if getattr(args, 'freeze_qnn', False):
        print(f"{C_YELLOW}>> MODE: Frozen QNN (Reservoir Computing){C_RESET}")
        initial_full_params = model.initialize_parameters(args.initialization)
        model_to_train = FrozenQNNWrapper(model, initial_full_params)
    else:
        model_to_train = model

    results = train_model(args, model_to_train, x_train, y_train, x_val, y_val, y_scaler)

    # Evaluation (Important: use model_to_train for forward passes if it converged early)
    print("\n[Model Selection] Comparing 'Best' vs 'Final' weights on VALIDATION set...")
    val_eval_best = evaluate_model(args, model_to_train, results['best_weights'], x_val, y_val, x_scaler, y_scaler)
    val_eval_final = evaluate_model(args, model_to_train, results['final_weights'], x_val, y_val, x_scaler, y_scaler)
    
    score_best = val_eval_best['metrics']['Global_open_MSE']
    score_final = val_eval_final['metrics']['Global_open_MSE']

    if score_best < score_final:
        selected_weights_type = "Best (Lowest Val Loss)"
        selected_weights = results['best_weights']
        selected_val_metrics = val_eval_best
    else:
        selected_weights_type = "Final (Last Iteration)"
        selected_weights = results['final_weights']
        selected_val_metrics = val_eval_final
    
    print(f"  > Selected: {selected_weights_type}")
    test_eval = evaluate_model(args, model_to_train, selected_weights, x_test, y_test, x_scaler, y_scaler)
    
    # Reconstruct full vector if frozen for saving
    if getattr(args, 'freeze_qnn', False):
        final_full_weights = np.copy(model_to_train.full_params)
        final_full_weights[model_to_train.c_indices] = selected_weights
        results['selected_weights'] = final_full_weights
    else:
        results['selected_weights'] = selected_weights

    saved_file = save_experiment_results(args, results, selected_val_metrics, test_eval, [x_scaler, y_scaler], qnn_dict, timestamp, selection_type=selected_weights_type)
    load_experiment_results(saved_file)

    if args.save_plot or args.show_plot:
        fig_dir = saved_file.replace("models", "figures")[:-4]
        os.makedirs(fig_dir, exist_ok=True)
        plot_convergence(args, results, filename=f"{fig_dir}/plot_convergence.png")
        plot_kinematics_branches(args, test_eval, filename=f"{fig_dir}/plot_branches_local.png")
        plot_kinematics_time_series(args, test_eval, loop='open', filename=f"{fig_dir}/plot_kinematics_open.png")
        plot_kinematics_time_series(args, test_eval, loop='closed', filename=f"{fig_dir}/plot_kinematics_closed.png")

if __name__=="__main__":
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--indices', type=int, nargs='+', default=None)
    parser.add_argument('--save_dir', type=str, default="")
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--data', type=str, default="dataset")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-select', '--select_features', type=str, nargs='+', default=['sv','wv', 'yr','ya','rarad'])
    group.add_argument('-drop', '--drop_features', type=str, nargs='+')
    parser.add_argument('-ws', '--window_size', type=int, default=5)
    parser.add_argument('-y', '--horizon', type=int, default=5)
    parser.add_argument('--predict', type=str, default='motion', choices=['delta', 'motion','motion_without_surge'])
    parser.add_argument('--norm', type=str2bool, default=True)
    parser.add_argument('-rt', '--reconstruct_train', type=str2bool, default=False)
    parser.add_argument('-rv', '--reconstruct_val', type=str2bool, default=False)
    parser.add_argument('--map', type=str, nargs='+')
    parser.add_argument('--reorder', type=str2bool, default=False)
    parser.add_argument('--encoding', type=str, default='compact')
    parser.add_argument('--entangle', type=str, default='reverse_linear')
    parser.add_argument('--ansatz', type=str, default='ugates')
    parser.add_argument('--reps', type=int, default=3)
    parser.add_argument('-init', '--initialization', type=str, default='uniform')
    parser.add_argument('--model', type=str, default='vanilla')
    parser.add_argument('--heads_config', default=None)
    parser.add_argument('-opt','--optimizer', type=str, default='cobyla')
    parser.add_argument('--maxiter', type=int, default=10000)
    parser.add_argument('-tol', '--tolerance', type=float, default=None)
    parser.add_argument('-lr','--learning_rate', type=float, default=0.01)
    parser.add_argument('-p','--perturbation', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--weights', type=str, default="[1.0, 1.0, 1.0, 1.0]")
    parser.add_argument('--show_plot', type=str2bool, default=False)
    parser.add_argument('--save_plot', type=str2bool, default=True)
    parser.add_argument('--check_existing', type=str2bool, default=False)
    parser.add_argument('--save_in_excel', type=str2bool, default=False)
    parser.add_argument('--use_hadamard', type=str2bool, default=False)
    # NEW FLAGS
    parser.add_argument('--freeze_qnn', type=str2bool, default=False)
    parser.add_argument('--convergence_stop', type=str2bool, default=False)
    parser.add_argument('--convergence_window', type=int, default=200)
    parser.add_argument('--validate_all', type=str2bool, default=False)

    args = parser.parse_args()
    if args.config:
        with open(args.config, 'r') as f:
            config_content = yaml.safe_load(f)
        config_list = [config_content] if isinstance(config_content, dict) else config_content
        for i, config_dict in enumerate(config_list):
            if args.indices and (i+1) not in args.indices: continue
            current_args = argparse.Namespace(**vars(args))
            for k, v in config_dict.items(): setattr(current_args, k, v)
            run(current_args)
    else:
        run(args)