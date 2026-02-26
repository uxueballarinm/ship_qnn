# Import from your existing utils
from qnn_utils import *
from qnn_utils import _compute_loss

def build_multihead_model(args):
    """
    Reconstructs the MultiHeadQNN exactly as done in the training script.
    """
    print(f"[Builder] Reconstructing Multi-Head QNN...")
    seed = args.run
    random.seed(seed)
    np.random.seed(seed)
    heads = []
    head_input_indices = []
    
    # Ensure full feature list is resolved
    full_features = map_names(args.select_features)
    
    for i, h_conf in enumerate(args.heads_config):
        # 1. Prepare Args for this specific head
        # We copy the global args and overwrite with head-specific config
        h_args = argparse.Namespace(**vars(args))
        for k, v in h_conf.items():
            setattr(h_args, k, v)
            
        # Resolve head features to find indices in the main X dataset
        head_feat_names = map_names(h_args.features)
        indices = [full_features.index(f) for f in head_feat_names]
        head_input_indices.append(indices)
        
        # 2. Build Circuit
        # Note: map_names is handled inside create_multivariate_circuit via utils
        qc, in_params, w_params = create_multivariate_circuit(h_args)
        
        # 3. Build EstimatorQNN
        # Using EstimatorV2 as in your main script
        backend = AerSimulator(seed_simulator=seed)

        estimator_options = {"run_options": {"shots": None, "seed": seed}, "backend_options": {"seed_simulator": seed}}
        estimator = Estimator(options=estimator_options)
        obsvs = [SparsePauliOp('I' * (qc.num_qubits - 1 - i) + 'Z' + 'I' * i) for i in range(qc.num_qubits)]

        qnn = EstimatorQNN(
                circuit=qc, input_params=in_params, weight_params=w_params, observables=obsvs, estimator=estimator, input_gradients=False,  pass_manager=generate_preset_pass_manager(backend=backend, optimization_level=1, seed_transpiler=seed))
            
        # 4. Wrap in WindowEncodingQNN
        # Output shape logic: (Batch, Horizon, Cols)
        # We need to construct a dummy output shape to init the class
        dummy_shape = (1, args.horizon, h_args.output_dim)
        head_model = WindowEncodingQNN(qnn, dummy_shape, seed=args.run)

        heads.append(head_model)
        
        print(f"  > Head {i+1} Rebuilt: {len(indices)} inputs -> {h_args.output_dim} outputs | Params: {head_model.total_params}")

    # 5. Combine into MultiHead
    model = MultiHeadQNN(heads, head_input_indices)
    return model

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a QNN model using COBYLA.")
    parser.add_argument('--model', type=str, required=True, help='Path to the .pkl model file (SPSA result)')
    parser.add_argument('--maxiter', type=int, default=200, help='Iterations for COBYLA fine-tuning')
    parser.add_argument('--tol', type=float, default=0.0001, help='Tolerance for convergence')
    parser.add_argument('--run', type=list, default=[0], help='Random seed for reproducibility')
    args_script = parser.parse_args()

    # 1. Load the SPSA Model
    if not os.path.exists(args_script.model):
        raise FileNotFoundError(f"Model file not found: {args_script.model}")
        
    print(f"\n[Loader] Loading model from: {args_script.model}")
    with open(args_script.model, 'rb') as f:
        saved_data = pickle.load(f)
        
    # Recover configuration
    config = saved_data['config']
    args = argparse.Namespace(**config)
    
    # 2. Setup Data
    # Use the same data directory as training
    print(f"[Loader] Loading data from: {args.data}")
    X, Y, x_scaler, y_scaler = prepare_dataset_from_directory(
        os.path.join(args.data, 'train'), args, fit_scalers=True
    )
    # Load Validation for monitoring
    X_val, Y_val, _, _ = prepare_dataset_from_directory(
        os.path.join(args.data, 'validation'), args, x_scaler=x_scaler, y_scaler=y_scaler
    )
    
    print(f"  > Train Shape: {X.shape}")
    print(f"  > Val Shape:   {X_val.shape}")

    # 3. Rebuild Model Architecture
    model = build_multihead_model(args)
    
    # 4. Set Initial Weights (From SPSA Result)
    initial_weights = saved_data['final_weights']
    print(f"[Init] Starting COBYLA with weights from previous run (Loss: {saved_data['final_eval_metrics'].get('Step_MSE', 'N/A')})")

    # 5. Define Optimization Loop
    # We define a custom loop here to ensure COBYLA uses the specific weights
    best_val_loss = float('inf')
    best_params = np.copy(initial_weights)
    train_history = []
    val_history = []
    
    # Ensure weights exist in args (backward compatibility)
    if not hasattr(args, 'weights'): args.weights = [1.0, 1.0, 1.0, 1.0]

    def objective_function(params):
        nonlocal best_val_loss, best_params
        
        # Forward pass (Full Batch - COBYLA requires stable function)
        preds = model.forward(X, params)
        loss = _compute_loss(args, preds, Y, args.reconstruct_train, args.weights, scaler=y_scaler)
        
        # Validation Check
        val_preds = model.forward(X_val, params)
        val_loss = _compute_loss(args, val_preds, Y_val, args.reconstruct_val, args.weights, scaler=y_scaler)
        
        train_history.append(loss)
        val_history.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = np.copy(params)
            
        if len(train_history) % 10 == 0:
            print(f"  > COBYLA Step {len(train_history):3d} | Train: {loss:.5f} | Val: {val_loss:.5f}")
            
        return loss

    # 6. Run Optimizer
    print(f"\n[Fine-Tuning] Starting COBYLA (Maxiter={args_script.maxiter})...")
    start_time = time.time()
    
    optimizer = COBYLA(maxiter=args_script.maxiter, tol=args_script.tol)
    result = optimizer.minimize(fun=objective_function, x0=initial_weights)
    
    print(f"Fine-tuning completed in {(time.time() - start_time)/60:.2f} min.")
    print(f"Final Validation Loss: {best_val_loss:.5f}")

    # 7. Evaluate Final Performance
    print("\n[Evaluation] Running Final Evaluation...")
    final_metrics = evaluate_model(args, model, best_params, X_val, Y_val, x_scaler, y_scaler)
    
    # 8. Save Result
    timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M-%S")
    new_filename = args_script.model.replace(".pkl", f"_finetuned_{timestamp}.pkl")
    
    save_payload = {
        "config": vars(args),
        "best_weights": best_params,
        "final_weights": result.x,
        "train_history": saved_data['train_history'] + train_history, # Append history
        "val_history": saved_data['val_history'] + val_history,
        "best_eval_metrics": final_metrics['metrics'], # Update metrics
        "final_eval_metrics": final_metrics['metrics'],
        "y_scaler": y_scaler, 
        "x_scaler": x_scaler, 
        "qnn_structure": saved_data['qnn_structure']
    }
    
    with open(new_filename, "wb") as f:
        pickle.dump(save_payload, f)
        
    print(f"[Saved] Fine-tuned model saved to: {new_filename}")
    
    # Use utils to print summary table
    load_experiment_results(new_filename)

if __name__ == "__main__":
    main()