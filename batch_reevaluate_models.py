import os
import pickle
import pandas as pd
import glob
import torch
import datetime
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.transpiler import generate_preset_pass_manager

# Import your unified functions
from utils import *

class Args:
    """Helper to convert dict to object for compatibility with utils"""
    def __init__(self, **entries):
        self.__dict__.update(entries)

def reevaluate_all():
    input_excel = "logs\\all_together\\All_the_experiments.xlsx"
    output_excel = "logs\\all_together\\all_experiments_correct.xlsx"
    models_root = "models"
    
    if not os.path.exists(input_excel):
        print(f"Error: {input_excel} not found.")
        return
        
    df = pd.read_excel(input_excel)
    print(f"Loaded {len(df)} experiments. Re-evaluating into {output_excel}...")

    all_pkl_files = glob.glob(os.path.join(models_root, "**", "*.pkl"), recursive=True)
    model_lookup = {os.path.basename(f): f for f in all_pkl_files}

    for index, row in df.iterrows():
        model_id = row['model_id']
        if model_id not in model_lookup:
            print(f"[{index+1}/{len(df)}] Skip: {model_id} not found in models/ folder.")
            continue

        model_path = model_lookup[model_id]
        with open(model_path, 'rb') as f:
            saved_data = pickle.load(f)
        
        # 1. SETUP CONFIG AND SCALERS
        args_dict = saved_data['config']
        args = Args(**args_dict)
        x_scaler, y_scaler = saved_data['x_scaler'], saved_data['y_scaler']
        
        # 2. LOAD DATA (Val and Test only)
        try:
            x_val, y_val, _, _ = prepare_dataset_from_directory(os.path.join(args.data, "validation"), args, x_scaler=x_scaler, y_scaler=y_scaler, fit_scalers=False)
            x_test, y_test, _, _ = prepare_dataset_from_directory(os.path.join(args.data, "test"), args, x_scaler=x_scaler, y_scaler=y_scaler, fit_scalers=False)
        except Exception as e:
            print(f"   > Error loading data for {model_id}: {e}")
            continue

        # 3. RECONSTRUCT MODEL ARCHITECTURE
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = None
        qnn_dict = None

        if getattr(args, 'model', 'vanilla') == 'classical':
            input_dim = x_val.shape[2]
            output_dim = y_val.shape[2]
            raw_model = ClassicalLSTM(input_size=input_dim, hidden_size=args.hidden_size, num_layers=args.layers, output_size=y_val.shape[1]*output_dim, seed=args.run)
            model = ClassicalWrapper(raw_model, device, output_shape=y_val.shape)
            qnn_dict = {"type": "Classical LSTM"}
        
        elif args.model == 'multihead':
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
                qnn = EstimatorQNN(circuit=qc, input_params=in_p, weight_params=w_p, observables=obsvs, estimator=est, pass_manager=pm)
                models.append(WindowEncodingQNN(qnn, (y_val.shape[0], args.horizon, head_cfg['output_dim']), args.run))
                input_indices_list.append([args.features.index(f) for f in map_names(head_cfg['features'])])
            model = MultiHeadQNN(models, input_indices_list)
            qnn_dict = saved_data['qnn_structure']
        else:
            qc = saved_data['qnn_structure']['qc']
            est_qnn = EstimatorQNN(circuit=qc, input_params=saved_data['qnn_structure']['input_params'], weight_params=saved_data['qnn_structure']['weight_params'], observables=[SparsePauliOp('I'*(qc.num_qubits-1-i)+'Z'+'I'*i) for i in range(qc.num_qubits)], estimator=Estimator(), pass_manager=generate_preset_pass_manager(backend=AerSimulator(), optimization_level=1))
            model = WindowEncodingQNN(est_qnn, y_val.shape, args.run)
            qnn_dict = saved_data['qnn_structure']

        # 4. SELECTION LOGIC (Best vs Final)
        print(f"[{index+1}/{len(df)}] Processing {model_id}...")
        val_eval_best = evaluate_model(args, model, saved_data['best_weights'], x_val, y_val, x_scaler, y_scaler)
        val_eval_final = evaluate_model(args, model, saved_data['final_weights'], x_val, y_val, x_scaler, y_scaler)
        
        m_key = 'Global_open_MSE'
        if val_eval_best['metrics'][m_key] < val_eval_final['metrics'][m_key]:
            sel_type, sel_w, sel_val = "Best (Lowest Val Loss)", saved_data['best_weights'], val_eval_best
        else:
            sel_type, sel_w, sel_val = "Final (Last Iteration)", saved_data['final_weights'], val_eval_final
        
        # 5. FINAL TEST PASS
        test_eval = evaluate_model(args, model, sel_w, x_test, y_test, x_scaler, y_scaler)
        
        # 6. SAVE UPDATED RESULTS
        # We use a unique reeval timestamp to avoid overwriting your original .pkl models
        reeval_ts = datetime.datetime.now().strftime("%m-%d_%H-%M-%S") + "_reeval"
        
        pseudo_results = {
            'best_weights': saved_data['best_weights'], 
            'final_weights': saved_data['final_weights'], 
            'selected_weights': sel_w, 
            'train_history': saved_data.get('train_history', []), 
            'val_history': saved_data.get('val_history', [])
        }

        if getattr(args, 'model', 'vanilla') == 'classical':
            save_classical_results(args, pseudo_results, sel_val, test_eval, [x_scaler, y_scaler], reeval_ts, selection_type=sel_type, excel_path=output_excel)
        else:
            save_experiment_results(args, pseudo_results, sel_val, test_eval, [x_scaler, y_scaler], qnn_dict, reeval_ts, selection_type=sel_type, excel_path=output_excel)

    print(f"\nDone! Re-evaluation completed. Corrected summary saved to: {output_excel}")

if __name__ == "__main__":
    reevaluate_all()