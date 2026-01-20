import os
import pickle
import argparse

import time
import datetime

import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from matplotlib.lines import Line2D


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap, EfficientSU2, ExcitationPreserving, PauliTwoDesign, RealAmplitudes
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_algorithms.optimizers import SPSA, COBYLA

colors = ['#E60000', '#FF8C00', '#C71585', '#008080', '#1E90FF']
#"Position X", "Position Y",
full_feature_set = [ "Surge Velocity", "Sway Velocity", "Yaw Rate", "Yaw Angle", "Speed U", "Rudder Angle (deg)", "Rudder Angle (rad)", "OOD Label"]


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def map_names(feature_list, reverse=False):
    """
    Maps between short codes (e.g., 'px') and full feature names (e.g., 'Position X').
    Args:
        feature_list (list): List of strings (codes or full names).
        to_short (bool): If True, converts Full Names -> Codes (for logging).
                         If False, converts Codes -> Full Names (for data loading).
    """
    # Central Dictionary
    code_to_name = {
        #"px": "Position X", "py": "Position Y",
        "wv":"Surge Velocity", "sv":"Sway Velocity", 
        "yr":"Yaw Rate", "ya":"Yaw Angle",
        "vu": "Speed U", "radeg": "Rudder Angle (deg)",
        "rarad": "Rudder Angle (rad)", "OOD": "OOD Label",
        "dwv":"delta Surge Velocity", "dsv":"delta Sway Velocity",
        "dyr":"delta Yaw Rate", "dya":"delta Yaw Angle",
        #"dx": "delta x", "dy": "delta y"
    }

    if reverse: # (Name -> Code)
        name_to_code = {v: k for k, v in code_to_name.items()}
        return [name_to_code.get(f, f) for f in feature_list]
    
    else: # (Code -> Name)
        columns = []
        for code in feature_list:
            if code in code_to_name:
                columns.append(code_to_name[code])
            elif code in code_to_name.values():
                columns.append(code) # Already a full name
            else:
                raise ValueError(f"Unknown feature code: '{code}'. Available: {list(code_to_name.keys())}")
        return columns

def get_seqs(df, feature_columns_used, prediction_columns_used):
    feature_seqs = df[feature_columns_used].to_numpy()
    prediction_seqs = df[prediction_columns_used].to_numpy()
    return feature_seqs, prediction_seqs

def get_fold_indices(total_length, num_folds=4):
    fold_size = math.ceil(total_length / num_folds)
    split_indices = [0]
    for i in range(1, num_folds):
        next_idx = min(fold_size * i, total_length)
        split_indices.append(next_idx)
    # Ensure the last index is exactly the total length
    if split_indices[-1] != total_length:
        split_indices.append(total_length)
    return split_indices

def make_sliding_window_ycustom_folds(x, y, window_size, horizon_size, num_folds=4):

    split_indices = get_fold_indices(len(x), num_folds)
    
    x_data_folds = []
    y_data_folds = []

    for k in range(num_folds):
        fold_x_data = []
        fold_y_data = []
        
        start_idx = split_indices[k]
        end_idx = split_indices[k+1]
        
        for i in range(start_idx, end_idx):
            if i + window_size + horizon_size <= end_idx:
                fold_x_data.append(x[i : i + window_size])
                # if cumulative_deltas: fold_y_data.append(np.cumsum(y[i + window_size : i + window_size + horizon_size]))
                fold_y_data.append(y[i + window_size : i + window_size + horizon_size])
            else:
                # Skip windows crossing fold boundaries
                pass
                
        x_data_folds.append(np.array(fold_x_data))
        y_data_folds.append(np.array(fold_y_data))
        
    return x_data_folds, y_data_folds


# CIRCUIT CONSTRUCTION
# ------------------------------------------------------------------------------------------------------------------------------------------------

def get_params_for_gates(chunk_idx, num_features, input_params, base_idx, rep_indices):
    """Helper to fetch 3 parameters for a U-gate, handling padding."""
    p = []
    for k in range(3):
        feat_ptr = chunk_idx * 3 + k
        if feat_ptr < num_features:
            real_feat_idx = rep_indices[feat_ptr]
            p.append(input_params[base_idx + real_feat_idx])
        else:
            p.append(0.0)
    return p

def apply_entanglement(qc, num_qubits, strategy='circular', layer_index=0):
    if num_qubits < 2: return
    if strategy == 'linear':
        for i in range(num_qubits - 1): 
            qc.cx(i, i + 1)
    elif strategy == 'reverse_linear':
        for i in range(num_qubits - 1, 0, -1): 
            qc.cx(i, i - 1)
    elif strategy == 'circular':
        for i in range(num_qubits): 
            qc.cx(i, (i + 1) % num_qubits) 
    elif strategy == 'full':
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits): 
                qc.cx(i, j)     
    elif strategy == 'pairwise':  # <--- Renamed from previous 'sca'
        for i in range(0, num_qubits - 1, 2): # Layer 1: Even pairs (0-1, 2-3...)
            qc.cx(i, i + 1)
        for i in range(1, num_qubits - 1, 2): # Layer 2: Odd pairs (1-2, 3-4...)
            qc.cx(i, i + 1)
    elif strategy == 'sca': # <--- True SCA implementation
        # Shifted Circular Alternating requires a shift based on layer_index
        shift = layer_index % num_qubits # Connect i to i+1, but shifted by the layer index
        for i in range(num_qubits):
            control = (i + shift) % num_qubits
            target = (i + shift + 1) % num_qubits
            qc.cx(control, target)
    else: 
        raise ValueError(f"Unknown entanglement strategy: {strategy}")

def append_ansatz_and_entangle(qc, args, weight_params, weight_idx, ansatz_obj, weights_per_layer, layer_idx):
    """Applies Entanglement + Ansatz (Common to all strategies)."""
    
    # 1. Entanglement
    apply_entanglement(qc, qc.num_qubits, strategy=args.entangle, layer_index=layer_idx)

    # 2. Ansatz (Trainable Weights)
    if args.ansatz == 'ugates': #TODO: Try different encodings (i.e. one qubit for every 3 features instead of consecutive ugates, or 3 features in each layer)
        for q in range(qc.num_qubits):  
            w1, w2, w3 = weight_params[weight_idx:weight_idx+3]
            qc.u(w1, w2, w3, q)
            weight_idx += 3
    else: 
        layer_weights = weight_params[weight_idx : weight_idx + weights_per_layer]
        bound_ansatz = ansatz_obj.assign_parameters(layer_weights) # NOTE: manually implementing the ansatz instead of using qiskit's template might be faster
        qc.compose(bound_ansatz, inplace=True)
        weight_idx += weights_per_layer
    
    return weight_idx

# --- STRATEGY 1: COMPACT (Stacked Gates) ---
def _build_compact_block(qc, args, config, input_params, weight_params, weight_idx, rep_indices, ansatz_obj, current_layer):
    """
    Encodes all features in 1 physical layer by stacking U-gates on the same qubit.
    Structure: [U(f1-3) -> U(f4-6)...] -> Entangle -> Ansatz
    """
    # 1. Encoding (Stack all chunks)
    for t in range(args.window_size):
        base_idx = t * config["num_features"]
        for chunk_idx in range(config["num_ugates"]):
            p = get_params_for_gates(chunk_idx, config["num_features"], input_params, base_idx, rep_indices)
            qc.u(p[0], p[1], p[2], t) # Always qubit 't'

    # 2. Entangle + Ansatz (Once per rep)
    weight_idx = append_ansatz_and_entangle(
        qc, args, weight_params, weight_idx, ansatz_obj, 
        config["weights_per_layer"], current_layer
    )
    return weight_idx, current_layer + 1

# --- STRATEGY 2: PARALLEL (Wider Circuit) ---
def _build_parallel_block(qc, args, config, input_params, weight_params, weight_idx, rep_indices, ansatz_obj, current_layer):
    """
    Encodes all features in 1 physical layer by using auxiliary qubits.
    Structure: [Q0: U(f1-3), Q1: U(f4-6)...] -> Entangle -> Ansatz
    """
    # 1. Encoding (Spread across width)
    for t in range(args.window_size):
        base_idx = t * config["num_features"]
        for chunk_idx in range(config["num_ugates"]):
            p = get_params_for_gates(chunk_idx, config["num_features"], input_params, base_idx, rep_indices)
            
            # Map timestep + chunk to specific qubit
            target_qubit = (t * config["qubits_per_step"]) + chunk_idx
            qc.u(p[0], p[1], p[2], target_qubit)

    # 2. Entangle + Ansatz (Once per rep)
    weight_idx = append_ansatz_and_entangle(
        qc, args, weight_params, weight_idx, ansatz_obj, 
        config["weights_per_layer"], current_layer
    )
    return weight_idx, current_layer + 1

# --- STRATEGY 3: SERIAL (Deeper Circuit) ---
def _build_serial_block(qc, args, config, input_params, weight_params, weight_idx, rep_indices, ansatz_obj, current_layer):
    """
    Encodes features across multiple physical layers.
    Structure: Layer A [U(f1-3)] -> Ent/Ans -> Layer B [U(f4-6)] -> Ent/Ans
    """
    # Iterate through each "Chunk" (set of 3 features) creating a new physical layer for each
    for s in range(config["num_ugates"]):
        
        # 1. Encoding (Only chunk 's')
        for t in range(args.window_size):
            base_idx = t * config["num_features"]
            p = get_params_for_gates(s, config["num_features"], input_params, base_idx, rep_indices)
            qc.u(p[0], p[1], p[2], t)

        # 2. Entangle + Ansatz (Happens AFTER EACH partial encoding)
        weight_idx = append_ansatz_and_entangle(
            qc, args, weight_params, weight_idx, ansatz_obj, 
            config["weights_per_layer"], current_layer
        )
        current_layer += 1
        
    return weight_idx, current_layer

def get_encoding_config(args):

    """Calculates dimensions based on strategy."""
    num_features = len(args.features)
    num_ugates = math.ceil(num_features / 3)

    if args.encoding == 'compact': # STANDARD: Standard width, stacked ugates in one layer
        qubits_per_step = 1
        num_qubits = args.window_size
        sub_layers_per_rep = 1
    elif args.encoding == 'parallel': # WIDER: Each timestep uses 'num_ugates' qubits
        qubits_per_step = num_ugates
        num_qubits = args.window_size * qubits_per_step
        sub_layers_per_rep = 1
    elif args.encoding == 'serial': # DEEPER: Standard width, but split encoding into multiple layers
        qubits_per_step = 1
        num_qubits = args.window_size
        sub_layers_per_rep = num_ugates
        
    return {
        "num_features": num_features,
        "num_ugates": num_ugates,
        "strategy": args.encoding,
        "num_qubits": num_qubits,
        "total_physical_layers": args.reps * sub_layers_per_rep,
        "qubits_per_step": qubits_per_step
    }

def create_multivariate_circuit(args, barriers=False): #TODO: Check if barriers have any effect

    # 1. Setup
    config = get_encoding_config(args)
    
    # 2. Ansatz Object Init
    if args.ansatz == 'ugates': 
        ansatz_obj = 'ugates'
        config["weights_per_layer"] = 3 * config["num_qubits"]
    elif args.ansatz == 'efficientsu2': 
        ansatz_obj = EfficientSU2(num_qubits=config["num_qubits"], reps=0)
        config["weights_per_layer"] = len(ansatz_obj.parameters)
    elif args.ansatz == 'realamplitudes': 
        ansatz_obj = RealAmplitudes(num_qubits=config["num_qubits"], reps=0)
        config["weights_per_layer"] = len(ansatz_obj.parameters)

    qc = QuantumCircuit(config["num_qubits"])
    input_params = ParameterVector('θ', args.window_size * config["num_features"])
    weight_params = ParameterVector('ω', config["total_physical_layers"] * config["weights_per_layer"])
    
    rng = np.random.default_rng(args.run)
    weight_idx = 0
    current_physical_layer = 0
    rep_indices = np.arange(config["num_features"])

    for r in range(args.reps):

        if config["strategy"] == 'compact':
            weight_idx, current_physical_layer = _build_compact_block(
                qc, args, config, input_params, weight_params, weight_idx, rep_indices, ansatz_obj, current_physical_layer
            )
        elif config["strategy"] == 'serial':
            weight_idx, current_physical_layer = _build_serial_block(
                qc, args, config, input_params, weight_params, weight_idx, rep_indices, ansatz_obj, current_physical_layer
            )
        elif config["strategy"] == 'parallel':
            weight_idx, current_physical_layer = _build_parallel_block(
                qc, args, config, input_params, weight_params, weight_idx, rep_indices, ansatz_obj, current_physical_layer
            )
        
        if barriers: qc.barrier()

        if args.reorder: 
            rep_indices = rng.permutation(rep_indices)

    return qc, input_params, weight_params


# MODEL DEFINITIONS
# ------------------------------------------------------------------------------------------------------------------------------------------------

class WindowEncodingQNN: # Numpy version

    def __init__(self, qnn, output_shape, seed):

        self.qnn = qnn
        self.input_dim = qnn.circuit.num_qubits
        self.horizon = output_shape[1]
        self.columns = output_shape[2]
        self.num_q_params = qnn.num_weights
        self.output_dim = self.horizon*self.columns
        self.num_c_params = (self.input_dim * self.output_dim) + self.output_dim # Classical Linear Readout Layer: Weights matrix (Inputs*Outputs) + Bias (Outputs)
        self.total_params = self.num_q_params + self.num_c_params
        print(f"Model Init: {self.input_dim} Qubits. Params: {self.num_q_params} (Q) + {self.num_c_params} (C)")
        if seed is not None: self.rng = np.random.default_rng(seed)

    def forward(self, x, params_flat): # Forward: Input -> QNN  -> Measure All Z -> Linear Layer (W*x + b) -> Output
        
        # QNN layer
        q_params = params_flat[:self.num_q_params]
        x_flat = x.reshape(x.shape[0], -1)
        y = self.qnn.forward(x_flat, q_params)

        # Readout layer
        c_params = params_flat[self.num_q_params:]
        W = c_params[:self.input_dim * self.output_dim].reshape(self.input_dim, self.output_dim)
        b = c_params[self.input_dim * self.output_dim:]
        y = np.dot(y, W) + b
        return y.reshape(x.shape[0], self.horizon, self.columns)
    
    def initialize_parameters(self, strategy):

        if strategy == 'identity': q_params = self.rng.uniform(-0.1, 0.1, size=self.num_q_params) # (almost) identity matrix initialization
        elif strategy == 'uniform': q_params = self.rng.uniform(0, 2*np.pi, size=self.num_q_params) # uniform random initialization
        limit = np.sqrt(6 / (self.input_dim + self.output_dim))
        c_params = self.rng.uniform(-limit, limit, size=self.num_c_params) # xavier/glorot initialization
        return np.concatenate([q_params, c_params])

class ClassicalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        
        # LSTM Layer
        # input_shape: (Batch, Seq_Len, Features)
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Readout Layer
        self.fc = nn.Linear(hidden_size, output_size,device=self.device)
    def forward(self, x):
        # x shape: (Batch, Window_Size, Features)
        # lstm_out shape: (Batch, Window_Size, Hidden_Size)
        lstm_out, _ = self.lstm(x)
        
        # We take the output of the LAST time step in the window to predict the future
        last_time_step = lstm_out[:, -1, :] 
        
        out = self.fc(last_time_step)
        return out
    
class ClassicalWrapper:
    def __init__(self, torch_model, device,output_shape = None):
        self.model = torch_model
        self.device = device
        self.num_targets = output_shape[2] if output_shape else 2
            
    def forward(self, x, params):
        # params is actually the state_dict which we load
        self.model.load_state_dict(params)
        self.model.to(self.device)
        self.model.eval()
        t_x = torch.tensor(x, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            out = self.model(t_x)
        total_out = out.shape[1]
        horizon = total_out // self.num_targets
        
        return out.cpu().numpy().reshape(x.shape[0], horizon, self.num_targets)


# TRAIN AND EVALUATE MODEL
# ------------------------------------------------------------------------------------------------------------------------------------------------

def compute_loss(args, pred, target, reconstruct, scaler=None):
    num_targets = target.shape[-1]
    
    if reconstruct and scaler:

        target_real = scaler.inverse_transform(target.reshape(-1, num_targets)).reshape(target.shape)
        pred_real = scaler.inverse_transform(pred.reshape(-1, num_targets)).reshape(pred.shape)

        if args.predict == 'delta':
            target_traj = np.cumsum(target_real, axis=1)
            pred_traj = np.cumsum(pred_real, axis=1)
            return np.mean((pred_traj - target_traj) ** 2)
        
        else: return np.mean((pred_real - target_real) ** 2)

    else:
        if reconstruct and not scaler: print('Warning: Scaler was not provided and trajectory could not be reconstructed.')
        return np.mean((pred - target) ** 2)

def train_model(args, model, x_train, y_train, x_val, y_val, scaler=None):

    best_val_loss = float('inf')
    best_params = None
    
    train_history = []
    val_history = []

    def objective_function(params):
        nonlocal best_val_loss, best_params

        preds = model.forward(x_train, params)
        train_mse = compute_loss(args, preds, y_train, args.reconstruct_train, scaler)

        val_preds = model.forward(x_val, params)
        val_mse = compute_loss(args, val_preds, y_val, args.reconstruct_val, scaler)
        
        train_history.append(train_mse)
        val_history.append(val_mse)
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            best_params = np.copy(params) #NOTE: Usually final weights are better
            
        if len(train_history) % 50 == 0:
            print(f"Iter {len(train_history):4d} | Train MSE: {train_mse:.5f} | Val MSE: {val_mse:.5f}")
            
        return train_mse

    start_time = time.time()
    print(f"\nStarting {args.optimizer.upper()} optimization...")
    initial_weights = model.initialize_parameters(args.initialization)

    if args.optimizer.upper() == 'COBYLA':
        opt = COBYLA(maxiter=args.maxiter,tol = None) #TODO: Add tolerance
        res = opt.minimize(fun=objective_function, x0=initial_weights)

    elif args.optimizer.upper() == 'SPSA':
        opt = SPSA(maxiter=args.maxiter) #TODO: Add learning_rate and perturbation
        res = opt.minimize(fun=objective_function, x0=initial_weights)

    elapsed_time = (time.time() - start_time) / 60
    print(f"Training completed in {elapsed_time:.2f} min.")

    return {
        "best_weights": best_params,
        "best_val_loss": best_val_loss,     
        "final_weights": res.x,             
        "train_history": train_history,     
        "val_history": val_history,         
        "nfev": res.nfev if hasattr(res, 'nfev') else len(train_history)
    }


def train_classical_model(args, model, x_train, y_train, x_val, y_val, y_scaler = None,device='cpu'):
    """
    Standard PyTorch training loop with Adam optimizer.
    """
    # 1. Prepare Data Loaders
    batch_size = args.batch_size if hasattr(args, 'batch_size') else 32

    # Convert numpy to torch tensors
    t_x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    t_y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    t_x_val = torch.tensor(x_val, dtype=torch.float32).to(device)
    t_y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_ds = TensorDataset(t_x_train, t_y_train)
    val_ds = TensorDataset(t_x_val, t_y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 2. Setup Training
    model = model.to(device)
    criterion = nn.MSELoss()
    if args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError("Unsupported optimizer")

    best_val_loss = float('inf')
    best_weights = None
    train_history = []
    val_history = []
    
    # Early stopping config
    patience = getattr(args, 'patience', 20)
    counter = 0
    
    print(f"\n--- Starting Classical Optimization (Adam) on {device} ---")
    start_time = time.time()
    for epoch in range(args.maxiter): # 'maxiter' acts as 'epochs' here
        model.train()
        epoch_loss = 0
        
        for x_batch, y_batch in train_loader:
            # Forward
            x_batch,y_batch = x_batch.to(device),y_batch.to(device)
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_train_loss = epoch_loss / len(train_loader)
        train_history.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss_accum = 0
        total_samples = 0
        num_targets = y_val.shape[-1]
        with torch.no_grad():
            for x_v, y_v in val_loader:
                val_preds = model(x_v)
                v_p_np = val_preds.cpu().numpy().reshape(-1, args.horizon, num_targets)
                v_y_np = y_v.cpu().numpy().reshape(-1, args.horizon, num_targets)                
                batch_loss = compute_loss(args, v_p_np, v_y_np, args.reconstruct_val, y_scaler) 
                val_loss_accum += batch_loss * x_v.size(0)
                total_samples += x_v.size(0)
        
        avg_val_loss = val_loss_accum / total_samples
        val_history.append(avg_val_loss)

        # Logging
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:4d} | Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f}")

        # Save Best Model Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_weights = model.state_dict() # Save PyTorch state dict
            counter = 0
        else:
            counter += 1
            
        # Early Stopping
        if counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    total_time = (time.time() - start_time) / 60
    print(f"Classical training finished in {total_time:.2f} min.")
    
    # IMPORTANT: Unlike QNN which returns a flat array, PyTorch returns a state_dict.
    # We must return it in a format compatible with your evaluation functions.
    # But since your eval functions expect `model.forward(x, params)`, 
    # we need a wrapper for inference later or just load weights into the model object now.
    
    # Reload best weights into the model for final state
    if best_weights is not None:
        model.load_state_dict(best_weights)
    
    # Return results dictionary
    return {
        "best_weights": best_weights, 
        "final_weights": model.state_dict(),
        "best_val_loss": best_val_loss,
        "train_history": train_history,
        "val_history": val_history,
        "nfev": epoch + 1
    }


def recursive_forward_pass(args, model, best_params, x_test, x_scaler, y_scaler):
    """
    Performs Recursive 'Fan' Prediction for N targets (Generic).
    Supports two modes:
      1. 'motion': Model predicts absolute values -> Direct Feedback.
      2. 'delta':  Model predicts changes -> Integration Feedback (New = Old + Pred).
    """
    
    # 1. Setup
    num_samples = x_test.shape[0] # N
    horizon = args.horizon        # H
    num_targets = len(args.targets)


    # 2. Map Targets to Feature Indices (for feedback)
    # We need to know: Output #0 corresponds to Input Feature #X?
    target_to_feature_idx = {}
    # Determine integration mode based on args
    is_delta_mode = (args.predict == 'delta')
    mode_label = "Integration (Delta)" if is_delta_mode else "Direct (Motion)"

    print(f"\n  {'Target Output':<20} | {'Input Feature':<20} | {'Feedback Mode'}")
    print(f"  {'-'*20}-+-{'-'*20}-+-{'-'*20}")
    
    for t_idx, t_name in enumerate(args.targets):
        # If we predict 'delta', the target name usually matches the feature name 
        # (e.g. we predict 'Surge Velocity' change, to update 'Surge Velocity' feature)
        # Note: If your target names are explicitly "Delta Surge", you might need a mapping dictionary here.
        # Assuming args.targets contains names that exist in args.features:
        if t_name in args.features:
            f_idx = args.features.index(t_name)
            target_to_feature_idx[t_idx] = f_idx
            print(f"  {t_name:<20} | {t_name:<20} | ✅ {mode_label}")
        else:
            print(f"  {t_name:<20} | {'None':<20} | ⚠️ Open-Loop")

    recursive_ratio = len(target_to_feature_idx) / len(args.targets) if args.targets else 0.0

    if not target_to_feature_idx:
        print("  > [Warning] No autoregressive targets found. Using Open-Loop.")
        return model.forward(x_test, best_params), 0.0

    # 3. Initialization
    # Initialize with the very first window
    curr_window = x_test[0:1, :, :] 
    
    # Track the "Real Physics" state of the last step to support integration
    # Shape (1, F)
    last_step_real = x_scaler.inverse_transform(curr_window[:, -1, :])

    # Output Container
    recursive_preds = np.zeros((num_samples, horizon, num_targets))

    # 4. Main Loop
    for i in range(num_samples):
        
        # A. Predict Full Horizon
        # model.forward returns (1, H, 2). We keep ALL of it.
        preds_full = model.forward(curr_window, best_params) # Shape (1, H, 2)
        recursive_preds[i] = preds_full[0]
        # B. Recursion: Prepare Next Window
        # We need to construct the input for t+1. 
        # Start by grabbing the Ground Truth (GT) for t+1 to fill in auxiliary features (like Rudder, RPM)
        next_gt_idx = min(i + 1, num_samples - 1)
        next_gt_features_norm = x_test[next_gt_idx:next_gt_idx+1, -1, :] 
        
        # Denormalize everything to work in Physics Space
        next_input_real = x_scaler.inverse_transform(next_gt_features_norm) # (1, F)
        
        # Get the immediate next prediction (step 0 of horizon)
        pred_next_step_norm = preds_full[:, 0, :]
        if y_scaler:
            pred_step_real = y_scaler.inverse_transform(pred_next_step_norm) # (1, T)
        else:
            pred_step_real = pred_next_step_norm
            
        # C. Update the features that are driven by the model
        for t_idx, f_idx in target_to_feature_idx.items():
            pred_val = pred_step_real[0, t_idx]
            
            if is_delta_mode:
                # INTEGRATION LOGIC: New State = Old State + Prediction
                # We use 'last_step_real' which holds the *simulated* state from the previous iteration
                new_val = last_step_real[0, f_idx] + pred_val
                next_input_real[0, f_idx] = new_val
            else:
                # DIRECT LOGIC: New State = Prediction
                next_input_real[0, f_idx] = pred_val

        # D. Update 'last_step_real' for the next iteration (i+2)
        last_step_real = next_input_real

        # E. Normalize and Shift Window
        new_row_norm = x_scaler.transform(next_input_real)
        curr_window = np.concatenate([curr_window[:, 1:, :], new_row_norm.reshape(1, 1, -1)], axis=1)

    return recursive_preds, recursive_ratio

def evaluate_model(args, model, params, x_test, y_test, x_scaler, y_scaler):
    """
    Runs BOTH One-Step (Teacher Forcing) and Recursive (Dead Reckoning) evaluations.
    """

    results = {}
    orig_shape = y_test.shape # (N, H, T)
    num_targets = orig_shape[-1]
    # ONE-STEP EVALUATION
    preds_norm_step = model.forward(x_test, params) 
    
    # Unscale
    if y_scaler:
        preds_real_step = y_scaler.inverse_transform(preds_norm_step.reshape(-1, num_targets)).reshape(orig_shape)
        y_gt_real = y_scaler.inverse_transform(y_test.reshape(-1, num_targets)).reshape(orig_shape)
    else:
        preds_real_step = preds_norm_step
        y_gt_real = y_test

    # Step metrics
    results['Step_MSE'] = mean_squared_error(y_gt_real.reshape(-1, num_targets), preds_real_step.reshape(-1, num_targets))
    results['Step_R2'] = r2_score(y_gt_real.reshape(-1, num_targets), preds_real_step.reshape(-1, num_targets))

    if args.predict == 'delta':
        true_path_backbone = np.concatenate([np.zeros((1, num_targets)), np.cumsum(y_gt_real[:, 0, :], axis=0)])
        true_path = true_path_backbone[:-1, None, :] + np.cumsum(y_gt_real, axis=1)
        pred_path_local = true_path_backbone[:-1, None, :] + np.cumsum(preds_real_step, axis=1)
        pred_path_backbone = np.concatenate([np.zeros((1, num_targets)), np.cumsum(preds_real_step[:, 0, :], axis=0)])
        pred_path_global_open = pred_path_backbone[:-1, None, :] + np.cumsum(preds_real_step, axis=1)
    else:
        true_path_backbone = y_gt_real[:, 0, :]
        true_path = y_gt_real
        pred_path_local = preds_real_step

        pred_path_backbone = preds_real_step[:, 0, :]
        pred_path_global_open = preds_real_step

    results['Local_MSE'] = mean_squared_error(true_path.reshape(-1, num_targets), pred_path_local.reshape(-1, num_targets))
    norm_local_error = np.linalg.norm(true_path - pred_path_local, axis=2)
    results['Local_ADE'] = np.mean(norm_local_error)

    results['Global_open_MSE'] = mean_squared_error(true_path.reshape(-1, num_targets), pred_path_global_open.reshape(-1, num_targets))
    results['Global_open_R2'] = r2_score(true_path.reshape(-1, num_targets), pred_path_global_open.reshape(-1, num_targets))
    norm_global_error = np.linalg.norm(true_path - pred_path_global_open, axis=2)
    results['Global_open_ADE'] = np.mean(norm_global_error)    
    results['Global_open_FDE'] = norm_global_error[-1,-1]      
    results['Global_open_Max'] = np.max(norm_global_error)   

    # RECURSIVE EVALUATION (Updated Unpacking)
    preds_norm_rec, rec_ratio = recursive_forward_pass(args, model, params, x_test, x_scaler, y_scaler)
    results['Recursivity'] = rec_ratio  # <--- Store Metric
    
    # Unscale
    if y_scaler:
        preds_real_rec = y_scaler.inverse_transform(preds_norm_rec.reshape(-1, num_targets)).reshape(orig_shape)
    else:
        preds_real_rec = preds_norm_rec
    
    if args.predict == 'delta':
        pred_path_backbone = np.concatenate([np.zeros((1, num_targets)), np.cumsum(preds_real_rec[:, 0, :], axis=0)])
        pred_path_global = pred_path_backbone[:-1, None, :] + np.cumsum(preds_real_rec, axis=1)
    else:
        pred_path_backbone = preds_real_rec[:, 0, :]
        pred_path_global = preds_real_rec
        
    results['Global_closed_MSE'] = mean_squared_error(true_path.reshape(-1, num_targets), pred_path_global.reshape(-1, num_targets))
    results['Global_closed_R2'] = r2_score(true_path.reshape(-1, num_targets), pred_path_global.reshape(-1, num_targets))
    norm_global_error = np.linalg.norm(true_path - pred_path_global, axis=2)
    results['Global_closed_ADE'] = np.mean(norm_global_error)    
    results['Global_closed_FDE'] = norm_global_error[-1,-1]      
    results['Global_closed_Max'] = np.max(norm_global_error) 

    return {
        "true_deltas_denorm": y_gt_real,
        "true_backbone": true_path_backbone,
        "true_path": true_path,
        "local":{              
            "pred_deltas_denorm": preds_real_step,
            "pred_path": pred_path_local,         
        },
        "global":{
            "closed":{
                "pred_deltas_denorm": preds_real_rec,
                "pred_path": pred_path_global,
            },
            "open":{
                "pred_deltas_denorm": preds_real_step,
                "pred_path": pred_path_global_open,
            }
               
        },
        "metrics": results
    }


# SAVE AND PLOT RESULTS
# ------------------------------------------------------------------------------------------------------------------------------------------------

def save_experiment_results(args, train_results, best_eval, final_eval, scalers, qnn_dict, timestamp):
    
    for folder in ["models", "logs", "figures"]:
        os.makedirs(folder, exist_ok=True)
    
    total_params = len(train_results['final_weights'])
    num_q_params = len(qnn_dict['weight_params'])
    num_c_params = total_params - num_q_params

    # Save model in pickle file
    short_args = map_names([args.ansatz, args.entangle], reverse=True)
    model_filename = f"models/{timestamp}_{args.model}_f{len(args.features)}_w{args.window_size}_h{args.horizon}_{short_args[0]}_{short_args[1]}_r{args.reps}.pkl"
    save_payload = {
        "config": vars(args),
        "best_weights": train_results['best_weights'],
        "final_weights": train_results['final_weights'],
        "train_history": train_results['train_history'],
        "val_history": train_results['val_history'],
        "best_eval_metrics": best_eval['metrics'],
        "final_eval_metrics": final_eval['metrics'],
        "y_scaler": scalers[1],
        "x_scaler": scalers[0],
        "qnn_structure": qnn_dict
    }
    with open(model_filename, "wb") as f:
        pickle.dump(save_payload, f)

    # Save experiment summary to excel
    excel_filename = "logs/experiments_summary.xlsx"
    m_final = final_eval['metrics']
    
    # Timestamp to date format
    try:
        dt_temp = datetime.datetime.strptime(timestamp, "%m-%d_%H-%M-%S")
        dt_object = dt_temp.replace(year=2026)
    except ValueError:
        dt_object = timestamp 
    
    ignore_keys = [
        'select_features', 'drop_features', # Redundant
        'save_plot', 'show_plot',           # UI flags
        'initialization'
    ]

    # Helper: Format values (Bool -> lowercase, List -> string)
    def clean_val(v):
        if isinstance(v, bool): return str(v).lower()
        if isinstance(v, list): return str(v)
        return v
    def normalize_loaded_bools(val):
        # If pandas loaded it as a real boolean
        if isinstance(val, bool):
            return str(val).lower()
        # If pandas loaded it as a string but it's "VERDADERO" or "TRUE"
        if isinstance(val, str):
            if val.upper() in ['TRUE', 'VERDADERO']: return 'true'
            if val.upper() in ['FALSE', 'FALSO']: return 'false'
        return val
    
    # 1. Add args (filtering ignores and formatting)
    raw_data = {}
    for key, value in vars(args).items():
        if key not in ignore_keys:
            if key in ['features', 'targets']:
                # Convert full names to short codes before saving
                short_list = map_names(value, reverse=True)
                raw_data[key] = clean_val(short_list)
            else:
                raw_data[key] = clean_val(value)

    # 2. Add Metrics & Params
    metrics_flat = {
        "step MSE": m_final.get('Step_MSE'),
        "step R2": m_final.get('Step_R2'),

        "local MSE": m_final.get('Local_MSE'),
        "local ADE": m_final.get('Local_ADE'),

        "global open MSE": m_final.get('Global_open_MSE'),
        "global open R2": m_final.get('Global_open_R2'),
        "global open ADE": m_final.get('Global_open_ADE'),
        "global open FDE": m_final.get('Global_open_FDE'),
        "global closed MSE": m_final.get('Global_closed_MSE'),
        "global closed R2": m_final.get('Global_closed_R2'),
        "global closed ADE": m_final.get('Global_closed_ADE'),
        "global closed FDE": m_final.get('Global_closed_FDE'),

        "recursivity": m_final.get('Recursivity'),
        "final val loss": train_results['val_history'][-1] if train_results['val_history'] else None,
        "total params": total_params,
        "q params": num_q_params,
        "c params": num_c_params,
        "iterations": len(train_results['train_history']),
        "date": dt_object,
        "model_id": model_filename.split('/')[-1]
    }
    raw_data.update(metrics_flat)

    column_order = [
        # 1. ID & Meta
        "date", "model_id", "run", "testing_fold",
        
        # 2. Dataset Info
        "data", "data_n", "data_dt", 
        "features", "targets", "window_size", "horizon", 
        "predict", "norm", "reconstruct_train", "reconstruct_val",
        
        # 3. Model Architecture
        "model", "encoding", "ansatz", "entangle", "reps", "reorder", "initialization",
        
        # 4. Complexity
        "total params", "q params", "c params",
        
        # 5. Training Config
        "optimizer", "maxiter", "tolerance", "iterations", "final val loss",
        
        # 7. Key Metrics
        "step MSE", "step R2",
        "local MSE", "local ADE",
        "global open MSE", "global open ADE", "global open FDE", "global open R2", 
        "global closed MSE", "global closed ADE", "global closed FDE", "global closed R2", "recursivity"

    ]

    # Construct the final ordered dictionary
    ordered_row = {}
    
    # Add ordered keys first
    for col in column_order:
        if col in raw_data:
            ordered_row[col] = raw_data.pop(col) # Remove from raw_data as we add
            
    # Add whatever is left (e.g. unknown new args) at the end
    for k, v in raw_data.items():
        ordered_row[k] = v
    
    df_new = pd.DataFrame([ordered_row])
    if os.path.exists(excel_filename):
        try:
            df_existing = pd.read_excel(excel_filename)
            df_existing = df_existing.applymap(normalize_loaded_bools)
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
            df_final.to_excel(excel_filename, index=False)
        except Exception as e:
            print(f"[Warning] Excel error: {e}. Saving to CSV backup.")
            df_new.to_csv(f"logs/backup_{timestamp}.csv", index=False)
    else:
        df_new.to_excel(excel_filename, index=False)

    # Text log
    log_filename = "logs/experiment_log.txt"
    short_feats = ", ".join(map_names(args.features, reverse=True))
    encd = getattr(args, 'encoding', 'N/A')
    ansatz = getattr(args, 'ansatz', 'N/A')
    ent = getattr(args, 'entangle', 'N/A')
    reps = getattr(args, 'reps', 'N/A')

    log_entry = (
        f"[{timestamp}] "
        f"{args.model:<10} | "
        f"F={len(args.features):<2} W={args.window_size:<2} H={args.horizon:<2} | "
        f"Circuit: {encd:<10} {ansatz:<15} {ent:<14} reps={reps:<2} | "
        f"Features: {short_feats} \n"
    )
    
    with open(log_filename, "a") as f:
        f.write(log_entry)
    print(f"\n[Logger] Model saved to {model_filename}")
    print(f"[Logger] Stats appended to {excel_filename}")

def save_classical_results(args, train_results, best_eval, final_eval, scalers, timestamp):
    """
    Saves detailed results specifically for Classical Models (LSTM/RNN).
    Saves to: logs/classical_experiments_summary.xlsx
    """
    # 1. Create Folders
    for folder in ["models", "logs", "figures"]:
        os.makedirs(folder, exist_ok=True)

    state_dict = train_results['final_weights']
    total_params = sum(p.numel() for p in state_dict.values())

    # 2. Save Model (Pickle)
    model_filename = f"models/{timestamp}_classical_f{len(args.features)}_w{args.window_size}_h{args.horizon}.pkl"
    
    save_payload = {
        "config": vars(args),
        "best_weights": train_results['best_weights'],
        "final_weights": train_results['final_weights'],
        "train_history": train_results['train_history'],
        "val_history": train_results['val_history'],
        "best_eval_metrics": best_eval['metrics'],
        "final_eval_metrics": final_eval['metrics'],
        "y_scaler": scalers[1],
        "x_scaler": scalers[0]
    }
    
    with open(model_filename, "wb") as f:
        pickle.dump(save_payload, f)

    # 3. Save Summary to Excel
    excel_filename = "logs/classical_experiments_summary.xlsx"
    m_final = final_eval['metrics']
    
    # Timestamp to date format (Default 2026)
    try:
        dt_temp = datetime.datetime.strptime(timestamp, "%m-%d_%H-%M-%S")
        dt_object = dt_temp.replace(year=2026)
    except ValueError:
        dt_object = timestamp 
    ignore_keys = [
        'select_features', 'drop_features', 
        'save_plot', 'show_plot',
    ]
    def clean_val(v):
        if isinstance(v, bool): return str(v).lower()
        if isinstance(v, list): return str(v)
        return v
    def normalize_loaded_bools(val):
        # If pandas loaded it as a real boolean
        if isinstance(val, bool):
            return str(val).lower()
        # If pandas loaded it as a string but it's "VERDADERO" or "TRUE"
        if isinstance(val, str):
            if val.upper() in ['TRUE', 'VERDADERO']: return 'true'
            if val.upper() in ['FALSE', 'FALSO']: return 'false'
        return val
    raw_data = {}
    for key, value in vars(args).items():
        if key not in ignore_keys:
            if key in ['features', 'targets']:
                short_list = map_names(value, reverse=True)
                raw_data[key] = clean_val(short_list)
            else:
                raw_data[key] = clean_val(value)
            
    metrics_flat = {
        "step MSE": m_final.get('Step_MSE'),
        "step R2": m_final.get('Step_R2'),

        "local MSE": m_final.get('Local_MSE'),
        "local ADE": m_final.get('Local_ADE'),

        "global open MSE": m_final.get('Global_open_MSE'),
        "global open R2": m_final.get('Global_open_R2'),
        "global open ADE": m_final.get('Global_open_ADE'),
        "global open FDE": m_final.get('Global_open_FDE'),
        "global closed MSE": m_final.get('Global_closed_MSE'),
        "global closed R2": m_final.get('Global_closed_R2'),
        "global closed ADE": m_final.get('Global_closed_ADE'),
        "global closed FDE": m_final.get('Global_closed_FDE'),


        "final val loss": train_results['val_history'][-1] if train_results['val_history'] else None,
        "total params": total_params,
        "iterations": len(train_results['train_history']),
        "date": dt_object,
        "model_id": model_filename.split('/')[-1],
        "model": "classical_lstm" # Explicitly naming the architecture
    }    
    
    raw_data.update(metrics_flat)
    
    column_order = [
        # 1. ID & Meta
        "date", "model_id", "run", "testing_fold",
        
        # 2. Dataset Info
        "data", "data_n", "data_dt", 
        "features", "targets", "window_size", "horizon", 
        "predict", "norm", "reconstruct_train", "reconstruct_val",
        
        # 3. Model Architecture (Classical Specific)
        "model", "hidden_size", "layers", 
        
        # 4. Complexity
        "total params",
        
        # 5. Training Config
        "optimizer", "maxiter", "learning_rate", "batch_size", "patience", "iterations", "final val loss",
        
        # 6. Key Metrics
        "step MSE", "step R2",
        "local MSE", "local ADE",
        "global open MSE", "global open ADE", "global open FDE", "global open R2", 
        "global closed MSE", "global closed ADE", "global closed FDE", "global closed R2", "recursivity"
    ]
    # Construct the final ordered dictionary
    ordered_row = {}
    
    # Add ordered keys first
    for col in column_order:
        if col in raw_data:
            ordered_row[col] = raw_data.pop(col) 
            
    # Add whatever is left at the end
    for k, v in raw_data.items():
        ordered_row[k] = v
    df_new = pd.DataFrame([ordered_row])

    if os.path.exists(excel_filename):
        try:
            df_existing = pd.read_excel(excel_filename)
            df_existing = df_existing.applymap(normalize_loaded_bools)
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
            df_final.to_excel(excel_filename, index=False)
        except Exception as e:
            print(f"[Warning] Excel error: {e}. Saving to CSV backup.")
            df_new.to_csv(f"logs/classical_backup_{timestamp}.csv", index=False)
    else:
        df_new.to_excel(excel_filename, index=False)

    # 4. Text Log
    log_filename = "logs/experiment_log.txt"
    short_feats = ", ".join(map_names(args.features, reverse=True))
    hidden = getattr(args, 'hidden_size', 'N/A')
    
    log_entry = (
        f"[{timestamp}] "
        f"{'classical':<10} | "
        f"F={len(args.features):<2} W={args.window_size:<2} H={args.horizon:<2} | "
        f"Hidden Size: {hidden:<45} | " # Padding ~48 chars to match the QNN 'Circuit' block width
        f"Features: {short_feats} \n"
    )
    with open(log_filename, "a") as f:
        f.write(log_entry)
        
    print(f"\n[Logger] Classical model saved to {model_filename}")
    print(f"[Logger] Stats appended to {excel_filename}")
    
def load_experiment_results(filepath):
    """
    Loads a saved experiment pickle file and prints a summary.
    automatically detecting if it is a Quantum or Classical model to 
    display the relevant hyperparameters.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    print(f"Loading results from: {filepath} ...")
    
    with open(filepath, "rb") as f:
        data = pickle.load(f)
        
    config = data.get('config', {})
    m_best = data.get('best_eval_metrics', {})
    m_final = data.get('final_eval_metrics', {})
    
    if config.get('model') is None:
        if 'classical' in filepath.lower() or 'hidden_size' in config:
            config['model'] = 'classical_lstm'
        else:
            config['model'] = 'N/A'

    # --- Print Summary ---
    print("\n" + "="*85)
    print(f"EXPERIMENT SUMMARY")
    print("="*85)
    fname = os.path.basename(filepath)
    try:
        parts = fname.split('_')
        timestamp = f"{parts[0]}_{parts[1]}"
    except:
        timestamp = "Unknown"
        
    print(f"Timestamp: {timestamp}")
    
    # --- DYNAMIC CONFIGURATION PRINTING ---
    print("\n--- Configuration ---")
    
    # 1. Common Keys (Shared by both)
    common_keys = ['model', 'features', 'window_size', 'horizon', 'optimizer']
    
    # 2. Model-Specific Keys
    quantum_keys = ['ansatz', 'encoding', 'reps', 'entangle']
    classical_keys = ['hidden_size', 'layers', 'learning_rate', 'batch_size', 'patience']

    # Detect Model Type
    model_type = config.get('model', 'unknown')
    
    # Select which keys to show
    keys_to_show = common_keys.copy()
    
    if 'lstm' in model_type.lower() or 'classical' in filepath.lower():
        keys_to_show.extend(classical_keys)
    else:
        # Default to Quantum if not explicitly classical
        keys_to_show.extend(quantum_keys)

    # Print the selected keys
    for k in keys_to_show:
        val = config.get(k, 'N/A')
        
        # Format lists nicely
        if isinstance(val, list) and k == 'features':
            val = f"{len(val)} features"
        elif isinstance(val, list):
            val = str(val)
            
        print(f"{k:<15}: {val}")
    
    print("\n--- Training Stats ---")
    train_hist = data.get('train_history', [])
    print(f"Total Iterations : {len(train_hist)}")
    if train_hist:
        print(f"Final Train Loss : {train_hist[-1]:.6f}")
    # Check for validation history
    val_hist = data.get('val_history', [])
    if val_hist:
        print(f"Final Val Loss   : {val_hist[-1]:.6f}")
            
    print("\n--- Performance Comparison (Best vs. Final Weights) ---")
    print(f"{'METRIC':<25} | {'BEST WEIGHTS':<18} | {'FINAL WEIGHTS':<18}")
    print("-" * 85)
    
    if m_best and m_final:
        def get_fmt(metrics, key):
            # Try exact key first, then lowercase match if needed (for backward compatibility)
            val = metrics.get(key)
            if val is None:
                # Try finding case-insensitive match
                key_lower = key.lower().replace(" ", "_")
                for k, v in metrics.items():
                    if k.lower().replace(" ", "_") == key_lower:
                        val = v
                        break
            
            if val is None or val == -1: return "N/A"
            return f"{val:.6f}" if isinstance(val, (int, float)) else str(val)

        # --- 1. Step Metrics ---
        print(f"{'Step MSE':<25} | {get_fmt(m_best, 'Step_MSE'):<18} | {get_fmt(m_final, 'Step_MSE'):<18}")
        print(f"{'Step R2 Score':<25} | {get_fmt(m_best, 'Step_R2'):<18} | {get_fmt(m_final, 'Step_R2'):<18}")

        # --- 2. Local Trajectory Metrics ---
        print("-" * 85)
        print(f"{'Local Traj MSE':<25} | {get_fmt(m_best, 'Local_MSE'):<18} | {get_fmt(m_final, 'Local_MSE'):<18}")
        print(f"{'Local ADE (m)':<25} | {get_fmt(m_best, 'Local_ADE'):<18} | {get_fmt(m_final, 'Local_ADE'):<18}")

        # --- 3. Global Trajectory Metrics Open ---
        print("-" * 85)
        print(f"{'Global Open Traj MSE':<25} | {get_fmt(m_best, 'Global_open_MSE'):<18} | {get_fmt(m_final, 'Global_open_MSE'):<18}")
        print(f"{'Global Open ADE (m)':<25} | {get_fmt(m_best, 'Global_open_ADE'):<18} | {get_fmt(m_final, 'Global_open_ADE'):<18}")
        print(f"{'Global Open FDE (m)':<25} | {get_fmt(m_best, 'Global_open_FDE'):<18} | {get_fmt(m_final, 'Global_open_FDE'):<18}")
        print(f"{'Global Open R2 Score':<25} | {get_fmt(m_best, 'Global_open_R2'):<18} | {get_fmt(m_final, 'Global_open_R2'):<18}")

        # --- 4. Global Trajectory Metrics Closed ---
        print("-" * 85)
        print(f"{'Global Closed Traj MSE':<25} | {get_fmt(m_best, 'Global_closed_MSE'):<18} | {get_fmt(m_final, 'Global_closed_MSE'):<18}")
        print(f"{'Global Closed ADE (m)':<25} | {get_fmt(m_best, 'Global_closed_ADE'):<18} | {get_fmt(m_final, 'Global_closed_ADE'):<18}")
        print(f"{'Global Closed FDE (m)':<25} | {get_fmt(m_best, 'Global_closed_FDE'):<18} | {get_fmt(m_final, 'Global_closed_FDE'):<18}")
        print(f"{'Global Closed R2 Score':<25} | {get_fmt(m_best, 'Global_closed_R2'):<18} | {get_fmt(m_final, 'Global_closed_R2'):<18}")


    else:
        print("Metric data missing in file.")

    print("\n--- Training Stats ---")
    train_hist = data.get('train_history', [])
    print(f"Total Epochs     : {len(train_hist)}")
    if train_hist:
        print(f"Final Train Loss : {train_hist[-1]:.6f}")
    
    val_hist = data.get('val_history', [])
    if val_hist:
        print(f"Final Val Loss   : {val_hist[-1]:.6f}")
        
    print("="*85 + "\n")
    
    return data



# --- GLOBAL STYLE CONFIGURATION ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Cambria"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Cambria",
    "mathtext.it": "Cambria:italic",
    "mathtext.bf": "Cambria:bold",
    "axes.unicode_minus": False,
    "font.size": 14,
    "axes.linewidth": 1.5,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
})

font_name = 'Cambria'
title_style = {'fontname': font_name, 'fontweight': 'bold', 'fontsize': 22}   
subtitle_style = {'fontname': font_name, 'fontsize': 16, 'fontweight': 'bold'} 
label_style = {'fontname': font_name, 'fontsize': 14}                          
legend_prop = fm.FontProperties(family='Cambria', style='italic', size=12)     

def _force_ticks_font(ax):
    """Helper to enforce Cambria and size on tick labels"""
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname(font_name)
        label.set_fontsize(12)

# ==========================================
# PLOT 0: CONVERGENCE (Updated: Readable Right Axis)
# ==========================================
def plot_convergence(args, results, filename=None):
    train_loss = results['train_history']
    val_loss = results['val_history']
    
    # Consistent Size (12, 8)
    fig, ax1 = plt.subplots(figsize=(12, 8))
    iterations = range(1, len(train_loss) + 1)
    
    ax1.set_xlabel(r"$\mathit{Iterations}$", **label_style)
    ax1.set_yscale('log')
    # Thicker grid
    ax1.grid(True, which="both", ls="--", alpha=0.5, linewidth=1.0)
    
    use_dual_axis = (args.reconstruct_train != args.reconstruct_val)
    
    # Colors (Standard high contrast if 'colors' not avail)
    c_train = '#1f77b4' # Tab:Blue equivalent
    c_val = '#ff7f0e'   # Tab:Orange equivalent
    
    if use_dual_axis:
        
        # Plot Train (Left Axis)
        ylabel_train = r"$\mathit{Train\ MSE\ (Reconstructed)}$" if args.reconstruct_train else r"$\mathit{Train\ MSE\ (Normalized)}$"
        ax1.set_ylabel(ylabel_train, color=c_train, **label_style)
        ax1.plot(iterations, train_loss, color=c_train, alpha=0.6, linewidth=2.0, label='Train Loss')
        ax1.tick_params(axis='y', labelcolor=c_train)
        t_min, t_max = min(train_loss), max(train_loss)
        ax1.set_ylim([t_min * 0.5, t_max * 2.0])

        # Plot Val (Right Axis)
        ax2 = ax1.twinx()
        ylabel_val = r"$\mathit{Val\ MSE\ (Reconstructed)}$" if args.reconstruct_val else r"$\mathit{Val\ MSE\ (Normalized)}$"
        ax2.set_ylabel(ylabel_val, color=c_val, **label_style)
        ax2.plot(iterations, val_loss, color=c_val, linewidth=2.0, label='Val Loss')
        ax2.tick_params(axis='y', labelcolor=c_val)
        ax2.set_yscale('log')
        v_min, v_max = min(val_loss), max(val_loss)
        ax2.set_ylim([v_min * 0.5, v_max * 2.0])

        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', prop=legend_prop)
        
        title_suffix = "(Dual Axis)"
        _force_ticks_font(ax2)

    else:
        ylabel = r"$\mathit{MSE\ Loss\ (Reconstructed)}$" if args.reconstruct_train else r"$\mathit{MSE\ Loss\ (Normalized)}$"
        ax1.set_ylabel(ylabel, **label_style)
        ax1.plot(iterations, train_loss, color=c_train, alpha=0.6, linewidth=2.0, label='Train Loss')
        ax1.plot(iterations, val_loss, color=c_val, linewidth=3.0, label='Val Loss')
        ax1.legend(loc='upper right', prop=legend_prop) 
        all_data = train_loss + val_loss
        g_min, g_max = min(all_data), max(all_data)
        ax1.set_ylim([g_min * 0.5, g_max * 2.0])      
        title_suffix = "(Single Axis)"

    _force_ticks_font(ax1)
    
    # Standard Title with Pad
    plt.title(f"Convergence Plot: {args.optimizer.upper()} {title_suffix}", pad=20, **title_style)
    
    fig.tight_layout()
    
    if filename and args.save_plot: 
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved to {filename}")
    if args.show_plot: plt.show()
    plt.close()

# ==========================================
# PLOT 1: LOCAL BRANCHES (Dynamics)
# ==========================================
def plot_kinematics_branches(args, data, step_interval=20, filename=None):
    """
    Plots short horizon predictions branching off the true path for all 4 targets.
    Replaces: plot_horizon_branches
    """
    true_path = data['true_backbone'] # (N, 4)
    pred_path = data['local']['pred_path'] # (N, H, 4)
    time_steps = np.arange(len(true_path))

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.25)

    targets = [
        {"name": "Surge Velocity", "unit": "m/s", "idx": 0},
        {"name": "Sway Velocity",  "unit": "m/s", "idx": 1},
        {"name": "Yaw Rate",       "unit": "rad/s", "idx": 2},
        {"name": "Yaw Angle",      "unit": "rad", "idx": 3}
    ]
    
    axes = []
    
    # 1. Setup Axes and Plot Background Truth
    for i, target in enumerate(targets):
        row, col = i // 2, i % 2
        ax = fig.add_subplot(gs[row, col])
        
        # Plot continuous True Path in background
        ax.plot(time_steps, true_path[:, target['idx']], 'k-', linewidth=1.5, alpha=0.3, label='True Path')
        
        ax.set_title(target['name'], **subtitle_style)
        ax.set_ylabel(rf"$\mathit{{{target['name'].split()[0]}}}$ [{target['unit']}]", **label_style)
        if row == 1: ax.set_xlabel(r"$\mathit{Time\ Step}$", **label_style)
        ax.grid(True, linestyle='--', alpha=0.5, linewidth=1.0)
        axes.append(ax)

    # 2. Plot Branches
    num_samples = pred_path.shape[0]
    color_branch = colors[1] if 'colors' in globals() and len(colors) > 1 else '#1f77b4'

    for i in range(0, num_samples, step_interval):
        # Create time indices for this specific branch
        t_indices = np.arange(i, i + args.horizon + 1)
        if t_indices[-1] >= len(time_steps): continue
        
        # Get the branch data (Start at true[i], then append preds)
        # We need to do this for all 4 dimensions
        curr_true = true_path[i].reshape(1, 4)
        curr_pred = pred_path[i] # (H, 4)
        branch_data = np.vstack([curr_true, curr_pred]) # (H+1, 4)

        # Plot on all 4 subplots
        for idx, ax in enumerate(axes):
            # Only label the first branch to avoid legend spam
            lbl = f'Pred Horizon (H={args.horizon})' if i == 0 else ""
            ax.plot(t_indices, branch_data[:, idx], color=color_branch, 
                    linestyle='-', linewidth=1.5, alpha=0.8, label=lbl)

    # Add legend to first plot
    axes[0].legend(prop=legend_prop if 'legend_prop' in globals() else None)
    for ax in axes: 
        if '_force_ticks_font' in globals(): _force_ticks_font(ax)

    fig.suptitle("Local Horizon Branches (Kinematics)", y=0.96, **title_style)
    
    if args.save_plot and filename:
        plt.savefig(filename + "_local_branches.png", dpi=300, bbox_inches='tight')
    if args.show_plot: plt.show()
    plt.close()
# ==========================================
# PLOT 2: GLOBAL DRIFT (Trajectory)
# ==========================================
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ==========================================
# PLOT 2: KINEMATICS VS TIME (4 Targets)
# ==========================================
def plot_kinematics_time_series(args, data, loop='closed', horizon_step=1, filename=None):
    """
    Plots Surge, Sway, Yaw Rate, and Yaw Angle vs Time in a 2x2 grid.
    """
    # 1. Extract Data
    # true_backbone shape expected: (N, 4) -> [Surge, Sway, Yaw Rate, Yaw Angle]
    true_data = data['true_backbone'] 
    
    # pred_path shape expected: (N, Horizon, 4)
    # We extract the specific horizon step 'k' to get a continuous trajectory (N, 4)
    raw_pred = data['global'][loop]['pred_path']
    
    # Extract the k-th step from the horizon predictions to build a continuous line
    k = horizon_step - 1
    if k < raw_pred.shape[1]:
        # Shift time to align with the horizon delay
        pred_seq = raw_pred[:-k, k, :] if k > 0 else raw_pred[:, k, :]
        true_seq = true_data[k:] if k > 0 else true_data
    else:
        print(f"Warning: Horizon step {horizon_step} is out of bounds.")
        return

    # Ensure lengths match exactly for plotting
    min_len = min(len(true_seq), len(pred_seq))
    true_seq = true_seq[:min_len]
    pred_seq = pred_seq[:min_len]
    time_steps = np.arange(min_len)

    # 2. Setup Figure
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.25)
    
    targets = [
        {"name": "Surge Velocity", "unit": "m/s", "idx": 0},
        {"name": "Sway Velocity",  "unit": "m/s", "idx": 1},
        {"name": "Yaw Rate",       "unit": "rad/s", "idx": 2},
        {"name": "Yaw Angle",      "unit": "rad", "idx": 3}
    ]

    # 3. Loop through the 4 targets and plot
    axes = []
    for i, target in enumerate(targets):
        row = i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col])
        
        # Plot True
        ax.plot(time_steps, true_seq[:, target['idx']], 'k-', linewidth=1.5, alpha=0.5, label='True')
        
        # Plot Pred
        color = '#D62728' # Red for prediction (or use your global colors)
        ax.plot(time_steps, pred_seq[:, target['idx']], '--', color=color, linewidth=2.0, alpha=0.9, label=f'Pred (k={horizon_step})')

        # Styling
        ax.set_title(target['name'], **subtitle_style)
        ax.set_ylabel(rf"$\mathit{{{target['name'].split()[0]}}}$ [{target['unit']}]", **label_style)
        
        # Only show X label on bottom rows
        if row == 1:
            ax.set_xlabel(r"$\mathit{Time\ Step}$", **label_style)
        
        ax.grid(True, linestyle='--', alpha=0.5, linewidth=1.0)
        
        # Force your custom tick font
        if '_force_ticks_font' in globals(): 
            _force_ticks_font(ax)
            
        axes.append(ax)

    # Add single legend to the first plot or outside
    axes[0].legend(prop=legend_prop if 'legend_prop' in globals() else None, loc='best')

    # 4. Final Layout
    header_title = f"Kinematics Analysis ({loop.capitalize()} Loop - Horizon Step {horizon_step})"
    fig.suptitle(header_title, y=0.96, **title_style)
    
    if args.save_plot and filename:
        save_path = filename + f"_kinematics_{loop}_k{horizon_step}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    if args.show_plot:
        plt.show()
    
    plt.close()

# ==========================================
# PLOT 3: ERROR ANALYSIS
# ==========================================
def plot_kinematics_errors(args, data, mode='global', loop='closed', horizon_mode='mean', filename=None):
    """
    Top: Aggregate Error (Norm of the 4D error vector).
    Bottom: Individual Error for each of the 4 targets over time.
    Replaces: plot_errors_and_position_time
    """
    if mode == 'local': pred_obj = data[mode]
    else: pred_obj = data[mode][loop]
    
    # Shapes: (N, H, 4)
    pred_path = pred_obj['pred_path'] 
    true_path = data['true_path'] # Assuming this is the expanded horizon version
    
    # Ensure matching lengths
    num_points = min(pred_path.shape[0], true_path.shape[0])
    pred_path = pred_path[:num_points]
    true_path = true_path[:num_points]
    time_steps = np.arange(num_points)

    # Calculate Errors
    # 1. Component-wise error: (N, H, 4) -> abs diff
    error_tensor = np.abs(true_path - pred_path)
    
    # 2. Norm error (Euclidean dist in 4D): (N, H)
    norm_error = np.linalg.norm(true_path - pred_path, axis=2)

    # Select Horizon Step (Mean vs Max vs Specific)
    if horizon_mode == 'mean':
        y_norm_err = np.mean(norm_error, axis=1) # (N,)
        y_comp_err = np.mean(error_tensor, axis=1) # (N, 4)
        label_str = "Mean over Horizon"
    elif horizon_mode == 'max':
        y_norm_err = np.max(norm_error, axis=1)
        y_comp_err = np.max(error_tensor, axis=1)
        label_str = "Max over Horizon"
    elif isinstance(horizon_mode, int):
        k = horizon_mode - 1
        y_norm_err = norm_error[:, k]
        y_comp_err = error_tensor[:, k, :]
        label_str = f"Horizon Step {horizon_mode}"

    # Plotting
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)

    # A. Top Plot: Aggregate Norm Error
    ax_top = plt.subplot(gs[0])
    
    # Plot Accumulated Error
    acc_error = np.cumsum(y_norm_err)
    c_acc, c_net = '#D62728', '#1F77B4'
    
    ax_top.plot(time_steps, acc_error, color=c_acc, linewidth=2.5, label='Accumulated Norm Error')
    ax_top.set_ylabel(r"$\mathit{Accumulated\ Error}$", color=c_acc, **label_style)
    ax_top.tick_params(axis='y', labelcolor=c_acc)
    
    # Twin axis for Net Error
    ax_top_r = ax_top.twinx()
    ax_top_r.plot(time_steps, y_norm_err, color=c_net, alpha=0.6, linestyle='--', linewidth=1.5, label='Inst. Norm Error')
    ax_top_r.set_ylabel(r"$\mathit{Instant\ Error}$", color=c_net, **label_style)
    ax_top_r.tick_params(axis='y', labelcolor=c_net)
    
    ax_top.set_title(f"Aggregate Error ({label_str})", **title_style)
    ax_top.grid(True, linestyle=':', alpha=0.6)
    
    # B. Bottom Plot: Error Per Component
    ax_bot = plt.subplot(gs[1], sharex=ax_top)
    
    comp_labels = ["Surge Err", "Sway Err", "YawRate Err", "YawAng Err"]
    comp_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd'] # Standard tab10 colors
    
    for i in range(4):
        ax_bot.plot(time_steps, y_comp_err[:, i], color=comp_colors[i], 
                    linewidth=1.5, alpha=0.8, label=comp_labels[i])
        
    ax_bot.set_title("Error by Component", **subtitle_style)
    ax_bot.set_ylabel(r"$\mathit{Absolute\ Error}$", **label_style)
    ax_bot.set_xlabel(r"$\mathit{Time\ Step}$", **label_style)
    ax_bot.grid(True, linestyle='--', alpha=0.5)
    ax_bot.legend(loc='upper right', prop=legend_prop if 'legend_prop' in globals() else None)

    for ax in [ax_top, ax_top_r, ax_bot]: 
        if '_force_ticks_font' in globals(): _force_ticks_font(ax)

    if args.save_plot and filename:
        plt.savefig(filename + f"_errors_{mode}_{loop}.png", dpi=300, bbox_inches='tight')
    if args.show_plot: plt.show()
    plt.close()
# ==========================================
# PLOT 4: BOXPLOTS
# ==========================================
def plot_kinematics_boxplots(args, data, mode='global', loop='closed', filename=None):
    """
    Boxplot of the error norm (4D) at each horizon step.
    Replaces: plot_horizon_euclidean_boxplots
    """
    if mode == 'local': pred_obj = data[mode]
    else: pred_obj = data[mode][loop]
    
    pred_path = pred_obj['pred_path']
    true_path = data['true_path']
    
    num_samples = min(pred_path.shape[0], true_path.shape[0])
    
    # Calculate Norm of error vector at each step (N, H)
    step_errors = np.linalg.norm(true_path[:num_samples] - pred_path[:num_samples], axis=2)
    
    horizon_steps = step_errors.shape[1]
    plot_data = [step_errors[:, k] for k in range(horizon_steps)]
    step_means = np.mean(step_errors, axis=0)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    box = ax.boxplot(plot_data, patch_artist=True, showfliers=False, widths=0.6,
                     medianprops=dict(linewidth=2.5, color='#000080'))
    
    c_face, c_edge = '#ADD8E6', '#1F77B4'
    for patch in box['boxes']:
        patch.set_facecolor(c_face)
        patch.set_edgecolor(c_edge)
        patch.set_alpha(0.7)

    x_pos = np.arange(1, horizon_steps + 1)
    ax.plot(x_pos, step_means, marker='D', color='#D62728', linestyle='None', markersize=8, label='Mean Error')
    
    ax.set_title(f"Horizon Error Distribution (State Norm) - {mode.capitalize()}", **title_style)
    ax.set_xlabel(r"$\mathit{Horizon\ Step}$", **label_style)
    ax.set_ylabel(r"$\mathit{Error\ Norm}$", **label_style)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    if '_force_ticks_font' in globals(): _force_ticks_font(ax)
    ax.legend(prop=legend_prop if 'legend_prop' in globals() else None)

    if args.save_plot and filename:
        plt.savefig(filename + f"_boxplots_{mode}.png", dpi=300, bbox_inches='tight')
    if args.show_plot: plt.show()
    plt.close()