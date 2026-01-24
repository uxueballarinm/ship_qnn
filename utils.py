import os
import pickle
import argparse
from copy import deepcopy

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

# ANSI Color Codes for Terminal Output
C_RED = '\033[91m'
C_YELLOW = '\033[93m'
C_GREEN = '\033[92m'
C_BLUE = '\033[94m'
C_RESET = '\033[0m'

colors = ['#E60000', '#FF8C00', '#C71585', '#008080', '#1E90FF']
#"Position X", "Position Y",
full_feature_set = [ "Surge Velocity", "Sway Velocity", "Yaw Rate", "Yaw Angle", "Speed U", "Rudder Angle (deg)", "Rudder Angle (rad)"]#, "OOD Label"]


# ==============================================================================
# 1. DATA & UTILITY FUNCTIONS
# ==============================================================================

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')


def map_names(feature_list, reverse=False):
    """Maps between short codes (wv) and full names (Surge Velocity)."""
    # Central Dictionary
    code_to_name = {
        "wv":"Surge Velocity", "sv":"Sway Velocity", 
        "yr":"Yaw Rate", "ya":"Yaw Angle",
        "vu": "Speed U", "radeg": "Rudder Angle (deg)",
        "rarad": "Rudder Angle (rad)", "OOD": "OOD Label",
        "dwv":"delta Surge Velocity", "dsv":"delta Sway Velocity",
        "dyr":"delta Yaw Rate", "dya":"delta Yaw Angle",
        # Ansatz
        "effsu2": "efficientsu2", "ugates": "ugates", "realamplitudes": "realamp",
        # Entanglement
        "lin": "linear", "rev": "reverse_linear", "circ": "circular", "full": "full", "pair": "pairwise", "sca": "sca"
    }

    if reverse:
        name_to_code = {v: k for k, v in code_to_name.items()}
        return [name_to_code.get(f, f) for f in feature_list]
    
    else: 
        columns = []
        for code in feature_list:
            if code in code_to_name:columns.append(code_to_name[code])
            elif code in code_to_name.values():columns.append(code) 
            else: raise ValueError(f"Unknown feature code: '{code}'. Available: {list(code_to_name.keys())}")
        return columns
def get_seqs(df, feature_columns_used, prediction_columns_used):
    return df[feature_columns_used].to_numpy(), df[prediction_columns_used].to_numpy()

def get_fold_indices(total_length, num_folds=4):
    fold_size = math.ceil(total_length / num_folds)
    split_indices = [0]
    for i in range(1, num_folds):
        next_idx = min(fold_size * i, total_length)
        split_indices.append(next_idx)
    if split_indices[-1] != total_length: # Ensure the last index is exactly the total length
        split_indices.append(total_length)
    return split_indices

def make_sliding_window_ycustom_folds(x, y, window_size, horizon_size, num_folds=4):

    split_indices = get_fold_indices(len(x), num_folds)
    
    x_data_folds, y_data_folds = [], []

    for k in range(num_folds):
        fold_x, fold_y = [], []
        
        start_idx, end_idx = split_indices[k], split_indices[k+1]
        
        for i in range(start_idx, end_idx):
            if i + window_size + horizon_size <= end_idx:
                fold_x.append(x[i : i + window_size])
                fold_y.append(y[i + window_size : i + window_size + horizon_size])
                
        x_data_folds.append(np.array(fold_x))
        y_data_folds.append(np.array(fold_y))
        
    return x_data_folds, y_data_folds

# ==============================================================================
# 2. CIRCUIT CONSTRUCTION
# ==============================================================================
def _parse_feature_map(map_input, selected_features):
    """Parses mixed tokens (int strings, feature codes) into integers."""
    if map_input is None: return None
    indices = []
    feat_to_idx = {name: i for i, name in enumerate(selected_features)}
    for item in map_input:
        try: # Try treating as integer (for indices or -1)
            val = int(item)
            indices.append(val)
        except ValueError: # Treat as Feature Code (e.g. 'px')
            full_name_list = map_names([item])
            if not full_name_list: raise ValueError(f"Unknown code: {item}")
            full_name = full_name_list[0]
            if full_name in feat_to_idx: indices.append(feat_to_idx[full_name])
            else: raise ValueError(f"Feature '{item}' ({full_name}) in map but NOT in selected features: {selected_features}")
    return indices
def _validate_chunk_completeness(chunk, num_features, layer_idx=None):
    """Validates that a layer contains exactly one instance of every feature."""
    valid = [x for x in chunk if x != -1]
    context = f"Layer {layer_idx}" if layer_idx is not None else "Template"
    
    if len(valid) != len(set(valid)):
        raise ValueError(f"[ERROR] [Map] Found duplicate features in {context}. Segment: {chunk}")
    if set(valid) != set(range(num_features)):
        raise ValueError(f"[ERROR] [Map] Missing or extra features in {context}. Segment: {chunk}")
    
def _load_and_validate_map(args, config):
    """Main processor for feature map parsing and validation."""
    reorder_active = getattr(args, 'reorder', True)
    num_padding = config["total_slots"] - config["num_features"]
    canonical_map = np.concatenate([np.arange(config["num_features"]), np.full(num_padding, -1)]).astype(int)
    raw_map = getattr(args, 'map', None)
    if reorder_active or raw_map is None:
        if raw_map is not None:
            print(f"{C_YELLOW}WARNING: Feature 'reorder' is active --> Ignoring custom input 'map'.{C_RESET}")
        return canonical_map, None
    flat_indices = _parse_feature_map(raw_map, args.features)
    n_slots, n_reps = config["total_slots"], args.reps
    
    if len(flat_indices) == n_slots:
        _validate_chunk_completeness(flat_indices, config["num_features"])
        return np.array(flat_indices, dtype=int), None
    elif len(flat_indices) == n_slots * n_reps:
        matrix = np.array(flat_indices, dtype=int).reshape(n_reps, n_slots)
        for i in range(n_reps): _validate_chunk_completeness(matrix[i], config["num_features"], layer_idx=i)
        return None, matrix
    else:
        raise ValueError(f"[ERROR] [Map] Invalid map length ({len(flat_indices)}). Expected {n_slots} or {n_slots * n_reps}.")
def _get_encoding_config(args):
    """Calculates circuit dimensions."""
    num_features = len(args.features)
    num_ugates = math.ceil(num_features / 3)
    total_slots = num_ugates * 3
    if args.encoding == 'compact':
        qubits_per_step, num_qubits, sub_layers = 1, args.window_size, 1
    elif args.encoding == 'parallel':
        qubits_per_step, num_qubits, sub_layers = num_ugates, args.window_size * num_ugates, 1
    elif args.encoding == 'serial':
        qubits_per_step, num_qubits, sub_layers = 1, args.window_size, num_ugates
    
    return {
        "num_features": num_features,
        "num_ugates": num_ugates,
        "total_slots": total_slots,
        "strategy": args.encoding,
        "num_qubits": num_qubits,
        "total_physical_layers": args.reps * sub_layers,
        "qubits_per_step": qubits_per_step,
        "weights_per_layer": 0
    }
def _get_params_for_gates(chunk_idx, num_features, input_params, base_idx, rep_indices):
    """Fetches parameters for a U-gate, handling padding."""
    p = []
    for k in range(3):
        slot_idx = chunk_idx * 3 + k # Calculate exactly which slot in the layer map we are accessing
       
        feat_idx = rep_indices[slot_idx]  # Get the feature index mapped to this slot
        if feat_idx != -1: p.append(input_params[base_idx + feat_idx]) # Valid feature: Read from input parameters
        else: p.append(0.0) # Sentinel -1: Padding/Empty slot -> 0.0 angle
    return p

def _apply_entanglement(qc, num_qubits, strategy='circular', layer_index=0):
    """Entanglement strategies."""
    if num_qubits < 2: return
    if strategy == 'linear':
        for i in range(num_qubits - 1): qc.cx(i, i + 1)
    elif strategy == 'reverse_linear':
        for i in range(num_qubits - 1, 0, -1): qc.cx(i, i - 1)
    elif strategy == 'circular':
        for i in range(num_qubits): qc.cx(i, (i + 1) % num_qubits) 
    elif strategy == 'full':
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits): qc.cx(i, j)     
    elif strategy == 'pairwise':  
        for i in range(0, num_qubits - 1, 2): qc.cx(i, i + 1) # Layer 1: Even pairs (0-1, 2-3...)
        for i in range(1, num_qubits - 1, 2): qc.cx(i, i + 1)# Layer 2: Odd pairs (1-2, 3-4...)
    elif strategy == 'sca': 
        shift = layer_index % num_qubits # Connect i to i+1, but shifted by the layer index
        for i in range(num_qubits): qc.cx((i + shift) % num_qubits, (i + shift + 1) % num_qubits)
    else: 
        raise ValueError(f"[ERROR] [Circuit] Unknown entanglement strategy: '{strategy}'")

def _append_ansatz_and_entangle(qc, args, weight_params, weight_idx, ansatz_obj, weights_per_layer, layer_idx):
    """Adds entanglement and trainable ansatz."""
    _apply_entanglement(qc, qc.num_qubits, strategy=args.entangle, layer_index=layer_idx)

    # 2. Ansatz (Trainable Weights)
    if args.ansatz == 'ugates':
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

# --- Block Builders ---
def _build_compact_block(qc, args, config, input_params, weight_params, weight_idx, rep_indices, ansatz_obj, current_layer):

    for t in range(args.window_size):
        base_idx = t * config["num_features"]
        for chunk_idx in range(config["num_ugates"]):
            p = _get_params_for_gates(chunk_idx, config["num_features"], input_params, base_idx, rep_indices)
            qc.u(p[0], p[1], p[2], t) 

    return _append_ansatz_and_entangle(qc, args, weight_params, weight_idx, ansatz_obj, config["weights_per_layer"], current_layer), current_layer + 1

def _build_parallel_block(qc, args, config, input_params, weight_params, weight_idx, rep_indices, ansatz_obj, current_layer):
    
    for t in range(args.window_size):
        base_idx = t * config["num_features"]
        for chunk_idx in range(config["num_ugates"]):
            p = _get_params_for_gates(chunk_idx, config["num_features"], input_params, base_idx, rep_indices)
            target_qubit = (t * config["qubits_per_step"]) + chunk_idx
            qc.u(p[0], p[1], p[2], target_qubit)
    return _append_ansatz_and_entangle(qc, args, weight_params, weight_idx, ansatz_obj, config["weights_per_layer"], current_layer), current_layer + 1

# --- STRATEGY 3: SERIAL (Deeper Circuit) ---
def _build_serial_block(qc, args, config, input_params, weight_params, weight_idx, rep_indices, ansatz_obj, current_layer):
    for s in range(config["num_ugates"]):
        for t in range(args.window_size):
            base_idx = t * config["num_features"]
            p = _get_params_for_gates(s, config["num_features"], input_params, base_idx, rep_indices)
            qc.u(p[0], p[1], p[2], t)
        weight_idx = _append_ansatz_and_entangle(qc, args, weight_params, weight_idx, ansatz_obj, config["weights_per_layer"], current_layer)
        current_layer += 1
    return weight_idx, current_layer

def create_multivariate_circuit(args, barriers=False): #TODO: Check if barriers have any effect

    # 1. Setup
    config = _get_encoding_config(args)
    
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
    # Map processing
    rep_indices, per_layer_orders = _load_and_validate_map(args, config)
    full_map_history = []
    for r in range(args.reps):
        # Select Indices
        if per_layer_orders is not None: current_indices = per_layer_orders[r]
        else: current_indices = rep_indices
        # 2. Shuffle
        if args.reorder: current_indices = rng.permutation(current_indices)
        
        # 3. Record
        full_map_history.extend(current_indices.tolist())
        
        # 4. Build
        if config["strategy"] == 'compact':
            weight_idx, current_physical_layer = _build_compact_block(
                qc, args, config, input_params, weight_params, weight_idx, current_indices, ansatz_obj, current_physical_layer
            )
        elif config["strategy"] == 'serial':
            weight_idx, current_physical_layer = _build_serial_block(
                qc, args, config, input_params, weight_params, weight_idx, current_indices, ansatz_obj, current_physical_layer
            )
        elif config["strategy"] == 'parallel':
            weight_idx, current_physical_layer = _build_parallel_block(
                qc, args, config, input_params, weight_params, weight_idx, current_indices, ansatz_obj, current_physical_layer
            )
        
        if barriers: qc.barrier()
        if per_layer_orders is None: rep_indices = current_indices
    args.map = full_map_history

    return qc, input_params, weight_params


# ==============================================================================
# 4. MODEL DEFINITIONS & TRAINING
# ==============================================================================

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
        print(f"[Model] Qubits: {self.input_dim} | Params: {self.num_q_params} (Q) + {self.num_c_params} (C)")        
        if seed is not None: self.rng = np.random.default_rng(seed)

    def forward(self, x, params_flat):
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
    def __init__(self, input_size, hidden_size, num_layers, output_size = None, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        
        # LSTM Layer
        # input_shape: (Batch, Seq_Len, Features)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
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
        with torch.no_grad(): out = self.model(t_x)
        total_out = out.shape[1]
        horizon = total_out // self.num_targets
        
        return out.cpu().numpy().reshape(x.shape[0], horizon, self.num_targets)

# Add/Replace in utils.py

class MultiHeadQNN:
    """
    A Generic Multi-Head QNN Wrapper.
    - Manages N independent models (heads).
    - Splits the flat parameter vector into chunks for each head.
    - Splits the input tensor 'x' based on feature indices for each head.
    - Concatenates the outputs of all heads into one final tensor.
    """
    def __init__(self, models_list, input_indices_list):
        self.models = models_list
        self.input_groups = input_indices_list
        
        # Calculate parameter boundaries
        self.param_splits = []
        total = 0
        for m in self.models:
            self.param_splits.append(m.total_params)
            total += m.total_params
        self.total_params = total
        
        print(f"\n[MultiHead] Initialized with {len(self.models)} heads.")
        for i, (n_p, grp) in enumerate(zip(self.param_splits, self.input_groups)):
            print(f"  > Head {i+1}: {n_p} params | Input Indices: {grp}")

    def forward(self, x, params):
        outputs = []
        param_start = 0
        
        # Iterate over each head
        for model, input_idx, n_params in zip(self.models, self.input_groups, self.param_splits):
            # 1. Slice Parameters for this head
            p_head = params[param_start : param_start + n_params]
            param_start += n_params
            
            # 2. Slice Inputs (Batch, Window, Selected_Features)
            x_head = x[:, :, input_idx]
            
            # 3. Forward Pass
            outputs.append(model.forward(x_head, p_head))
            
        # 4. Concatenate all outputs along the feature dimension (last axis)
        return np.concatenate(outputs, axis=2)

    def initialize_parameters(self, strategy):
        # Initialize each head and concatenate
        params_list = [m.initialize_parameters(strategy) for m in self.models]
        return np.concatenate(params_list)


def _compute_loss(args, pred, target, reconstruct, scaler=None):
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
        if reconstruct and not scaler: print(f'{C_YELLOW}WARNING: Scaler missing --> Cannot reconstruct trajectory.{C_RESET}')
        return np.mean((pred - target) ** 2)

def train_model(args, model, x_train, y_train, x_val, y_val, scaler=None):

    best_val_loss = float('inf')
    best_params = None
    
    train_history, val_history = [], []
    optimizer_name = args.optimizer.upper()
    use_batching = (optimizer_name == 'SPSA')
    # Use default 32 if not specified in args
    batch_size = getattr(args, 'batch_size', 32)
    
    num_train_samples = x_train.shape[0]
    def objective_function(params):
        nonlocal best_val_loss, best_params
        if use_batching:
            indices = np.random.choice(num_train_samples, size=batch_size, replace=False)
            x_input, y_target = x_train[indices], y_train[indices]
        else:
            x_input,y_target = x_train, y_train
        preds = model.forward(x_input, params)
        train_mse = _compute_loss(args, preds, y_target, args.reconstruct_train, scaler)

        check_val = True 
        if use_batching and len(train_history) % 10 != 0:
            check_val = False
        if check_val:
            val_preds = model.forward(x_val, params)
            val_mse = _compute_loss(args, val_preds, y_val, args.reconstruct_val, scaler)
            if val_mse < best_val_loss:
                best_val_loss = val_mse
                best_params = np.copy(params) #NOTE: Usually final weights are better
        else:
            val_mse = val_history[-1] if val_history else train_mse
        train_history.append(train_mse); val_history.append(val_mse)

        log_interval = 100 if use_batching else 50
        if len(train_history) % log_interval == 0:
            print(f"  > Iter {len(train_history):4d} | Train MSE: {train_mse:.5f} | Val MSE: {val_mse:.5f}")
        return train_mse
   

    start_time = time.time()
    print(f"\n[Training] Starting {args.optimizer.upper()} optimization...")
    if use_batching:
        print(f"  > Mode: Mini-Batch (Size: {batch_size})")
    else:
        print(f"  > Mode: Full-Batch (Size: {num_train_samples})")
    initial_weights = model.initialize_parameters(args.initialization)

    if args.optimizer.upper() == 'COBYLA':
        opt = COBYLA(maxiter=args.maxiter, tol = None)
        res = opt.minimize(fun=objective_function, x0=initial_weights)
    elif args.optimizer.upper() == 'SPSA':
        opt = SPSA(maxiter=args.maxiter,learning_rate=args.learning_rate, perturbation=args.perturbation) 
        res = opt.minimize(fun=objective_function, x0=initial_weights)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported.")
    print(f"Training completed in {(time.time() - start_time) / 60:.2f} min.")
    # Fallback if best_params never updated (rare)
    if best_params is None: best_params = res.x

    return {
        "best_weights": best_params, "best_val_loss": best_val_loss, "final_weights": res.x,             
        "train_history": train_history, "val_history": val_history,         
    }


def train_classical_model(args, model, x_train, y_train, x_val, y_val, y_scaler = None, device='cpu'):
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
    train_loader = DataLoader(TensorDataset(t_x_train, t_y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(t_x_val, t_y_val), batch_size=batch_size, shuffle=False)

    # 2. Setup Training
    model = model.to(device)
    criterion = nn.MSELoss()
    if args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError("Unsupported optimizer")

    best_val_loss = float('inf')
    best_weights = None
    train_history, val_history = [], []
    
    # Early stopping config
    patience = getattr(args, 'patience', 20)
    counter = 0
    
    print(f"\n[Training] Starting Classical Optimization (Adam) on {device}...")
    start_time = time.time()
    for epoch in range(args.maxiter): # 'maxiter' acts as 'epochs' here
        model.train()
        epoch_loss = 0
        
        for x_batch, y_batch in train_loader:
            # Forward
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            
            # Backward
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            
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
                batch_loss = _compute_loss(args, v_p_np, v_y_np, args.reconstruct_val, y_scaler) 
                val_loss_accum += batch_loss * x_v.size(0)
                total_samples += x_v.size(0)
        
        avg_val_loss = val_loss_accum / total_samples
        val_history.append(avg_val_loss)

        # Logging
        if (epoch + 1) % 10 == 0:
            print(f"[Training] Epoch {epoch+1:4d} | Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f}")

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
    print(f"Classical training finished in {(time.time() - start_time) / 60:.2f} min.")
    if best_weights is not None: model.load_state_dict(best_weights)
    return {
        "best_weights": best_weights, "final_weights": model.state_dict(),"best_val_loss": best_val_loss,
        "train_history": train_history,"val_history": val_history,
    }

# ==============================================================================
# 5. RECURSIVE EVALUATION & PLOTTING (PUBLIC)
# ==============================================================================
def recursive_forward_pass(args, model, best_params, x_test, x_scaler, y_scaler):
    """
    Performs Recursive 'Fan' Prediction with Automatic Physics Logic.
    Handles 4 update modes based on Target vs. Feature names.
    """
    
    # 1. Setup
    num_samples = x_test.shape[0]; horizon = args.horizon        
    num_targets = len(args.targets)

    direct_updates = []
    physics_updates = []
    update_rules = {} 
    
    for t_idx, t_name in enumerate(args.targets):
        update_rules[t_idx] = []
        is_target_delta = "delta" in t_name.lower() or args.predict == 'delta'
        
        # --- LOGIC 1: Primary Update (Direct Match) ---
        if t_name in args.features:
            f_idx = args.features.index(t_name)
            # If target is Delta and Feature is Delta -> DIRECT and If target is Motion and Feature is Motion -> DIRECT
            update_rules[t_idx].append((f_idx, "DIRECT"))
            direct_updates.append(t_name)
        secondary_name = None
        mode = None
        
        if is_target_delta:
            # We have Delta, look for Motion to Integrate
            clean = t_name.replace("delta ", "").strip()
            if clean in args.features: 
                secondary_name = clean
                mode = "INTEGRATE"
                physics_updates.append(f"{clean} (Integrated)")
        else:
            # We have Motion, look for Delta to Differentiate
            delta_ver = f"delta {t_name}"
            if delta_ver in args.features: 
                secondary_name = delta_ver
                mode = "DIFF"
                physics_updates.append(f"{delta_ver} (Differentiated)")
        
        if secondary_name:
            f_idx_sec = args.features.index(secondary_name)
            update_rules[t_idx].append((f_idx_sec, mode))
    driven_feats = set()
    for rules in update_rules.values():
        for f_idx, _ in rules:
            driven_feats.add(f_idx)
    num_driven = len(update_rules)
    feat_names = args.features
    recursive_ratio = num_driven / len(feat_names)
    if num_driven == 0:
        print(f"  > {C_YELLOW}Fully Open-Loop (No recursion).{C_RESET}")
        preds_full = model.forward(x_test, best_params)
        return preds_full, 0.0
    print(f"  > Recursive Loop Active ({num_driven}/{len(feat_names)} features driven)")
    if direct_updates:
        print(f"    Direct Feedback:  {', '.join(direct_updates)}")
    if physics_updates:
        print(f"    Physics Feedback: {', '.join(physics_updates)}")
    # 2. Initialization
    recursive_preds = np.zeros((num_samples, horizon, num_targets))
    curr_window = x_test[0:1, :, :].copy()
    last_feat_real = x_scaler.inverse_transform(curr_window[:, -1, :])
    
    # Trackers for differentiation
    last_pred_values = np.zeros(num_targets) 
    
    # 3. Main Loop
    for i in range(num_samples):
        preds_full = model.forward(curr_window, best_params)
        recursive_preds[i] = preds_full[0]

        next_gt_idx = min(i + 1, num_samples - 1)
        next_gt_features_norm = x_test[next_gt_idx:next_gt_idx+1, -1, :] 
        
        # Start with Ground Truth
        next_input_real = x_scaler.inverse_transform(next_gt_features_norm)
        
        pred_step_norm = preds_full[:, 0, :]
        if y_scaler: pred_step_real = y_scaler.inverse_transform(pred_step_norm)
        else: pred_step_real = pred_step_norm

        # Apply Updates
        for t_idx, updates in update_rules.items():
            pred_val = pred_step_real[0, t_idx]
            
            for f_idx, mode in updates:
                if mode == "DIRECT":
                    next_input_real[0, f_idx] = pred_val
                elif mode == "INTEGRATE":
                    # New Pos = Old Pos + Pred Delta
                    old_val = last_feat_real[0, f_idx]
                    next_input_real[0, f_idx] = old_val + pred_val
                elif mode == "DIFF":
                    if i == 0: diff_val = pred_val 
                    else: diff_val = pred_val - last_pred_values[t_idx]
                    next_input_real[0, f_idx] = diff_val

        # Update trackers
        last_pred_values = pred_step_real[0]
        last_feat_real = next_input_real   
        
        new_row_norm = x_scaler.transform(next_input_real)
        curr_window = np.concatenate([curr_window[:, 1:, :], new_row_norm.reshape(1, 1, -1)], axis=1)

    return recursive_preds, recursive_ratio

# def recursive_forward_pass(args, model, best_params, x_test, x_scaler, y_scaler):
    
#     num_samples = x_test.shape[0]; horizon = args.horizon
#     feat_names = args.features 
#     def get_idx(name): return feat_names.index(name) if name in feat_names else None
#     idx_wv, idx_sv, idx_yr, idx_ya = get_idx('Surge Velocity'), get_idx('Sway Velocity'), get_idx('Yaw Rate'), get_idx('Yaw Angle')
#     idx_dwv, idx_dsv, idx_dyr, idx_dya = get_idx('delta Surge Velocity'), get_idx('delta Sway Velocity'), get_idx('delta Yaw Rate'), get_idx('delta Yaw Angle')

#     if args.predict == 'motion' and (idx_wv is None or idx_sv is None or idx_yr is None or idx_ya is None):
#         print(f"  > {C_YELLOW}WARNING: Model cannot predict absolute kinematics without input kinematics features.{C_RESET}")

#     direct_updates = []
#     physics_updates = []
#     if args.predict == 'delta':
#         if idx_dwv is not None: direct_updates.append('delta Surge Velocity')
#         if idx_dsv is not None: direct_updates.append('delta Sway Velocity')
#         if idx_dyr is not None: direct_updates.append('delta Yaw Rate')
#         if idx_dya is not None: direct_updates.append('delta Yaw Angle')
#         if idx_wv is not None: physics_updates.append('Surge Velocity(integrated from delta Surge Velocity)')
#         if idx_sv is not None: physics_updates.append('Sway Velocity(integrated from delta Sway Velocity)')
#         if idx_yr is not None: physics_updates.append('Yaw Rate(integrated from delta Yaw Rate)')
#         if idx_ya is not None: physics_updates.append('Yaw Angle(integrated from delta Yaw Angle)')
#     else: # motion
#         if idx_wv is not None: direct_updates.append('Surge Velocity')
#         if idx_sv is not None: direct_updates.append('Sway Velocity')
#         if idx_yr is not None: direct_updates.append('Yaw Rate')
#         if idx_ya is not None: direct_updates.append('Yaw Angle')
#         if idx_dwv is not None: physics_updates.append('delta Surge Velocity (differentiated from Surge Velocity)')
#         if idx_dsv is not None: physics_updates.append('delta Sway Velocity (differentiated from Sway Velocity)')
#         if idx_dyr is not None: physics_updates.append('delta Yaw Rate (differentiated from Yaw Rate)')
#         if idx_dya is not None: physics_updates.append('delta Yaw Angle (differentiated from Yaw Angle)')

#     num_driven = len(direct_updates) + len(physics_updates)
#     total_feats = len(feat_names)

#     if num_driven == 0:
#         print("  > Fully Open-Loop (No recursion).")
#         preds_full = model.forward(x_test, best_params)
#         return preds_full, 0.0

#     print(f"  > Recursive Loop Active ({num_driven}/{total_feats} features)")
#     if direct_updates:
#         print(f"    Direct Feedback: {', '.join(direct_updates)}")
#     if physics_updates:
#         print(f"    Physics Feedback: {', '.join(physics_updates)}")

#     recursive_ratio = num_driven / total_feats
    
#     curr_window = x_test[0:1, :, :] 
#     last_step_real = x_scaler.inverse_transform(curr_window[:, -1, :])
#     recursive_preds = np.zeros((num_samples, horizon, len(args.targets)))
#     last_wv, last_sv, last_yr, last_ya = 0.0, 0.0, 0.0, 0.0

#     for i in range(num_samples):
#         preds_full = model.forward(curr_window, best_params)
#         recursive_preds[i] = preds_full[0]
#         pred_step = preds_full[:, 0, :] 
        
#         next_gt_idx = min(i + 1, num_samples - 1)
#         next_gt_features_norm = x_test[next_gt_idx:next_gt_idx+1, -1, :] 

#         if y_scaler: pred_step_real = y_scaler.inverse_transform(pred_step)
#         else: pred_step_real = pred_step

#         next_gt_real = x_scaler.inverse_transform(next_gt_features_norm)
#         new_row_real = np.copy(next_gt_real)
        
#         if args.predict == 'delta':
#             dwv, dsv,dyr,dya = pred_step_real[0, 0], pred_step_real[0, 1], pred_step_real[0, 2], pred_step_real[0, 3]
#             if idx_dwv is not None: new_row_real[0, idx_dwv] = dwv
#             if idx_dsv is not None: new_row_real[0, idx_dsv] = dsv
#             if idx_dyr is not None: new_row_real[0, idx_dyr] = dyr
#             if idx_dya is not None: new_row_real[0, idx_dya] = dya
#             if idx_wv is not None: new_row_real[0, idx_wv] = last_step_real[0, idx_wv] + dwv
#             if idx_sv is not None: new_row_real[0, idx_sv] = last_step_real[0, idx_sv] + dsv
#             if idx_yr is not None: new_row_real[0, idx_yr] = last_step_real[0, idx_yr] + dyr
#             if idx_ya is not None: new_row_real[0, idx_ya] = last_step_real[0, idx_ya] + dya
#         else: 
#             wv, sv, yr, ya = pred_step_real[0, 0], pred_step_real[0, 1], pred_step_real[0, 2], pred_step_real[0, 3]
#             if idx_wv is not None: new_row_real[0, idx_wv] = wv
#             if idx_sv is not None: new_row_real[0, idx_sv] = sv
#             if idx_yr is not None: new_row_real[0, idx_yr] = yr
#             if idx_ya is not None: new_row_real[0, idx_ya] = ya
#             if idx_dwv is not None: 
#                 new_row_real[0, idx_dwv] = wv- last_wv
#                 last_wv = wv
#             if idx_dsv is not None:
#                 new_row_real[0, idx_dsv] = sv - last_sv 
#                 last_sv = sv
#             if idx_dyr is not None:
#                 new_row_real[0, idx_dyr] = yr - last_yr 
#                 last_yr = yr
#             if idx_dya is not None:
#                 new_row_real[0, idx_dya] = ya - last_ya 
#                 last_ya = ya

#         last_step_real = new_row_real
#         new_row_norm = x_scaler.transform(new_row_real)
#         curr_window = np.concatenate([curr_window[:, 1:, :], new_row_norm.reshape(1, 1, -1)], axis=1)

#     return recursive_preds, recursive_ratio
    

def evaluate_model(args, model, params, x_test, y_test, x_scaler, y_scaler):
    """Runs BOTH one-step (Teacher Forcing) and recursive (Dead Reckoning) evaluations."""

    results = {}
    orig_shape = y_test.shape # (N, H, T)
    num_targets = orig_shape[-1]
    target_names = ["Surge_Velocity", "Sway_Velocity", "Yaw_Rate", "Yaw_Angle"]
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

    for i, name in enumerate(target_names):
        results[f'{name}_Step_MSE'] = mean_squared_error(y_gt_real[..., i].flatten(), preds_real_step[..., i].flatten())
        results[f'{name}_Step_R2']  = r2_score(y_gt_real[..., i].flatten(), preds_real_step[..., i].flatten())

    if args.predict == 'delta':
        true_path_backbone = np.concatenate([np.zeros((1, num_targets)), np.cumsum(y_gt_real[:, 0, :], axis=0)])
        true_path = true_path_backbone[:-1, None, :] + np.cumsum(y_gt_real, axis=1)
        pred_path_local = true_path_backbone[:-1, None, :] + np.cumsum(preds_real_step, axis=1)
        pred_path_backbone = np.concatenate([np.zeros((1, num_targets)), np.cumsum(preds_real_step[:, 0, :], axis=0)])
        pred_path_global_open = pred_path_backbone[:-1, None, :] + np.cumsum(preds_real_step, axis=1)
    else:
        true_path_backbone = y_gt_real[:, 0, :]
        true_path = y_gt_real
        pred_path_backbone = preds_real_step[:, 0, :]
        pred_path_local = preds_real_step
        pred_path_global_open = preds_real_step

    results['Local_MSE'] = mean_squared_error(true_path.reshape(-1, num_targets), pred_path_local.reshape(-1, num_targets))
    norm_local_error = np.linalg.norm(true_path - pred_path_local, axis=2)
    results['Local_ADE'] = np.mean(norm_local_error)
    for i, name in enumerate(target_names):
        true_i = true_path[..., i].flatten()
        pred_i = pred_path_local[..., i].flatten()
        results[f'{name}_Local_MSE'] = mean_squared_error(true_i, pred_i)
        results[f'{name}_Local_ADE'] = np.mean(np.abs(true_path[..., i] - pred_path_local[..., i]))

    results['Global_open_MSE'] = mean_squared_error(true_path.reshape(-1, num_targets), pred_path_global_open.reshape(-1, num_targets))
    results['Global_open_R2'] = r2_score(true_path.reshape(-1, num_targets), pred_path_global_open.reshape(-1, num_targets))
    norm_global_error = np.linalg.norm(true_path - pred_path_global_open, axis=2)
    results['Global_open_ADE'] = np.mean(norm_global_error)    
    results['Global_open_FDE'] = np.mean(norm_global_error[:, -1])  
    results['Global_open_Max'] = np.max(norm_global_error)   

    for i, name in enumerate(target_names):
        true_i = true_path[..., i].flatten()
        pred_i = pred_path_global_open[..., i].flatten()
        
        # Calculate Abs Error for Max logic
        abs_err = np.abs(true_path[..., i] - pred_path_global_open[..., i])
        
        results[f'{name}_Global_open_MSE'] = mean_squared_error(true_i, pred_i)
        results[f'{name}_Global_open_R2']  = r2_score(true_i, pred_i)
        results[f'{name}_Global_open_ADE'] = np.mean(abs_err)    
        results[f'{name}_Global_open_FDE'] = np.mean(abs_err[:, -1])

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
    results['Global_closed_FDE'] = np.mean(norm_global_error[:, -1])
    results['Global_closed_Max'] = np.max(norm_global_error)  
    for i, name in enumerate(target_names):
        true_i = true_path[..., i].flatten()
        pred_i = pred_path_global[..., i].flatten()
        abs_err = np.abs(true_path[..., i] - pred_path_global[..., i])
        
        mse_i = mean_squared_error(true_i, pred_i)
        r2_i = r2_score(true_i, pred_i)
        max_i = np.max(abs_err)
        
        # Store individual metrics
        results[f'{name}_Global_closed_MSE'] = mse_i
        results[f'{name}_Global_closed_R2']  = r2_i
        results[f'{name}_Global_closed_ADE'] = np.mean(abs_err)    
        results[f'{name}_Global_closed_FDE'] = np.mean(abs_err[:, -1])  


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

# ==============================================================================
# SAVE AND PLOT RESULTS
# ==============================================================================
def save_experiment_results(args, train_results, best_eval, final_eval, scalers, qnn_dict, timestamp):
    
    for folder in ["models", "logs", "figures"]: os.makedirs(folder, exist_ok=True)
    
    # ==========================================================================
    # 1. ROBUST PARAMETER COUNTING (Prevents Crash on Multi-Head)
    # ==========================================================================
    total_params = len(train_results['final_weights'])
    
    # Handle qnn_dict being a list (MultiHead) or dict (Vanilla)
    num_q_params = 0
    if isinstance(qnn_dict, list):
        # Sum quantum params from all heads
        for head_dict in qnn_dict:
            if isinstance(head_dict, dict) and 'weight_params' in head_dict:
                num_q_params += len(head_dict['weight_params'])
    elif isinstance(qnn_dict, dict) and 'weight_params' in qnn_dict:
        # Vanilla case
        num_q_params = len(qnn_dict['weight_params'])
    
    num_c_params = total_params - num_q_params

    # ==========================================================================
    # 2. SMART CONFIG RESOLUTION
    # ==========================================================================
    final_ansatz = map_names([args.ansatz], reverse=True)[0]
    final_entangle = map_names([args.entangle], reverse=True)[0]
    final_reps = getattr(args, 'reps', 'N/A')
    final_encoding = getattr(args, 'encoding', 'N/A')
    final_map = getattr(args, 'map', 'N/A')
    final_features = map_names(args.features, reverse=True) 

    def resolve_multi_val(values_list):
        if not values_list: return 'N/A'
        if all(x == values_list[0] for x in values_list):
            return values_list[0] 
        return str(values_list)

    if getattr(args, 'model', '') == 'multihead' and hasattr(args, 'heads_config') and args.heads_config:
        list_ansatz = []
        list_entangle = []
        list_reps = []
        list_encoding = []
        list_map = []
        list_features = []

        for h in args.heads_config:
            short_a = map_names([h.get('ansatz', args.ansatz)], reverse=True)[0]
            short_e = map_names([h.get('entangle', args.entangle)], reverse=True)[0]
            list_ansatz.append(short_a)
            list_entangle.append(short_e)
            list_reps.append(h.get('reps', getattr(args, 'reps', 'N/A')))
            list_encoding.append(h.get('encoding', getattr(args, 'encoding', 'N/A')))
            list_map.append(h.get('map', getattr(args, 'map', 'N/A')))
            list_features.append(map_names(h.get('features', []), reverse=True))

        final_ansatz = resolve_multi_val(list_ansatz)
        final_entangle = resolve_multi_val(list_entangle)
        final_reps = resolve_multi_val(list_reps)
        final_encoding = resolve_multi_val(list_encoding)
        final_map = resolve_multi_val(list_map)
        final_features = resolve_multi_val(list_features)

    # ==========================================================================
    # 3. SAVE PICKLE
    # ==========================================================================
    def clean_filename_str(s):
        s = str(s).replace('[', '').replace(']', '').replace("'", "").replace('"', "")
        return s.replace(', ', '_').replace(',', '_')

    safe_ansatz = clean_filename_str(final_ansatz)
    safe_entangle = clean_filename_str(final_entangle)
    safe_reps = clean_filename_str(final_reps)

    model_filename = os.path.join("models", f"{timestamp}_{args.model}_f{len(args.features)}_w{args.window_size}_h{args.horizon}_{safe_ansatz}_{safe_entangle}_r{safe_reps}.pkl")
    
    save_payload = {
        "config": vars(args),
        "best_weights": train_results['best_weights'],
        "final_weights": train_results['final_weights'],
        "train_history": train_results['train_history'],
        "val_history": train_results['val_history'],
        "best_eval_metrics": best_eval['metrics'],
        "final_eval_metrics": final_eval['metrics'],
        "y_scaler": scalers[1], "x_scaler": scalers[0], "qnn_structure": qnn_dict
    }
    with open(model_filename, "wb") as f: pickle.dump(save_payload, f)

    # ==========================================================================
    # 4. PREPARE EXCEL DATA (Strict Column Order)
    # ==========================================================================
    excel_filename = os.path.join("logs", "experiments_summary.xlsx")
    m_final = final_eval['metrics']
    
    try:
        dt_temp = datetime.datetime.strptime(timestamp, "%m-%d_%H-%M-%S")
        dt_object = dt_temp.replace(year=2026)
    except ValueError:
        dt_object = timestamp 

    raw_data = {}
    
    # A. Add Args
    ignore_keys = ['select_features', 'drop_features', 'save_plot', 'show_plot', 'initialization', 'heads_config', 
                   'ansatz', 'entangle', 'reps', 'encoding', 'map', 'features']
    
    def clean_val(v):
        if isinstance(v, bool): return str(v).lower()
        if isinstance(v, list): return str(v)
        return v
    
    for key, value in vars(args).items():
        if key not in ignore_keys:
            raw_data[key] = clean_val(value)

    # B. Add Resolved Configs
    raw_data['features'] = str(final_features)
    raw_data['ansatz'] = str(final_ansatz)
    raw_data['entangle'] = str(final_entangle)
    raw_data['reps'] = str(final_reps)
    raw_data['encoding'] = str(final_encoding)
    raw_data['map'] = str(final_map)
    
    # C. Add Meta & Metrics
    raw_data['date'] = dt_object
    raw_data['model_id'] = os.path.basename(model_filename)
    raw_data['data_n'] = getattr(args, 'data_n', 'N/A')
    raw_data['data_dt'] = getattr(args, 'data_dt', 'N/A')
    raw_data['heads_config'] = str(getattr(args, 'heads_config', 'N/A'))

    metric_map = {
        "step MSE": m_final.get('Step_MSE'),
        "step R2": m_final.get('Step_R2'),
        "local MSE": m_final.get('Local_MSE'),
        "local ADE": m_final.get('Local_ADE'),
        "global open MSE": m_final.get('Global_open_MSE'),
        "global open ADE": m_final.get('Global_open_ADE'),
        "global open FDE": m_final.get('Global_open_FDE'),
        "global open R2": m_final.get('Global_open_R2'),
        "global closed MSE": m_final.get('Global_closed_MSE'),
        "global closed ADE": m_final.get('Global_closed_ADE'),
        "global closed FDE": m_final.get('Global_closed_FDE'),
        "global closed R2": m_final.get('Global_closed_R2'),
        "recursivity": m_final.get('Recursivity'),
        "final val loss": train_results['val_history'][-1] if train_results['val_history'] else None,
        "iterations": len(train_results['train_history']),
        "total params": total_params,
        "q params": num_q_params,
        "c params": num_c_params
    }
    raw_data.update(metric_map)

    # D. Add Per-Target Metrics
    target_names = ["Surge Velocity", "Sway Velocity", "Yaw Rate", "Yaw Angle"]
    metric_suffixes = ["Step MSE", "Step R2", "Local MSE", "Local ADE", 
                       "Global open MSE", "Global open R2", "Global open ADE", "Global open FDE",
                       "Global closed MSE", "Global closed R2", "Global closed ADE", "Global closed FDE"]

    for tgt_space in target_names:
        tgt_under = tgt_space.replace(" ", "_")
        for m_suffix in metric_suffixes:
            m_under = m_suffix.replace(" ", "_")
            key_in_metrics = f"{tgt_under}_{m_under}"
            col_name_excel = f"{tgt_space} {m_suffix}"
            if key_in_metrics in m_final:
                raw_data[col_name_excel] = m_final[key_in_metrics]

    # E. Build Final Column List (User Defined Order)
    final_column_order = [
        "date", "model_id", "run", "testing_fold", "data", "data_n", "data_dt", 
        "features", "targets", "window_size", "horizon", "predict", "norm", 
        "reconstruct_train", "reconstruct_val", "model", "encoding", "ansatz", 
        "entangle", "reps", "map", "reorder", "total params", "q params", 
        "c params", "optimizer", "maxiter", "iterations", "final val loss", 
        "tolerance", "batch_size", "learning_rate", "perturbation", "step MSE", "step R2", 
        "local MSE", "local ADE", "global open MSE", "global open ADE", 
        "global open FDE", "global open R2", "global closed MSE", 
        "global closed ADE", "global closed FDE", "global closed R2", 
        "recursivity"
    ]
    
    # Append per-target cols in the order of targets -> metrics
    for tgt in target_names:
        for m in metric_suffixes:
            final_column_order.append(f"{tgt} {m}")
            
    # Append heads_config at the end as requested
    final_column_order.append("heads_config")

    # Construct Row
    ordered_row = {}
    for col in final_column_order:
        ordered_row[col] = raw_data.get(col, None)
    
    # Safety: Add any extra keys that might exist but weren't in the list
    for k, v in raw_data.items():
        if k not in ordered_row:
            ordered_row[k] = v

    df_new = pd.DataFrame([ordered_row])

    def normalize_loaded_bools(val):
        if isinstance(val, bool): return str(val).lower()
        if isinstance(val, str):
            if val.upper() in ['TRUE', 'VERDADERO']: return 'true'
            if val.upper() in ['FALSE', 'FALSO']: return 'false'
        return val

    try:
        if os.path.exists(excel_filename):
            df_existing = pd.read_excel(excel_filename)
            df_existing = df_existing.map(normalize_loaded_bools)
            
            # Concatenate
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
            
            # Reorder columns to match your preferred order, putting new columns at end
            cols_existing = list(df_existing.columns)
            # We prioritize the User Order, then append whatever else exists
            full_order = final_column_order + [c for c in cols_existing if c not in final_column_order]
            # Filter out duplicates if any
            full_order = list(dict.fromkeys(full_order))
            
            df_final = df_final.reindex(columns=full_order)
            df_final.to_excel(excel_filename, index=False)
        else:
            df_new = df_new.reindex(columns=final_column_order)
            df_new.to_excel(excel_filename, index=False)
            
    except PermissionError:
        print(f"\n{C_RED}[ERROR] Excel file is open! Saving to CSV backup.{C_RESET}")
        df_new.to_csv(f"logs/backup_{timestamp}.csv", index=False)
    except Exception as e:
        print(f"{C_YELLOW}[Warning] Excel error: {e}. Saving to CSV backup.{C_RESET}")
        df_new.to_csv(f"logs/backup_{timestamp}.csv", index=False)

    log_filename = "logs/experiment_log.txt"
    log_entry = f"[{timestamp}] {args.model:<10} | F={len(args.features):<2} W={args.window_size:<2} H={args.horizon:<2} | Circuit: {str(final_encoding):<10} {str(final_ansatz):<15} {str(final_entangle):<14} reps={str(final_reps):<2} | MSE={m_final.get('Step_MSE', 0):.4f}\n"
    with open(log_filename, "a") as f: f.write(log_entry)
    
    print(f"\n[Logger] Model saved to {model_filename}")
    print(f"[Logger] Stats appended to {excel_filename}")
    return model_filename
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

    target_names = ["Surge_Velocity", "Sway_Velocity", "Yaw_Rate", "Yaw_Angle"]
    metric_types = [
        "Step_MSE", "Step_R2",
        "Local_MSE", "Local_ADE",
        # Added ADE and FDE here to match your evaluate_model updates
        "Global_open_MSE", "Global_open_R2", "Global_open_ADE", "Global_open_FDE",
        "Global_closed_MSE", "Global_closed_R2", "Global_closed_ADE", "Global_closed_FDE"
    ]
    per_target_cols = []
    
    for tgt in target_names:
        for m_type in metric_types:
            key_in_dict = f"{tgt}_{m_type}" # e.g. Surge_Velocity_Global_closed_RMSE
            
            if key_in_dict in m_final:
                # Clean up name for Excel (Surge Velocity Global closed RMSE)
                col_name = key_in_dict.replace("_", " ") 
                raw_data[col_name] = m_final[key_in_dict]
                per_target_cols.append(col_name)
        
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
    final_column_order = column_order + per_target_cols
    # Construct the final ordered dictionary
    ordered_row = {}
    
    # Add ordered keys first
    for col in final_column_order:
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
    Loads a saved experiment pickle file and prints a comprehensive summary,
    including a detailed per-feature metric table with Step, Local, Open, and Closed metrics.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    print(f"Loading results from: {filepath} ...")
    
    with open(filepath, "rb") as f:
        data = pickle.load(f)
        
    config = data.get('config', {})
    m_best = data.get('best_eval_metrics', {})
    m_final = data.get('final_eval_metrics', {}) # We report final weights metrics
    
    # Model type detection fallback
    if config.get('model') is None:
        if 'classical' in filepath.lower() or 'hidden_size' in config:
            config['model'] = 'classical_lstm'
        else:
            config['model'] = 'N/A'

    # --- Print Summary Header ---
    print("\n" + "="*120)
    print(f"EXPERIMENT SUMMARY")
    print("="*120)
    fname = os.path.basename(filepath)
    try:
        parts = fname.split('_')
        timestamp = f"{parts[0]}_{parts[1]}"
    except:
        timestamp = "Unknown"
        
    print(f"Timestamp: {timestamp}")
    
    # --- Configuration ---
    print("\n--- Configuration ---")
    common_keys = ['model', 'features', 'window_size', 'horizon', 'optimizer']
    quantum_keys = ['ansatz', 'encoding', 'reps', 'entangle']
    classical_keys = ['hidden_size', 'layers', 'learning_rate', 'batch_size', 'patience']

    model_type = config.get('model', 'unknown')
    keys_to_show = common_keys.copy()
    
    if 'lstm' in model_type.lower() or 'classical' in filepath.lower():
        keys_to_show.extend(classical_keys)
    else:
        keys_to_show.extend(quantum_keys)

    for k in keys_to_show:
        val = config.get(k, 'N/A')
        if isinstance(val, list) and k == 'features':
            val = f"{len(val)} features"
        elif isinstance(val, list):
            val = str(val)
        print(f"{k:<15}: {val}")
    
    # --- Performance Table (Aggregate) ---
    print("\n--- Aggregate Performance (Final Weights) ---")
    
    if m_final:
        def get_fmt(metrics, key):
            val = metrics.get(key)
            if val is None: return "N/A"
            return f"{val:.6f}" if isinstance(val, (int, float)) else str(val)

        # Aggregate Row 1
        print(f"Step MSE: {get_fmt(m_final, 'Step_MSE'):<12} | Step R2: {get_fmt(m_final, 'Step_R2'):<12} | Local MSE: {get_fmt(m_final, 'Local_MSE'):<12} | Local ADE: {get_fmt(m_final, 'Local_ADE')}")
        print("-" * 120)
        # Aggregate Row 2
        print(f"Global OPEN   -> MSE: {get_fmt(m_final, 'Global_open_MSE'):<10} | R2: {get_fmt(m_final, 'Global_open_R2'):<10} | ADE: {get_fmt(m_final, 'Global_open_ADE'):<10} | FDE: {get_fmt(m_final, 'Global_open_FDE')}")
        print(f"Global CLOSED -> MSE: {get_fmt(m_final, 'Global_closed_MSE'):<10} | R2: {get_fmt(m_final, 'Global_closed_R2'):<10} | ADE: {get_fmt(m_final, 'Global_closed_ADE'):<10} | FDE: {get_fmt(m_final, 'Global_closed_FDE')}")

        # --- DETAILED PER-TARGET TABLE ---
        print("\n--- Detailed Breakdown per Target (All Phases) ---")
        
        # Define Columns
        headers = [
            "TARGET", 
            "Step MSE",
            "Step R2",
            "Loc MSE", "Loc ADE", 
            "Open MSE", "Open R2", "Open ADE", "Open FDE", 
            "Clos MSE", "Clos R2", "Clos ADE", "Clos FDE"
        ]
        
        # Print Header
        # Adjust spacing: 16 for name, 9 for short numbers
        header_str = "{:<16} | {:<9} {:<9} | {:<9} {:<9} | {:<9} {:<9} {:<9} {:<9} | {:<9} {:<9} {:<9} {:<9}".format(*headers)
        print("-" * len(header_str))
        print(header_str)
        print("-" * len(header_str))

        target_names = ["Surge_Velocity", "Sway_Velocity", "Yaw_Rate", "Yaw_Angle"]
        
        for tgt in target_names:
            # Helper to safely get metric for this specific target
            def t_get(metric_suffix):
                key = f"{tgt}_{metric_suffix}"
                val = m_final.get(key)
                if val is None: return "N/A"
                return f"{val:.5f}" if isinstance(val, (int, float)) else str(val)

            row_vals = [
                tgt.replace("_", " "), # Name
                t_get("Step_MSE"),
                t_get("Step_R2"),
                t_get("Local_MSE"), t_get("Local_ADE"),
                t_get("Global_open_MSE"), t_get("Global_open_R2"), t_get("Global_open_ADE"), t_get("Global_open_FDE"),
                t_get("Global_closed_MSE"), t_get("Global_closed_R2"), t_get("Global_closed_ADE"), t_get("Global_closed_FDE")
            ]

            print( "{:<16} | {:<9} {:<9} | {:<9} {:<9} | {:<9} {:<9} {:<9} {:<9} | {:<9} {:<9} {:<9} {:<9}".format(*row_vals))
            
        print("-" * len(header_str))

    else:
        print("Metric data missing in file.")

    # --- Training Stats ---
    print("\n--- Training Stats ---")
    train_hist = data.get('train_history', [])
    val_hist = data.get('val_history', [])
    print(f"Total Epochs     : {len(train_hist)}")
    if train_hist: print(f"Final Train Loss : {train_hist[-1]:.6f}")
    if val_hist:   print(f"Final Val Loss   : {val_hist[-1]:.6f}")
        
    print("="*120 + "\n")
    
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

def _ensure_dir_exists(filename):
    """Creates the directory structure for a file if it doesn't exist."""
    if filename:
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

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
        _ensure_dir_exists(filename)  # <--- FIXED HERE
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Convergence plot saved to {filename}")
    if args.show_plot: plt.show()
    plt.close()

# ==========================================
# PLOT 1: LOCAL BRANCHES (Dynamics)
# ==========================================
def plot_kinematics_branches(args, data, horizons=[1,5],step_interval=20, filename=None):
    """
    Plots short horizon predictions branching off the true path for all 4 targets.
    Replaces: plot_horizon_branches
    """
    true_path = data['true_backbone'] # (N, 4)
    pred_path = data['local']['pred_path'] # (N, H, 4)
    time_steps = np.arange(len(true_path))

    if horizons is None:
        horizons = [args.horizon]
    # If single int, convert to list
    elif isinstance(horizons, (int, float)):
        horizons = [int(horizons)]

    max_h = pred_path.shape[1]
    horizons = [min(h, max_h) for h in horizons]
    horizons.sort(reverse=True)

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
        # Loop through requested horizons (e.g. 5, then 1)
        for h_idx, h in enumerate(horizons):
            
            # Create time indices for this specific branch length
            t_indices = np.arange(i, i + h + 1)
            if t_indices[-1] >= len(time_steps): continue
            
            # Get the branch data: Start at true[i], then append preds up to step h
            curr_true = true_path[i].reshape(1, 4)
            curr_pred = pred_path[i, :h, :] # Slice: take only first h steps
            branch_data = np.vstack([curr_true, curr_pred]) # (h+1, 4)

            # Pick color: Cycle through colors based on horizon index
            # If 'colors' exists globally use it, else fallback
            if 'colors' in globals() and len(colors) > 0:
                c = colors[(h_idx + 1) % len(colors)] 
            else:
                c = ['#1f77b4', '#ff7f0e', '#2ca02c'][h_idx % 3]

            # Plot on all 4 subplots
            for idx, ax in enumerate(axes):
                # Legend logic: Only label the first branch of the first timestep
                lbl = f'Pred (H={h})' if (i == 0 and idx == 0) else ""
                
                # Plot
                ax.plot(t_indices, branch_data[:, idx], color=c, 
                        linestyle='-', linewidth=1.5, alpha=0.8, label=lbl if idx==0 else "")

    # Add legend to first plot (it will contain entries for True Path and each Horizon length)
    axes[0].legend(prop=legend_prop if 'legend_prop' in globals() else None)
    
    for ax in axes: 
        if '_force_ticks_font' in globals(): _force_ticks_font(ax)

    h_str = ",".join(map(str, horizons))
    fig.suptitle(f"Local Horizon Branches (H={h_str})", y=0.96, **title_style)
    
    if args.save_plot and filename:
        _ensure_dir_exists(filename) 
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    if args.show_plot: plt.show()
    plt.close()

# ==========================================
# PLOT 2: KINEMATICS VS TIME (4 Targets)
# ==========================================
def plot_kinematics_time_series(args, data, loop='closed', horizon_steps=[1,5], filename=None):
    """
    Plots Surge, Sway, Yaw Rate, and Yaw Angle vs Time in a 2x2 grid.
    """
    if isinstance(horizon_steps, (int, float)):
        horizon_steps = [int(horizon_steps)]
    
    # Sort so legend is orderly
    horizon_steps.sort()
    # 1. Extract Data
    # true_backbone shape expected: (N, 4) -> [Surge, Sway, Yaw Rate, Yaw Angle]
    true_data = data['true_backbone'] 
    
    # pred_path shape expected: (N, Horizon, 4)
    # We extract the specific horizon step 'k' to get a continuous trajectory (N, 4)
    raw_pred = data['global'][loop]['pred_path']

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
        row, col = i // 2, i % 2
        ax = fig.add_subplot(gs[row, col])
        
        # Plot True
        time_steps_true = np.arange(len(true_data))
        ax.plot(time_steps_true, true_data[:, target['idx']], 'k-', linewidth=1.5, alpha=0.4, label='True')
        
        for h_idx, h in enumerate(horizon_steps):
            k = h - 1
            if k >= raw_pred.shape[1]: 
                continue # Skip if horizon is out of bounds

            # Shift logic to align "k-th step prediction" with "Time t"
            if k == 0:
                pred_seq = raw_pred[:, k, :]
                true_seq_aligned = true_data
                start_t = 0
            else:
                pred_seq = raw_pred[:-k, k, :]
                true_seq_aligned = true_data[k:]
                start_t = k # Plot offset so X-axis matches reality

            min_len = min(len(true_seq_aligned), len(pred_seq))
            pred_seq = pred_seq[:min_len]
            
            # Time axis for this specific line
            t_axis = np.arange(start_t, start_t + min_len)

            # Color cycling
            if 'colors' in globals(): c = colors[h_idx % len(colors)]
            else: c = ['#D62728', '#1f77b4', '#2ca02c'][h_idx % 3]

            # Plot Prediction
            ax.plot(t_axis, pred_seq[:, target['idx']], '--', color=c, 
                    linewidth=1.8, alpha=0.9, label=f'Pred (k={h})')

        # Styling
        ax.set_title(target['name'], **subtitle_style)
        ax.set_ylabel(rf"$\mathit{{{target['name'].split()[0]}}}$ [{target['unit']}]", **label_style)
        if row == 1: ax.set_xlabel(r"$\mathit{Time\ Step}$", **label_style)
        ax.grid(True, linestyle='--', alpha=0.5, linewidth=1.0)
        if '_force_ticks_font' in globals(): _force_ticks_font(ax)
        axes.append(ax)

    # Legend on first plot
    axes[0].legend(prop=legend_prop if 'legend_prop' in globals() else None, loc='best')

    # Final Layout
    h_str = ",".join(map(str, horizon_steps))
    fig.suptitle(f"Kinematics Analysis ({loop.capitalize()} - Steps: {h_str})", y=0.96, **title_style)
    
    if args.save_plot and filename:
        _ensure_dir_exists(filename) # <--- FIXED HERE
        if not filename.endswith('.png'): filename += f"_k{h_str.replace(',', '-')}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    if args.show_plot: plt.show()
    plt.close()

# ==========================================
# PLOT 3: ERROR ANALYSIS
# ==========================================
def plot_kinematics_errors(args, data, mode='global', loop='closed', horizon_mode='mean', filename=None):
    """
    Generates 4 separate plots (one per target).
    Each plot has 2 subplots:
      1. Top: Accumulated Error (Left Axis) vs Net Error (Right Axis).
      2. Bottom: True Trajectory of that feature.
    """
    if mode not in data: return
    
    # 1. Setup Data
    if mode == 'local': pred_obj = data[mode]
    else: pred_obj = data[mode][loop]
    
    # Shapes: (N, H, 4)
    pred_deltas_all_h = pred_obj['pred_deltas_denorm']
    true_deltas_all_h = data['true_deltas_denorm']
    pred_path_all_h = pred_obj['pred_path'] 
    true_path_all_h = data['true_path']
    true_backbone = data['true_backbone'] # Shape (N, 4)

    # Sync lengths
    num_points = min(pred_deltas_all_h.shape[0], true_deltas_all_h.shape[0])
    pred_deltas_all_h = pred_deltas_all_h[:num_points]
    true_deltas_all_h = true_deltas_all_h[:num_points]
    pred_path_all_h = pred_path_all_h[:num_points]
    true_path_all_h = true_path_all_h[:num_points]
    
    # True path flat (N, 4) used for the bottom plot
    true_path_flat = true_backbone[:num_points]         
    time_steps = np.arange(num_points)

    # Define Targets
    targets = [
        {"name": "Surge Velocity", "unit": "m/s", "idx": 0},
        {"name": "Sway Velocity",  "unit": "m/s", "idx": 1},
        {"name": "Yaw Rate",       "unit": "rad/s", "idx": 2},
        {"name": "Yaw Angle",      "unit": "rad", "idx": 3}
    ]

    # Ensure output directory exists (handles the 'compare_error' subfolder)
    if filename:
        _ensure_dir_exists(filename)

    # 2. Iterate over each Target to create separate plots
    for tgt in targets:
        idx = tgt['idx']
        t_name = tgt['name']
        t_unit = tgt['unit']
        
        # Calculate Errors for this SPECIFIC target
        # Abs difference: (N, H)
        raw_step_errors = np.abs(true_deltas_all_h[:, :, idx] - pred_deltas_all_h[:, :, idx])
        raw_pos_errors  = np.abs(true_path_all_h[:, :, idx] - pred_path_all_h[:, :, idx])
        max_h = raw_step_errors.shape[1]

        # Prepare Tasks (Mean, Max, or Specific Horizon)
        tasks = []
        if isinstance(horizon_mode, (str, int)): horizon_mode_list = [horizon_mode]
        else: horizon_mode_list = horizon_mode
            
        for h in horizon_mode_list:
            if h == 'mean':
                tasks.append( ("Avg H", np.mean(raw_step_errors, axis=1), np.mean(raw_pos_errors, axis=1)) )
            elif h == 'max':
                tasks.append( ("Max H", np.max(raw_step_errors, axis=1), np.max(raw_pos_errors, axis=1)) )
            elif isinstance(h, int):
                k = h - 1
                if k < max_h:
                    tasks.append( (f"H{h}", raw_step_errors[:, k], raw_pos_errors[:, k]) )

        # --- PLOTTING ---
        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)

        # A. Top Plot: Accumulated vs Net Error
        ax_top_left = plt.subplot(gs[0])
        ax_top_right = ax_top_left.twinx()

        num_lines = len(tasks)
        if num_lines == 1:
            colors_acc = ['#D62728'] # Red
            colors_net = ['#1F77B4'] # Blue
        else:
            colors_acc = [cm.Reds(x) for x in np.linspace(0.5, 1.0, num_lines)]
            colors_net = [cm.Blues(x) for x in np.linspace(0.5, 1.0, num_lines)]

        lines_legend = []
        for i, (label, s_err, p_err) in enumerate(tasks):
            accumulated_error = np.cumsum(s_err)
            
            # Left Axis: Accumulated
            l1, = ax_top_left.plot(time_steps, accumulated_error, color=colors_acc[i], 
                                   alpha=0.9, linewidth=2.5, label=f'Acc Error ({label})')
            
            # Right Axis: Net
            l2, = ax_top_right.plot(time_steps, p_err, color=colors_net[i], 
                                    alpha=0.7, linewidth=2.0, linestyle='--', label=f'Net Error ({label})')
            
            lines_legend.extend([l1, l2])

        # Styling Top
        ax_top_left.set_ylabel(rf"$\mathit{{Accumulated\ Error}}$ [{t_unit}]", color=colors_acc[0], **label_style)
        ax_top_left.tick_params(axis='y', labelcolor=colors_acc[0])
        ax_top_left.grid(True, linestyle=':', alpha=0.6, linewidth=1.5)
        
        ax_top_right.set_ylabel(rf"$\mathit{{Net\ Error}}$ [{t_unit}]", color=colors_net[0], **label_style)
        ax_top_right.tick_params(axis='y', labelcolor=colors_net[0])

        ax_top_left.legend(handles=lines_legend, loc='upper left', prop=legend_prop, ncol=2)
        
        horizon_str = ", ".join([t[0] for t in tasks])
        ax_top_left.set_title(f"{t_name}: Error Analysis ({mode.capitalize()} - {loop} - {horizon_str})", pad=20, **title_style)
        ax_top_left.set_xlim(0, num_points)
        ax_top_left.set_ylim(bottom=0); ax_top_right.set_ylim(bottom=0)

        # B. Bottom Plot: True Path of this Target
        ax_bot = plt.subplot(gs[1])
        c_path = '#2CA02C' # Green

        # Plot the single feature trajectory
        ax_bot.plot(time_steps, true_path_flat[:, idx], color=c_path, linewidth=2.5, label=f'True {t_name}')

        ax_bot.set_ylabel(rf"$\mathit{{{t_name}}}$ [{t_unit}]", color='k', **label_style)
        ax_bot.legend(loc='upper left', prop=legend_prop)
        
        ax_bot.set_title(f"True Trajectory: {t_name}", pad=20, **title_style)
        ax_bot.set_xlabel(r"$\mathit{Time\ Step}$", **label_style)
        ax_bot.set_xlim(0, num_points)
        ax_bot.grid(True, linestyle='--', alpha=0.5, linewidth=1.0)

        # Force Fonts
        for ax in [ax_top_left, ax_top_right, ax_bot]: 
            if '_force_ticks_font' in globals(): _force_ticks_font(ax)

        # --- SAVING ---
        if args.save_plot and filename:
            # Construct filename: .../plot_error_vs_time_Surge_Velocity_global_closed.png
            clean_name = t_name.replace(" ", "_")
            if horizon_mode != ['mean'] and horizon_mode != ['max']:
                h_suffix = "H_" + "_".join([str(h) for h in horizon_mode_list])
            else:
                h_suffix = horizon_mode_list[0]

            if mode == 'local': 
                save_path = f"{filename}_{clean_name}_{mode}_{h_suffix}.png"
            else: 
                save_path = f"{filename}_{clean_name}_{mode}_{loop}_{h_suffix}.png"
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if args.show_plot: 
            plt.show()
            
        plt.close()
# ==========================================
# PLOT 4: BOXPLOTS
# ==========================================
def plot_kinematics_boxplots(args, data, mode='global', loop='closed', filename=None):
    """
    Generates a 2x2 grid of boxplots (one per feature).
    Each boxplot shows the distribution of Absolute Error at each Horizon Step.
    """
    if mode == 'local': pred_obj = data[mode]
    else: pred_obj = data[mode][loop]
    
    # 1. Get Real Data (Physical Units)
    pred_path = pred_obj['pred_path'] 
    true_path = data['true_path']
    
    num_samples = min(pred_path.shape[0], true_path.shape[0])
    
    # Calculate Absolute Error per component: Shape (N, H, 4)
    abs_error = np.abs(true_path[:num_samples] - pred_path[:num_samples])
    
    horizon_steps = abs_error.shape[1]
    
    # 2. Setup Figure (2x2 Grid)
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.25)

    targets = [
        {"name": "Surge Velocity", "unit": "m/s", "idx": 0},
        {"name": "Sway Velocity",  "unit": "m/s", "idx": 1},
        {"name": "Yaw Rate",       "unit": "rad/s", "idx": 2},
        {"name": "Yaw Angle",      "unit": "rad", "idx": 3}
    ]

    # 3. Loop through targets
    for i, tgt in enumerate(targets):
        row, col = i // 2, i % 2
        ax = fig.add_subplot(gs[row, col])
        
        idx = tgt['idx']
        
        # Prepare Data for Boxplot: List of (N,) arrays, one per horizon step
        # Extract error for specific feature 'idx'
        feature_error = abs_error[:, :, idx] 
        plot_data = [feature_error[:, k] for k in range(horizon_steps)]
        
        # Calculate Mean for the Diamond marker
        step_means = np.mean(feature_error, axis=0)

        # Draw Boxplot
        box = ax.boxplot(plot_data, patch_artist=True, showfliers=False, widths=0.6,
                         medianprops=dict(linewidth=2.0, color='#000080')) # Navy Median
        
        # Style Boxes
        c_face = '#ADD8E6' # Light Blue
        c_edge = '#1F77B4' # Dark Blue
        for patch in box['boxes']:
            patch.set_facecolor(c_face)
            patch.set_edgecolor(c_edge)
            patch.set_alpha(0.7)
            
        # Plot Mean Markers
        x_pos = np.arange(1, horizon_steps + 1)
        ax.plot(x_pos, step_means, marker='D', color='#D62728', linestyle='None', 
                markersize=6, label='Mean Error')

        # Labels & Grid
        ax.set_title(tgt['name'], **subtitle_style)
        ax.set_ylabel(rf"$\mathit{{Abs\ Error}}$ [{tgt['unit']}]", **label_style)
        
        # Only X-label on bottom rows
        if row == 1: 
            ax.set_xlabel(r"$\mathit{Horizon\ Step}$", **label_style)
            
        ax.grid(True, linestyle='--', alpha=0.5)
        if '_force_ticks_font' in globals(): _force_ticks_font(ax)

    # Add Legend to the first plot only (to avoid clutter)
    # Creating a custom legend handle for the box
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ADD8E6', edgecolor='#1F77B4', label='IQR (Distribution)'),
        Line2D([0], [0], color='#000080', linewidth=2.0, label='Median'),
        Line2D([0], [0], marker='D', color='#D62728', linestyle='None', markersize=6, label='Mean'),
    ]
    ax = fig.axes[0]
    ax.legend(handles=legend_elements, loc='upper left', prop=legend_prop)

    # Title
    fig.suptitle(f"Horizon Error Distribution ({mode.capitalize()}-{loop})", y=0.96, **title_style)
    if args.save_plot and filename:
        _ensure_dir_exists(filename) # <--- FIXED HERE
        plt.savefig(filename + f"_boxplots_{mode}.png", dpi=300, bbox_inches='tight')
    if args.show_plot: plt.show()
    plt.close()