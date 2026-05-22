# Data gestion
import os
import pickle
import argparse
from copy import deepcopy
import glob
import json

# Time libraries
import time
import datetime

# Math, data manipulation and plotting
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Data preprocessing and metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge

# Data handling
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

#Qiskit framework
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap, EfficientSU2, ExcitationPreserving, PauliTwoDesign, RealAmplitudes
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_algorithms.optimizers import SPSA, COBYLA
import qiskit_algorithms

# Logging and warnings
C_RED = '\033[91m'
C_YELLOW = '\033[93m'
C_GREEN = '\033[92m'
C_BLUE = '\033[94m'
C_RESET = '\033[0m'

colors = ['#E60000', '#FF8C00', '#C71585', '#008080', '#1E90FF']

# Dataset variables
full_feature_set = [ "Surge Velocity", "Sway Velocity", "Yaw Rate", "Yaw Angle", "Speed U", "Rudder Angle (deg)", "Rudder Angle (rad)", "Abs Sway", "Abs Rudder"]


# ==============================================================================
# 1. DATA & UTILITY FUNCTIONS
# ==============================================================================

def str2bool(v): # To handle boolean flags

    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')


def map_names(feature_list, reverse=False): # To shorten the feature names for mapping and display purposes
    
    code_to_name = {
        "sv":"Surge Velocity", "wv":"Sway Velocity", 
        "yr":"Yaw Rate", "ya":"Yaw Angle",
        "vu": "Speed U", "radeg": "Rudder Angle (deg)",
        "rarad": "Rudder Angle (rad)", "OOD": "OOD Label",
        "dsv":"delta Surge Velocity", "dwv":"delta Sway Velocity",
        "dyr":"delta Yaw Rate", "dya":"delta Yaw Angle",
        "asv":"Abs Sway", "ararad":"Abs Rudder",
        "dasv":"delta Abs Sway", "dararad":"delta Abs Rudder",
        "effsu2": "efficientsu2", "ugates": "ugates", "realamplitudes": "realamp",
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

def process_single_df(df): # Performs feature engineering on a single dataframe (calculte deltas and shift controls)
    
    for col in ['Surge Velocity', 'Sway Velocity', 'Yaw Rate', 'Yaw Angle']:
        if col in df.columns:
            df[f'delta {col}'] = df[col].diff().fillna(0)
    control_cols = ["Rudder Angle (deg)", "Rudder Angle (rad)"]
    for col in control_cols:
        if col in df.columns:
            df[col] = df[col].shift(-1)
    df.dropna(inplace=True)
    return df

def sliding_window(x, y, window_size, horizon): # Creates windows from a single continuous trajectory taking the horizons into account.
    
    x_wins, y_wins = [], [] # wins means window not winning.
    limit = len(x) - window_size - horizon + 1
    
    for i in range(limit):
        x_wins.append(x[i : i + window_size])
        y_wins.append(y[i + window_size : i + window_size + horizon])
        
    return np.array(x_wins), np.array(y_wins)

def prepare_dataset_from_directory(directory, args, x_scaler=None, y_scaler=None, fit_scalers=False): # Loads all CSVs from a directory, processes them individually normalizes them (fitting on Train only), and stacks them into a dataset.

    files = glob.glob(os.path.join(directory, "*.csv")) # Assumes all CSV files in the directory are part of the dataset. Adjust if there are non-data CSVs.
    if not files:
        raise ValueError(f"No CSV files found in {directory}")
    
    print(f"Loading {len(files)} files from {directory}...")
    raw_dfs = [pd.read_csv(f, index_col=0) for f in files] # Load all CSVs into dataframes
    processed_dfs = [process_single_df(df.copy()) for df in raw_dfs] # Process each dataframe (calculate deltas + shift controls)

    if not hasattr(args, 'features_resolved') or not args.features_resolved: # Select features based on args (select or drop)
        select_list = getattr(args, 'select_features', None)
        drop_list = getattr(args, 'drop_features', None)
        if select_list:
            args.features = map_names(args.select_features)
        elif drop_list:
            args.features = [f for f in full_feature_set if f not in map_names(args.drop_features)]
        else:
            args.features = full_feature_set
        predict_type = getattr(args, 'predict', 'motion')
        if predict_type == 'motion': args.targets = ["Surge Velocity","Sway Velocity","Yaw Rate","Yaw Angle"]
        elif predict_type == 'delta': args.targets = ["delta Surge Velocity", "delta Sway Velocity", "delta Yaw Rate", "delta Yaw Angle"]
        elif predict_type == 'custom': args.targets = map_names(args.custom_targets)
        args.features_resolved = True
    x_seqs, y_seqs = [], []
    for df in processed_dfs:
        x_seqs.append(df[args.features].values) # Select features for X based on the resolved feature list and target list
        y_seqs.append(df[args.targets].values) # Select targets for Y based on the resolved target list

    if fit_scalers:
        all_x = np.concatenate(x_seqs, axis=0) # Fit scalers on the entire input feature dataset  if fit_scalers is True (should only be True for the Train set to avoid data leakage) puting first the first dataset, then the second and so on, ensuring that the temporal order is maintained.
        all_y = np.concatenate(y_seqs, axis=0) # Fit scalers on the entire input target dataset  if fit_scalers is True (should only be True for the Train set to avoid data leakage) puting first the first dataset, then the second and so on, ensuring that the temporal order is maintained.
        x_scaler = MinMaxScaler(feature_range=(0, np.pi)) # Scale inputs to [0, pi] for angle encoding, to ensure that the range goes from a pure state to a maximally mixed one.
        x_scaler.fit(all_x) # Only fit for the Train set.
        if getattr(args, 'norm', True):
            y_scaler = MinMaxScaler(feature_range=(-1, 1)) # Scale targets to [-1, 1] if normalization is enabled, otherwise keep original scale (which might be large for some features and small for others, but the QNN should be able to handle it as long as the input encoding is consistent)
            y_scaler.fit(all_y) # Only fit for the Train set.
        else:
            y_scaler = None # If not normalizing, we won't use a scaler for Y.
    final_x_wins, final_y_wins = [], []
    for x, y in zip(x_seqs, y_seqs):# We iterate through each sequence, normalize it, and create windows. We do this after fitting the scalers on the entire dataset to ensure that the scaling is consistent across all sequences.
        x_norm = x_scaler.transform(x) 
        x_norm = np.clip(x_norm, 0, np.pi) # Ensure that all inputs are within the expected range after scaling, to prevent issues with the quantum encoding. This is a safeguard in case there are outliers or if the scaler produces values slightly outside the range due to numerical precision.
        
        if getattr(args, 'norm', True) and y_scaler:
            y_norm = y_scaler.transform(y) # Only normalize Y if normalization is enabled and we have a fitted scaler. If normalization is disabled, we keep the original target values, which might be on different scales but should still be learnable by the QNN as long as the input encoding is consistent.
        else:
            y_norm = y # If the normalization is not enabled.
        xw, yw = sliding_window(x_norm, y_norm, args.window_size, args.horizon) # Create windows from the normalized sequences. If the sequence is too short to create any windows given the window_size and horizon, we skip it.
        if len(xw) > 0: 
            final_x_wins.append(xw)
            final_y_wins.append(yw)
    if not final_x_wins:
        raise ValueError(f"No valid windows created from {directory}. Check window_size/horizon vs file lengths.")

    # 6. Stack
    X_data = np.concatenate(final_x_wins, axis=0) # Final data with the windows created. (If it was 1,2,3,4,5 it becomes [[1,2],[2,3],[3,4],[4,5]] for window_size=2 and horizon=1 for example and if it was 1,2,3,4,5 and horizon was 2 it becomes [[[1,2],[2,3]], [[2,3],[3,4]], [[3,4],[4,5]]])
    Y_data = np.concatenate(final_y_wins, axis=0) # Final targets with the windows created. (If it was 1,2,3,4,5 it becomes [[3],[4],[5]] for window_size=2 and horizon=1 for example and if it was 1,2,3,4,5 and horizon was 2 it becomes [[[3],[4]], [[4],[5]]])
    
    return X_data, Y_data, x_scaler, y_scaler



# ==============================================================================
# 2. CIRCUIT CONSTRUCTION
# ==============================================================================
def _parse_feature_map(map_input, selected_features): # Parses mixed tokens (int strings, feature codes) into integers.

    if map_input is None: return None
    indices = []
    feat_to_idx = {name: i for i, name in enumerate(selected_features)}
    for item in map_input: # We try to parse each item as an integer index first. If that fails, we treat it as a feature code and look it up in the mapping. If it's not found in the mapping, we raise an error.
        try: 
            val = int(item)
            indices.append(val)
        except ValueError: 
            full_name_list = map_names([item])
            if not full_name_list: raise ValueError(f"Unknown code: {item}")
            full_name = full_name_list[0]
            if full_name in feat_to_idx: indices.append(feat_to_idx[full_name])
            else: raise ValueError(f"Feature '{item}' ({full_name}) in map but NOT in selected features: {selected_features}")
    return indices

def _validate_chunk_completeness(chunk, num_features, layer_idx=None):
    """
    Validates that a layer contains all features at least once.
    ALLOWS duplicate features for parameter repetition (re-uploading).
    """
    valid = [x for x in chunk if x != -1]
    context = f"Layer {layer_idx}" if layer_idx is not None else "Template"
    required_features = set(range(num_features))
    current_features = set(valid)
    
    if current_features != required_features:
        missing = required_features - current_features
        extra = current_features - required_features
        
        error_msg = f"[ERROR] [Map] {context} is incomplete."
        if missing: 
            error_msg += f" Missing features: {missing}."
        if extra: 
            error_msg += f" Invalid feature indices found: {extra}."
        
        raise ValueError(f"{error_msg} Segment: {chunk}")
def _load_and_validate_map(args, config):
    """Main processor for feature map parsing and validation."""
    num_padding = config["total_slots"] - config["num_features"]
    canonical_map = np.concatenate([np.arange(config["num_features"]), np.full(num_padding, -1)]).astype(int)
    raw_map = getattr(args, 'map', None)
    if raw_map is None:
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
    raw_map = getattr(args, 'map', None)
    min_ugates = math.ceil(num_features / 3)
    min_slots_per_layer = min_ugates * 3
    if raw_map is None:
        num_ugates = min_ugates
    else:
        total_map_len = len(raw_map)
        if args.reps > 1 and total_map_len % (args.reps * 3) == 0 and total_map_len >= (min_slots_per_layer * args.reps):
            num_ugates = (total_map_len // args.reps) // 3
        else:
            if total_map_len % 3 != 0:
                raise ValueError(f"Map length ({total_map_len}) must be a multiple of 3.")
            num_ugates = total_map_len // 3
    total_slots = num_ugates * 3
    hidden_layer = getattr(args, 'hidden_layer', 0)
    if args.encoding == 'compact':
        qubits_per_step, num_qubits, sub_layers = 1, args.window_size + hidden_layer, 1
    elif args.encoding == 'parallel':
        qubits_per_step, num_qubits, sub_layers = num_ugates, (args.window_size * num_ugates)+hidden_layer, 1
    elif args.encoding == 'serial':
        qubits_per_step, num_qubits, sub_layers = 1, args.window_size + hidden_layer, num_ugates
    elif args.encoding == 'accumulated':
        qubits_per_step,num_qubits, sub_layers = 1, args.window_size + hidden_layer, 1
    
    return {
        "num_features": num_features,
        "num_ugates": num_ugates,
        "total_slots": total_slots,
        "strategy": args.encoding,
        "num_qubits": num_qubits,
        "total_physical_layers": args.reps * sub_layers,
        "qubits_per_step": qubits_per_step,
        "weights_per_layer": 0,
        "hidden_layer": hidden_layer
    }


def _get_params_for_gates(chunk_idx, num_features, input_params, base_idx, rep_indices, encoding_params=None):
    """Fetches parameters for a U-gate, handling padding."""
    p = []
    for k in range(3):
        slot_idx = chunk_idx * 3 + k # Calculate exactly which slot in the layer map we are accessing
        feat_idx = rep_indices[slot_idx]  # Get the feature index mapped to this slot
        if feat_idx != -1: 
            val = input_params[base_idx + feat_idx]
            if encoding_params is not None: val = val*encoding_params[feat_idx] # Add any additional encoding parameters if provided (e.g., for parameter repetition)
            p.append(val) # Valid feature: Read from input parameters
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

def _append_ansatz_and_entangle(qc, args, weight_params, weight_idx, ansatz_obj, weights_per_layer, layer_idx, apply_entanglement=True):
    """Adds entanglement and trainable ansatz."""
    if apply_entanglement:
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
def _build_compact_block(qc, args, config, input_params, weight_params, weight_idx, rep_indices, ansatz_obj, current_layer, encoding_params = None):

    for t in range(args.window_size):
        base_idx = t * config["num_features"]
        for chunk_idx in range(config["num_ugates"]):
            p = _get_params_for_gates(chunk_idx, config["num_features"], input_params, base_idx, rep_indices, encoding_params)
            qc.u(p[0], p[1], p[2], t) 

    return _append_ansatz_and_entangle(qc, args, weight_params, weight_idx, ansatz_obj, config["weights_per_layer"], current_layer), current_layer + 1

def _build_parallel_block(qc, args, config, input_params, weight_params, weight_idx, rep_indices, ansatz_obj, current_layer, encoding_params = None):
    
    for t in range(args.window_size):
        base_idx = t * config["num_features"]
        for chunk_idx in range(config["num_ugates"]):
            p = _get_params_for_gates(chunk_idx, config["num_features"], input_params, base_idx, rep_indices, encoding_params)
            target_qubit = (t * config["qubits_per_step"]) + chunk_idx
            qc.u(p[0], p[1], p[2], target_qubit)
    return _append_ansatz_and_entangle(qc, args, weight_params, weight_idx, ansatz_obj, config["weights_per_layer"], current_layer), current_layer + 1

# --- STRATEGY 3: SERIAL (Deeper Circuit) ---
def _build_serial_block(qc, args, config, input_params, weight_params, weight_idx, rep_indices, ansatz_obj, current_layer, encoding_params = None):
    for s in range(config["num_ugates"]):
        for t in range(args.window_size):
            base_idx = t * config["num_features"]
            p = _get_params_for_gates(s, config["num_features"], input_params, base_idx, rep_indices, encoding_params)
            qc.u(p[0], p[1], p[2], t)
        weight_idx = _append_ansatz_and_entangle(qc, args, weight_params, weight_idx, ansatz_obj, config["weights_per_layer"], current_layer)
        current_layer += 1
    return weight_idx, current_layer

def _build_accumulated_block(qc, args, config, input_params, weight_params, weight_idx, rep_indices, ansatz_obj, current_layer, encoding_params = None):
    for chunk_idx in range(config["num_ugates"]):
        for t in range(args.window_size):
            base_idx = t * config["num_features"]
            p = _get_params_for_gates(chunk_idx, config["num_features"], input_params, base_idx, rep_indices, encoding_params)
            qc.u(p[0], p[1], p[2], t) 
        _apply_entanglement(qc, qc.num_qubits, strategy=args.entangle, layer_index=current_layer)
    weight_idx = _append_ansatz_and_entangle(qc, args, weight_params, weight_idx, ansatz_obj, config["weights_per_layer"], current_layer, apply_entanglement=False)
    return weight_idx, current_layer + 1
def create_multivariate_circuit(args, barriers=False): #TODO: Check if barriers have any effect

    config = _get_encoding_config(args)
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
    if getattr(args, 'use_hadamard', False):
        for i in range(config["num_qubits"]): qc.h(i) 
    input_params = ParameterVector('θ', args.window_size * config["num_features"])
    use_trainable = getattr(args, 'trainable_encoding', False)
    num_lambda = config["num_features"] if use_trainable else 0
    num_omega = config["total_physical_layers"] * config["weights_per_layer"]
    weight_params = ParameterVector('ω', num_lambda + num_omega)
    encoding_params = weight_params[:num_lambda] if use_trainable else None
    ansatz_params = weight_params[num_lambda:] if use_trainable else weight_params
    
    # Map processing
    rep_indices, per_layer_orders = _load_and_validate_map(args, config)
    full_map_history, weight_idx, current_physical_layer = [], 0, 0
    rng = np.random.default_rng(getattr(args, 'run', 0))
    for r in range(args.reps):
        # Select Indices
        curr_idx = per_layer_orders[r] if per_layer_orders is not None else rep_indices 
        if getattr(args, 'reorder', False): curr_idx = rng.permutation(curr_idx)

        # 3. Record
        full_map_history.extend(curr_idx.tolist())
        
        # 4. Build
        if config["strategy"] == 'compact':
            weight_idx, current_physical_layer = _build_compact_block(
                qc, args, config, input_params, ansatz_params, weight_idx, curr_idx, ansatz_obj, current_physical_layer, encoding_params
            )
        elif config["strategy"] == 'serial':
            weight_idx, current_physical_layer = _build_serial_block(
                qc, args, config, input_params, ansatz_params, weight_idx, curr_idx, ansatz_obj, current_physical_layer, encoding_params
            )
        elif config["strategy"] == 'parallel':
            weight_idx, current_physical_layer = _build_parallel_block(
                qc, args, config, input_params, ansatz_params, weight_idx, curr_idx, ansatz_obj, current_physical_layer, encoding_params
            )
        elif config["strategy"] == 'accumulated':
             weight_idx, current_physical_layer = _build_accumulated_block(
                qc, args, config, input_params, ansatz_params, weight_idx, curr_idx, ansatz_obj, current_physical_layer, encoding_params
            )
        
        if barriers: qc.barrier()
        if per_layer_orders is None: rep_indices = curr_idx
    if getattr(args, 'reorder', False) or per_layer_orders is not None:
        args.map = full_map_history
    else:
        args.map = rep_indices.tolist()

    return qc, input_params, weight_params

class WindowEncodingQNN:

    def __init__(self, qnn, output_shape, seed):

        self.qnn = qnn
        self.input_dim = qnn.circuit.num_qubits
        self.horizon = output_shape[1]
        self.columns = output_shape[2]
        self.num_q_params = qnn.num_weights
        self.output_dim = self.horizon*self.columns
        self.num_c_params = (self.input_dim * self.output_dim) + self.output_dim
        self.total_params = self.num_q_params + self.num_c_params
        print(f"[Model] Qubits: {self.input_dim} | Params: {self.num_q_params} (Q) + {self.num_c_params} (C)")        
        if seed is not None: self.rng = np.random.default_rng(seed)

    def forward(self, x, params_flat):
        q_params = params_flat[:self.num_q_params]
        x_flat = x.reshape(x.shape[0], -1)
        y = self.qnn.forward(x_flat, q_params)
        c_params = params_flat[self.num_q_params:]
        W = c_params[:self.input_dim * self.output_dim].reshape(self.input_dim, self.output_dim)
        b = c_params[self.input_dim * self.output_dim:]
        y = np.dot(y, W) + b
        return y.reshape(x.shape[0], self.horizon, self.columns)
    # Inside class WindowEncodingQNN
    def get_quantum_features(self, x, params_flat):
        """Extracts expectation values from the quantum part only."""
        q_params = params_flat[:self.num_q_params]
        x_flat = x.reshape(x.shape[0], -1)
        return self.qnn.forward(x_flat, q_params)
    def initialize_parameters(self, strategy, optimizer_name = 'spsa', trainable_encoding=False, num_features=0):
        scaling_weights = []
        is_qelm = str(optimizer_name).lower() == 'ridge'
        if trainable_encoding:
            if is_qelm: scaling_weights = [self.rng.uniform(0.1, 2.0, size=num_features)]
            else: scaling_weights = [np.ones(num_features)]
        limit = np.sqrt(6 / (self.input_dim + self.output_dim))
        num_lambda = num_features if trainable_encoding else 0
        num_remaining_q = self.num_q_params - num_lambda        
        if strategy == 'identity':
            if optimizer_name.lower() == 'spsa':
                # SPSA: Scale matching is essential for global perturbation
                q_params = self.rng.uniform(-limit, limit, size=num_remaining_q)
                c_params = self.rng.uniform(-limit, limit, size=self.num_c_params)
            else:
                # COBYLA: Stay near identity but provide seed direction
                q_params = self.rng.uniform(-0.1, 0.1, size=num_remaining_q)
                c_params = self.rng.uniform(-limit, limit, size=self.num_c_params)
        elif strategy == 'uniform': 
            q_params = self.rng.uniform(0, 2*np.pi, size=num_remaining_q) 
            c_params = self.rng.uniform(-limit, limit, size=self.num_c_params)
        return np.concatenate(scaling_weights + [q_params, c_params])

class ClassicalMLP(nn.Module):
    def __init__(self, input_size, window_size, hidden_size, num_layers, output_size = None, seed=42, horizon=5, num_targets=4):
        super().__init__()
        torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.horizon = horizon
        self.num_targets = num_targets
        flat_input_dim = input_size * window_size
        layers = []
        layers.append(nn.Linear(flat_input_dim, hidden_size))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers).to(self.device)
    def set_weights(self, weights):
        """Helper to inject flat numpy weights into the PyTorch layers."""
        weights_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)
        ptr = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(weights_tensor[ptr : ptr + numel].view(p.shape))
            ptr += numel
    def forward(self, x, params=None):
        if params is not None: self.set_weights(params)
        if isinstance(x, np.ndarray): x = torch.tensor(x, dtype=torch.float32).to(self.device)
        x_flat = x.view(x.size(0), -1)
        out = self.network(x_flat)
        # Reshape to [Batch, Horizon, Targets] to match y_batch shape
        return out.view(x.size(0), self.horizon, self.num_targets)
class ClassicalMultiHeadMLP(nn.Module):
    def __init__(self, heads_config, features, window_size, horizon, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.heads = nn.ModuleList()
        self.input_indices = []
        self.horizon = horizon

        for h_cfg in heads_config:
            # Map head features to indices based on global feature list
            head_feat_names = map_names(h_cfg.get('features', []))
            indices = [features.index(f) for f in head_feat_names]
            self.input_indices.append(indices)
            
            # Architecture: Input (Window * Head Features) -> Output (Horizon * Head Targets)
            in_dim = len(head_feat_names) * window_size
            out_dim = horizon * h_cfg.get('output_dim', 1)
            hidden_size = h_cfg.get('hidden_size', 4)
            
            head_net = nn.Sequential(
                nn.Linear(in_dim, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, out_dim)
            )
            self.heads.append(head_net.to(self.device))

    def set_weights(self, weights):
        """Helper to inject flat numpy weights into all heads."""
        weights_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)
        ptr = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(weights_tensor[ptr : ptr + numel].view(p.shape))
            ptr += numel

    def forward(self, x, params=None):
        if params is not None: self.set_weights(params)
        if isinstance(x, np.ndarray): x = torch.tensor(x, dtype=torch.float32).to(self.device)

        head_outputs = []
        for i, head in enumerate(self.heads):
            # Slice input for this head
            x_head = x[:, :, self.input_indices[i]].reshape(x.size(0), -1)
            c_out = head(x_head)
            # Reshape head output to [Batch, Horizon, Targets_per_head]
            head_outputs.append(c_out.view(x.size(0), self.horizon, -1))
            
        return torch.cat(head_outputs, dim=2)
class ClassicalWrapper:
    def __init__(self, torch_model, device,output_shape = None):
        self.model = torch_model
        self.device = device
        self.num_targets = output_shape[2] if output_shape else 4
    def initialize_parameters(self, method='uniform', optimizer_name='spsa', trainable_encoding=False, num_features=0):
        total_params = sum(p.numel() for p in self.model.parameters())
        fan_in = list(self.model.parameters())[0].shape[1] 
        fan_out = list(self.model.parameters())[-1].shape[0]
        limit = np.sqrt(6 / (fan_in + fan_out))
        if method == 'identity':
            weights = np.random.uniform(-limit, limit, total_params)
        elif method == 'uniform':
            weights = np.random.uniform(-0.5, 0.5, total_params)
        else:
            weights = np.random.randn(total_params) * 0.1
            
        self.set_weights(weights)
        return weights
    def get_weights(self):
        return torch.cat([p.flatten() for p in self.model.parameters()]).detach().cpu().numpy()
    def set_weights(self, weights):
        weights = np.clip(weights, -5.0, 5.0) 
        weights_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)
        ptr = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.data.copy_(weights_tensor[ptr:ptr + numel].view(p.shape))
            ptr += numel
    def forward(self, x, weights=None):
        if weights is not None:
            self.set_weights(weights)
        self.model.eval()
        t_x = torch.tensor(x, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            out = self.model(t_x)
        return out.cpu().numpy().reshape(x.shape[0], -1, self.num_targets)

class MultiHeadQNN:
    def __init__(self, models_list, input_indices_list):
        self.models = models_list
        self.input_groups = input_indices_list
        self.param_splits, total, total_q, total_c = [], 0, 0, 0 # Track totals
        for m in self.models:
            self.param_splits.append(m.total_params)
            total += m.total_params
            total_q += m.num_q_params
            total_c += m.num_c_params
        self.total_params, self.num_q_params, self.num_c_params = total, total_q, total_c
        print(f"\n[MultiHead] Initialized with {len(self.models)} heads.")
        for i, (n_p, grp) in enumerate(zip(self.param_splits, self.input_groups)):
            print(f"  > Head {i+1}: {n_p} params | Input Indices: {grp}")
    def get_quantum_features(self, x, params_flat):
        """Concatenates quantum features from all heads."""
        all_phi, param_start = [], 0
        for i, (model, input_idx) in enumerate(zip(self.models, self.input_groups)):
            n_params = model.total_params
            p_head = params_flat[param_start : param_start + n_params]
            x_head = x[:, :, input_idx]
            all_phi.append(model.get_quantum_features(x_head, p_head))
            param_start += n_params
        return np.concatenate(all_phi, axis=1)
    def forward(self, x, params):
        outputs, param_start = [], 0
        for model, input_idx, n_params in zip(self.models, self.input_groups, self.param_splits):
            p_head = params[param_start : param_start + n_params]
            outputs.append(model.forward(x[:, :, input_idx], p_head))
            param_start += n_params
        return np.concatenate(outputs, axis=2)

    def initialize_parameters(self, strategy, optimizer_name='spsa', trainable_encoding=False, num_features=0):
        """
        Initializes parameters for all heads. 
        Note: We ignore the passed 'num_features' and use the specific count for each head.
        """
        params_list = []
        for i, model in enumerate(self.models):
            p_head = model.initialize_parameters(
                strategy, 
                optimizer_name=optimizer_name, 
                trainable_encoding=trainable_encoding, 
                num_features=len(self.input_groups[i])# Calculate how many features THIS specific head uses
            )
            params_list.append(p_head)
            
        return np.concatenate(params_list)
def _compute_loss(args, pred, target, reconstruct, weights, scaler=None):
    num_targets = target.shape[-1]
    if len(weights) != num_targets: weights = np.ones(num_targets)
    if reconstruct and scaler:

        target_real = scaler.inverse_transform(target.reshape(-1, num_targets)).reshape(target.shape)
        pred_real = scaler.inverse_transform(pred.reshape(-1, num_targets)).reshape(pred.shape)

        if getattr(args, 'predict', 'motion') == 'delta':
            target_traj = np.cumsum(target_real, axis=1)
            pred_traj = np.cumsum(pred_real, axis=1)
            sq_diff = (pred_traj - target_traj) ** 2
            weighted_diff = sq_diff * weights 
        else: 
            sq_diff = (pred_real - target_real) ** 2
            weighted_diff = sq_diff * weights
        return np.mean(weighted_diff)
    else:
        if reconstruct and not scaler: print(f'{C_YELLOW}WARNING: Scaler missing --> Cannot reconstruct trajectory.{C_RESET}')
        sq_diff = (pred - target) ** 2
        weighted_diff = sq_diff * weights
        return np.mean(weighted_diff)
    

def train_model(args, model, x_train, y_train, x_val, y_val, scaler=None):

    best_val_loss, best_params, train_hist, val_hist = float('inf'), None, [], []
    
    optimizer_name = args.optimizer.upper()
    batch_size = getattr(args, 'batch_size', 32)
    use_batching = (optimizer_name == 'SPSA') and (batch_size < x_train.shape[0])
    
    num_train_samples = x_train.shape[0]
    learning_rate_arg = args.learning_rate
    if optimizer_name == 'SPSA':
        if isinstance(learning_rate_arg,list) and len(learning_rate_arg) == 2:
            lr_start, lr_end = learning_rate_arg
            print(f"[Optimizer] Dynamic SPSA Learning Rate: {lr_start} -> {lr_end}")
            
            def get_lr_at_k(k):
                progress = min(k, args.maxiter) / args.maxiter
                return lr_start - (lr_start - lr_end) * progress
            def lr_generator_factory():
                k = 0
                while True:
                    yield get_lr_at_k(k)
                    k += 1
            spsa_lr_optimizer = lr_generator_factory
            spsa_learning_rate = get_lr_at_k
        else:
            val = learning_rate_arg[0] if isinstance(learning_rate_arg, list) else learning_rate_arg
            print(f"[Optimizer] Static SPSA Learning Rate: {val}")
            def static_generator():
                while True: yield float(val)
            spsa_lr_optimizer = static_generator
            spsa_learning_rate = lambda k: float(val)
    current_batch_x, current_batch_y, call_counter = None, None, 0
    def objective_function(params):
        nonlocal best_val_loss, best_params, current_batch_x, current_batch_y, call_counter
        if use_batching:
            if call_counter % 2 == 0:
                indices = np.random.choice(num_train_samples, size=batch_size, replace=False)
                current_batch_x, current_batch_y = x_train[indices], y_train[indices]
            
            x_input, y_target = current_batch_x, current_batch_y
        else:
            x_input,y_target = x_train, y_train
        call_counter += 1
        preds = model.forward(x_input, params)
        train_mse = _compute_loss(args, preds, y_target, args.reconstruct_train, args.weights, scaler)
        current_iter = call_counter // 2
        check_val = True 
        if use_batching and current_iter % 50 != 0:
            check_val = False
        if check_val:
            val_preds = model.forward(x_val, params)
            val_mse = _compute_loss(args, val_preds, y_val, args.reconstruct_val, args.weights, scaler)
            if val_mse < best_val_loss:
                best_val_loss = val_mse
                best_params = np.copy(params)
        else:
            val_mse = val_hist[-1] if val_hist else train_mse
        if optimizer_name == 'SPSA':
            if call_counter % 2 == 0:
                train_hist.append(train_mse)
                val_hist.append(val_mse)

                log_interval = 100
                if len(train_hist) % log_interval == 0:
                    if spsa_learning_rate:
                        lr_val = spsa_learning_rate(len(train_hist))
                        lr_str = f" {lr_val:.5f}"
                    else:
                        lr_str = "N/A"
                    print(f"  > Iter {len(train_hist):4d} | Train: {train_mse:.5f} | Val: {val_mse:.5f} | LR: {lr_str}")
        else:
            train_hist.append(train_mse); val_hist.append(val_mse)

            log_interval = 100 if use_batching else 50
            if len(train_hist) % log_interval == 0:
                print(f"  > Iter {len(train_hist):4d} | Train MSE: {train_mse:.5f} | Val MSE: {val_mse:.5f}")
        return train_mse
   

    start_time = time.time()
    print(f"\n[Training] Starting {args.optimizer.upper()} optimization...")
    if use_batching:
        print(f"  > Mode: Mini-Batch (Size: {batch_size})")
    else:
        print(f"  > Mode: Full-Batch (Size: {num_train_samples})")
    initial_weights = model.initialize_parameters(
            args.initialization, 
            optimizer_name=args.optimizer,
            trainable_encoding=getattr(args, 'trainable_encoding', False),
            num_features=len(args.features)
        )
    if args.optimizer.upper() == 'COBYLA':
        opt = COBYLA(maxiter=args.maxiter, tol = args.tolerance, rhobeg = 0.1)
        res = opt.minimize(fun=objective_function, x0=initial_weights)
    elif args.optimizer.upper() == 'SPSA':
        opt = SPSA(maxiter=args.maxiter,learning_rate=spsa_lr_optimizer, perturbation=args.perturbation) 
        res = opt.minimize(fun=objective_function, x0=initial_weights)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not supported.")
    print(f"Training completed in {(time.time() - start_time) / 60:.2f} min.")
    if best_params is None: best_params = res.x

    return {
        "best_weights": best_params, "best_val_loss": best_val_loss, "final_weights": res.x,             
        "train_history": train_hist, "val_history": val_hist,         
    }

def train_qelm_ridge(args, model, x_train, y_train, x_val, y_val, alpha=1.0):
    print(f"\n{C_BLUE}[qelm-Ridge] Solving linear readout...{C_RESET}")
    
    # 1. Initialize random quantum parameters (the reservoir)
    init_params = model.initialize_parameters(
            args.initialization, 
            optimizer_name=args.optimizer,
            trainable_encoding=getattr(args, 'trainable_encoding', False),
            num_features=len(args.features)
        )
    
    if hasattr(model, 'models'): # Multi-Head Logic
        best_p_list, p_st, t_st = [], 0, 0
        for i, m in enumerate(model.models):
            # A. Extract Head Parts
            p_head_init = init_params[p_st : p_st + m.total_params]
            q_head = p_head_init[:m.num_q_params]
            x_train_h = x_train[:, :, model.input_groups[i]]
            
            # B. Get Quantum Features for this head
            phi_train = m.get_quantum_features(x_train_h, p_head_init)
            
            # C. Extract Targets for this head (N, Horizon * Targets_per_head)
            y_train_h = y_train[:, :, t_st : t_st + m.columns].reshape(y_train.shape[0], -1)
            
            # D. Fit Ridge
            ridge = Ridge(alpha=alpha).fit(phi_train, y_train_h)
            c_head = np.concatenate([ridge.coef_.T.flatten(), ridge.intercept_.flatten()])
            
            # E. Store solved head weights
            best_p_list.append(np.concatenate([q_head, c_head]))
            p_st += m.total_params
            t_st += m.columns
            
        best_weights = np.concatenate(best_p_list)

    else: # Vanilla logic
        q_params = init_params[:model.num_q_params]
        phi_train = model.get_quantum_features(x_train, init_params)
        y_train_flat = y_train.reshape(y_train.shape[0], -1)
        
        ridge = Ridge(alpha=alpha).fit(phi_train, y_train_flat)
        c_params = np.concatenate([ridge.coef_.T.flatten(), ridge.intercept_.flatten()])
        best_weights = np.concatenate([q_params, c_params])

    # Calculate validation MSE for the logger
    val_preds = model.forward(x_val, best_weights)
    val_mse = _compute_loss(args, val_preds, y_val, args.reconstruct_val, args.weights, None)
    
    return {
        "best_weights": best_weights, "final_weights": best_weights,
        "train_history": [0.0], "val_history": [val_mse]
    }

def train_classical_model(args, model, x_train, y_train, x_val, y_val, y_scaler = None, device='cpu'):
    """
    Standard PyTorch training loop with Adam optimizer.
    """
    batch_size = args.batch_size if hasattr(args, 'batch_size') else 32
    t_x_train = torch.tensor(x_train, dtype=torch.float32).to(device)
    t_y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    t_x_val = torch.tensor(x_val, dtype=torch.float32).to(device)
    t_y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
    train_loader = DataLoader(TensorDataset(t_x_train, t_y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(t_x_val, t_y_val), batch_size=batch_size, shuffle=False)
    model = model.to(device)
    criterion = nn.MSELoss()
    if args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError("Unsupported optimizer")

    best_val_loss = float('inf')
    best_weights = None
    train_history, val_history = [], []
    patience = getattr(args, 'patience', 20)
    counter = 0
    
    print(f"\n[Training] Starting Classical Optimization (Adam) on {device}...")
    start_time = time.time()
    for epoch in range(args.maxiter):
        model.train()
        epoch_loss = 0
        
        for x_batch, y_batch in train_loader:

            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = model(x_batch)
            loss = criterion(preds, y_batch) # Forward
            
            optimizer.zero_grad(); loss.backward(); optimizer.step() # Backward

            
            epoch_loss += loss.item()
            
        avg_train_loss = epoch_loss / len(train_loader)
        train_history.append(avg_train_loss)

        model.eval()
        val_loss_accum = 0
        total_samples = 0
        num_targets = y_val.shape[-1]
        with torch.no_grad():
            for x_v, y_v in val_loader:
                val_preds = model(x_v)
                v_p_np = val_preds.cpu().numpy().reshape(-1, args.horizon, num_targets)
                v_y_np = y_v.cpu().numpy().reshape(-1, args.horizon, num_targets)                
                batch_loss = _compute_loss(args, v_p_np, v_y_np, args.reconstruct_val, args.weights, y_scaler) 
                val_loss_accum += batch_loss * x_v.size(0)
                total_samples += x_v.size(0)
        
        avg_val_loss = val_loss_accum / total_samples
        val_history.append(avg_val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"[Training] Epoch {epoch+1:4d} | Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_weights = model.state_dict() 
            counter = 0
        else:
            counter += 1
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
    
    num_samples, horizon, num_targets = x_test.shape[0], args.horizon, len(args.targets)
    state_vars = args.targets
    direct_updates, physics_updates = [], []
    update_rules = {} 
    
    for t_idx, t_name in enumerate(args.targets):
        update_rules[t_idx] = []
        is_target_delta = "delta" in t_name.lower() or args.predict == 'delta'
        
        if t_name in args.features:
            f_idx = args.features.index(t_name)
            update_rules[t_idx].append((f_idx, "DIRECT"))
            direct_updates.append(t_name)
        secondary_name, mode = None, None
        
        if is_target_delta:
            clean = t_name.replace("delta ", "").strip()
            if clean in args.features: secondary_name, mode = clean, "INTEGRATE"; physics_updates.append(f"{clean} (Integrated)")
        else:
            delta_ver = f"delta {t_name}"
            if delta_ver in args.features: secondary_name, mode = delta_ver, "DIFF"; physics_updates.append(f"{delta_ver} (Differentiated)")
        
        if secondary_name:
            f_idx_sec = args.features.index(secondary_name)
            update_rules[t_idx].append((f_idx_sec, mode))
    driven_feats = set()
    for rules in update_rules.values():
        for f_idx, _ in rules: driven_feats.add(f_idx)
    leakage_warning = False
    for state in state_vars:
        if state in args.features:
            s_idx = args.features.index(state)
            if s_idx not in driven_feats:
                print(f"{C_YELLOW}[WARNING] State Variable '{state}' is in INPUT but not updated by OUTPUT. Using Zero-Order Hold (Not GT) to prevent leakage.{C_RESET}")
                leakage_warning = True
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
    recursive_preds = np.zeros((num_samples, horizon, num_targets))
    curr_window = x_test[0:1, :, :].copy()
    last_feat_real = x_scaler.inverse_transform(curr_window[:, -1, :])
    
    last_pred_values = np.zeros(num_targets) 
    for i in range(num_samples):
        preds_full = model.forward(curr_window, best_params)
        recursive_preds[i] = preds_full[0]

        next_gt_idx = min(i + 1, num_samples - 1)
        next_gt_features_norm = x_test[next_gt_idx:next_gt_idx+1, -1, :] 
        next_input_real = last_feat_real.copy()
        next_gt_real = x_scaler.inverse_transform(next_gt_features_norm)

        for f_idx, f_name in enumerate(args.features):
             if "Rudder" in f_name or "Action" in f_name:
                next_input_real[0, f_idx] = next_gt_real[0, f_idx]

        pred_step_norm = preds_full[:, 0, :]
        if y_scaler: pred_step_real = y_scaler.inverse_transform(pred_step_norm)
        else: pred_step_real = pred_step_norm
        for t_idx, updates in update_rules.items():
            pred_val = pred_step_real[0, t_idx]
            for f_idx, mode in updates:
                if mode == "DIRECT": next_input_real[0, f_idx] = pred_val
                elif mode == "INTEGRATE": next_input_real[0, f_idx] = last_feat_real[0, f_idx] + pred_val
                elif mode == "DIFF":
                    diff_val = pred_val if i == 0 else pred_val - last_pred_values[t_idx]
                    next_input_real[0, f_idx] = diff_val
        last_pred_values = pred_step_real[0]
        last_feat_real = next_input_real   
        
        new_row_norm = x_scaler.transform(next_input_real)
        curr_window = np.concatenate([curr_window[:, 1:, :], new_row_norm.reshape(1, 1, -1)], axis=1)

    return recursive_preds, recursive_ratio


def evaluate_model(args, model, params, x_test, y_test, x_scaler, y_scaler):
    """Runs BOTH one-step (Teacher Forcing) and recursive (Dead Reckoning) evaluations."""

    results = {}
    num_targets, orig_shape = y_test.shape[-1], y_test.shape
    target_names = args.targets
    preds_norm_step = model.forward(x_test, params) 
    
    if y_scaler:
        preds_real_step = y_scaler.inverse_transform(preds_norm_step.reshape(-1, num_targets)).reshape(orig_shape)
        y_gt_real = y_scaler.inverse_transform(y_test.reshape(-1, num_targets)).reshape(orig_shape)
    else:
        preds_real_step = preds_norm_step; y_gt_real = y_test
    results['Step_MSE'] = mean_squared_error(y_gt_real.reshape(-1, num_targets), preds_real_step.reshape(-1, num_targets))
    results['Step_R2'] = r2_score(y_gt_real.reshape(-1, num_targets), preds_real_step.reshape(-1, num_targets))

    for i, name in enumerate(target_names):
        clean_name = name.replace(" ", "_")
        results[f'{clean_name}_Step_MSE'] = mean_squared_error(y_gt_real[..., i].flatten(), preds_real_step[..., i].flatten())
        results[f'{clean_name}_Step_R2']  = r2_score(y_gt_real[..., i].flatten(), preds_real_step[..., i].flatten())

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
    for i, name in enumerate(target_names):
        clean_name = name.replace(" ", "_")
        true_i = true_path[..., i].flatten()
        pred_i = pred_path_local[..., i].flatten()
        results[f'{clean_name}_Local_MSE'] = mean_squared_error(true_i, pred_i)

    results['Global_open_MSE'] = mean_squared_error(true_path.reshape(-1, num_targets), pred_path_global_open.reshape(-1, num_targets))
    results['Global_open_R2'] = r2_score(true_path.reshape(-1, num_targets), pred_path_global_open.reshape(-1, num_targets))
    norm_global_error = np.linalg.norm(true_path - pred_path_global_open, axis=2)
    results['Global_open_Max'] = np.max(norm_global_error)   

    for i, name in enumerate(target_names):
        clean_name = name.replace(" ", "_")
        true_i = true_path[..., i].flatten()
        pred_i = pred_path_global_open[..., i].flatten()
        
        abs_err = np.abs(true_path[..., i] - pred_path_global_open[..., i])
        
        results[f'{clean_name}_Global_open_MSE'] = mean_squared_error(true_i, pred_i)
        results[f'{clean_name}_Global_open_R2']  = r2_score(true_i, pred_i)

    preds_norm_rec, rec_ratio = recursive_forward_pass(args, model, params, x_test, x_scaler, y_scaler)
    results['Recursivity'] = rec_ratio  
    
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
    results['Global_closed_Max'] = np.max(norm_global_error)  
    for i, name in enumerate(target_names):
        clean_name = name.replace(" ", "_")
        true_i = true_path[..., i].flatten()
        pred_i = pred_path_global[..., i].flatten()
        abs_err = np.abs(true_path[..., i] - pred_path_global[..., i])
        
        results[f'{clean_name}_Global_closed_MSE'] = mean_squared_error(true_i, pred_i)
        results[f'{clean_name}_Global_closed_R2']  =r2_score(true_i, pred_i)


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
def save_experiment_results(args, train_results, val_eval, test_eval, scalers, qnn_dict, timestamp, selection_type="Unknown", excel_path=None, freeze=False):
    save_dir = getattr(args, 'save_dir', '')
    models_dir = os.path.join("models", save_dir)
    logs_dir = os.path.join("logs", save_dir)
    figs_dir = os.path.join("figures", save_dir)
    for folder in [models_dir, logs_dir, figs_dir]: 
        os.makedirs(folder, exist_ok=True)
    if excel_path is None:
        excel_filename = os.path.join(logs_dir, "experiments_summary.xlsx")
    else:
        excel_filename = excel_path
    final_w = train_results.get('selected_weights', train_results['final_weights'])
    # ==========================================================================
    # 1. ROBUST PARAMETER COUNTING (Prevents Crash on Multi-Head)
    # ==========================================================================
    total_params = len(final_w)
    num_q_params = 0
    if isinstance(qnn_dict, list):
        for head_dict in qnn_dict:
            if isinstance(head_dict, dict) and 'weight_params' in head_dict: num_q_params += len(head_dict['weight_params'])
    elif isinstance(qnn_dict, dict) and 'weight_params' in qnn_dict: num_q_params = len(qnn_dict['weight_params'])
    
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

    if getattr(args, 'model', 'vanilla') == 'multihead':
        head_number = len(getattr(args, 'heads_config', []))
    else:
        head_number = 1
    
    def resolve_multi_val(values_list):
        if not values_list: return 'N/A'
        if all(x == values_list[0] for x in values_list):
            return values_list[0] 
        return str(values_list)
    heads_config_str = "N/A"
    if getattr(args, 'model', '') == 'multihead' and hasattr(args, 'heads_config') and args.heads_config:
        list_ansatz, list_entangle, list_reps, list_encoding, list_map, list_features, clean_heads_list = [], [], [], [], [], [],[]
        for h in args.heads_config:
            a_val = map_names([h.get('ansatz', args.ansatz)], reverse=True)[0]
            e_val = map_names([h.get('entangle', args.entangle)], reverse=True)[0]
            r_val = h.get('reps', getattr(args, 'reps', 'N/A'))
            enc_val = h.get('encoding', getattr(args, 'encoding', 'N/A'))
            m_val = h.get('map', getattr(args, 'map', 'N/A'))
            f_val = map_names(h.get('features', []), reverse=True)

            list_ansatz.append(a_val)
            list_entangle.append(e_val)
            list_reps.append(r_val)
            list_encoding.append(enc_val)
            list_map.append(m_val)
            list_features.append(f_val)
            head_clean = {
                'features': f_val,
                'output_dim': h.get('output_dim', 'N/A'),
                'reps': r_val,
                'encoding': enc_val,
                'ansatz': a_val,
                'entangle': e_val,
                'map': m_val
            }
            clean_heads_list.append(head_clean)

        final_ansatz = resolve_multi_val(list_ansatz)
        final_entangle = resolve_multi_val(list_entangle)
        final_reps = resolve_multi_val(list_reps)
        final_encoding = resolve_multi_val(list_encoding)
        final_map = resolve_multi_val(list_map)
        final_features = resolve_multi_val(list_features)
        heads_config_str = str(clean_heads_list)
    # ==========================================================================
    # 3. SAVE PICKLE
    # ==========================================================================
    def clean_filename_str(s):
        s = str(s).replace('[', '').replace(']', '').replace("'", "").replace('"', "")
        return s.replace(', ', '_').replace(',', '_')
    if getattr(args, 'model', 'vanilla') == 'multihead':
        h_count = len(getattr(args, 'heads_config', []))
    else:
        h_count = 1
    run_num = getattr(args, 'run', 0)
    safe_ansatz = clean_filename_str(final_ansatz)
    safe_entangle = clean_filename_str(final_entangle)
    safe_reps = clean_filename_str(final_reps)
    safe_encoding = clean_filename_str(final_encoding)
    model_name = (
        f"{timestamp}_{args.model}_{h_count}heads_run{run_num}_"
        f"{args.optimizer}_{safe_encoding}_f{len(args.features)}_"
        f"w{args.window_size}_h{args.horizon}_{safe_ansatz}_"
        f"{safe_entangle}_r{safe_reps}.pkl"
    )
    model_filename = os.path.join(models_dir, model_name)
    save_payload = {
        "config": vars(args),
        "selected_weights": final_w,
        "weight_selection_method": selection_type,
        "train_history": train_results['train_history'],
        "val_history": train_results['val_history'],
        "val_metrics": val_eval['metrics'],
        "test_metrics": test_eval['metrics'],
        "y_scaler": scalers[1], "x_scaler": scalers[0], "qnn_structure": qnn_dict
    }
    with open(model_filename, "wb") as f: pickle.dump(save_payload, f)

    # ==========================================================================
    # 4. PREPARE EXCEL DATA (Strict Column Order)
    # ==========================================================================
    m_val = val_eval['metrics']
    m_test = test_eval['metrics']
    
    try: dt_object = datetime.datetime.strptime(timestamp, "%m-%d_%H-%M-%S").replace(year=2026)
    except ValueError: dt_object = timestamp 

    raw_data = {}
    explicit_keys = [
        'run', 'features', 'targets','window_size', 'horizon', 'predict', 'reconstruct_train','reconstruct_val', 'head_config', 'head_number','encoding','ansatz','entangle','reps','map','optimizer', 
        'maxiter', 'learning_rate', 'batch_size', 'initialization','perturbation',
        'trainable_encoding', 'freeze_qnn', 'use_hadamard'
    ]
    for key in explicit_keys:
        if hasattr(args, key):
            val = getattr(args, key)
            raw_data[key] = str(val).lower() if isinstance(val, bool) else val

    # Ensure 'freeze' is explicitly captured from the argument passed to the function
    ignore_keys = ['select_features', 'drop_features', 'save_plot', 'show_plot', 'ansatz', 'entangle', 'reps', 'encoding', 'map', 'features']
    
    def clean_val(v):
        if isinstance(v, bool): return str(v).lower()
        if isinstance(v, list): return str(v)
        return v
    
    for key, value in vars(args).items():
        if key not in ignore_keys: raw_data[key] = clean_val(value)
    raw_data['features'] = str(final_features); raw_data['ansatz'] = str(final_ansatz)
    raw_data['entangle'] = str(final_entangle); raw_data['reps'] = str(final_reps)
    raw_data['encoding'] = str(final_encoding); raw_data['map'] = str(final_map)
    raw_data['date'] = dt_object
    raw_data['model_id'] = os.path.basename(model_filename)
    raw_data['data_n'] = getattr(args, 'data_n', 'N/A')
    raw_data['data_dt'] = getattr(args, 'data_dt', 'N/A')
    raw_data['weight_selection'] = selection_type
    raw_data['head_number'] = head_number
    raw_data['heads_config'] = heads_config_str
    raw_data['weights'] = getattr(args, 'weights',"[1.0, 1.0, 1.0, 1.0]")
    raw_data['initialization'] = getattr(args, 'initialization', 'N/A')
    raw_data['perturbation'] = getattr(args, 'perturbation', 'N/A')
    val_keys_map = {
        'Step_MSE': 'Val Step MSE',
        'Step_R2': 'Val Step R2',
        'Global_open_MSE': 'Val Global Open MSE',
        'Global_open_R2': 'Val Global Open R2',
        'Global_closed_R2': 'Val Global Closed R2'
    }
    for k, v in val_keys_map.items():
        raw_data[v] = m_val.get(k)
    metric_map = {
        "step MSE": m_test.get('Step_MSE'),
        "step R2": m_test.get('Step_R2'),
        "local MSE": m_test.get('Local_MSE'),
        "global open MSE": m_test.get('Global_open_MSE'),
        "global open R2": m_test.get('Global_open_R2'),
        "global closed MSE": m_test.get('Global_closed_MSE'),
        "global closed R2": m_test.get('Global_closed_R2'),
        "recursivity": m_test.get('Recursivity'),
        "final val loss": train_results['val_history'][-1] if train_results['val_history'] else None,
        "iterations": len(train_results['train_history']),
        "total params": total_params,
        "q params": num_q_params,
        "c params": num_c_params
    }
    raw_data.update(metric_map)
    target_names = ["Surge Velocity", "Sway Velocity", "Yaw Rate", "Yaw Angle"]
    metric_suffixes = ["Step MSE", "Step R2", "Local MSE",
                       "Global open MSE", "Global open R2"]

    for tgt_space in target_names:
        tgt_under = tgt_space.replace(" ", "_")
        for m_suffix in metric_suffixes:
            m_under = m_suffix.replace(" ", "_")
            
            # Check for standard name AND delta-prefixed name
            key_standard = f"{tgt_under}_{m_under}"
            key_delta = f"delta_{tgt_under}_{m_under}"
            
            col_name_excel = f"{tgt_space} {m_suffix}"
            
            if key_standard in m_test:
                raw_data[col_name_excel] = m_test[key_standard]
            elif key_delta in m_test:
                raw_data[col_name_excel] = m_test[key_delta]
    final_column_order = [
        "date", "model_id", "run", "weight_selection","data", "data_n", "data_dt", 
        "features", "targets", "window_size", "horizon", "predict", "norm", 
        "reconstruct_train", "reconstruct_val", "model", "heads_config",  "head_number","encoding", "ansatz", "freeze_qnn", "use_hadamard", "trainable_encoding",
        "entangle", "reps", "map", "reorder", 
        "optimizer", "maxiter","iterations",  "tolerance", "batch_size", "learning_rate", "perturbation", "weights",
        "total params", "q params", "c params","final val loss", 
        "Val Step MSE", "Val Step R2", "Val Global Open MSE", "Val Global Open R2", "Val Global Closed R2",
        "step MSE", "step R2", 
        "local MSE", "global open MSE", "global open R2", "global closed MSE", "global closed R2", 
        "recursivity", "initialization",
    ]
    for tgt in target_names:
        for m in metric_suffixes:
            final_column_order.append(f"{tgt} {m}")
    ordered_row = {}
    for col in final_column_order:
        ordered_row[col] = raw_data.get(col, None)
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
            df_new['freeze'] = str(freeze).lower()
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
            cols_existing = list(df_existing.columns)
            full_order = final_column_order + [c for c in cols_existing if c not in final_column_order]
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
        if getattr(args, 'model', 'vanilla') == 'multihead':
            head_num = len(getattr(args, 'heads_config', []))
        else:
            head_num = 1
        print(f"{C_YELLOW}[Warning] Excel error: {e}. SAttempting CSV backup with retries...{C_RESET}")
        
        # Unique backup name per run to prevent overwriting
        backup_folder = os.path.join(logs_dir, "backups")
        os.makedirs(backup_folder, exist_ok=True)
        
        backup_name = os.path.join(backup_folder, f"backup_{timestamp}_{h_count}heads_run{getattr(args, 'run', 0)}.csv")
        
        
        # RETRY LOOP
        for attempt in range(5): 
            try:
                df_new.to_csv(backup_name, index=False)
                print(f"{C_GREEN}[Success] Backup saved to {backup_name} on attempt {attempt+1}{C_RESET}")
                return model_filename # Successfully exited
            except Exception as save_error:
                wait = random.uniform(2, 7) # Avoid synchronized retries
                print(f"Save attempt {attempt+1} failed ({save_error}). Retrying in {wait:.1f}s...")
                time.sleep(wait)
        
        print(f"{C_RED}CRITICAL: Could not save results for run {getattr(args, 'run', 0)} after 5 attempts.{C_RESET}")

    log_filename = "logs/experiment_log.txt"
    log_entry = f"[{timestamp}] {args.model:<10} {args.optimizer:<8}| F={len(args.features):<2} W={args.window_size:<2} H={args.horizon:<2} | Circuit: {str(final_encoding):<10} {str(final_ansatz):<56} {str(final_entangle):<36} reps={str(final_reps):<15} | MSE={m_test.get('Step_MSE', 0):.4f}\n"
    with open(log_filename, "a") as f: f.write(log_entry)
    
    print(f"\n[Logger] Model saved to {model_filename}")
    print(f"[Logger] Stats appended to {excel_filename}")
    return model_filename
def find_existing_experiment(current_args, models_root="models"):
    print(f"{C_BLUE}[Cache Search] Scanning {models_root} (including all subfolders)...{C_RESET}")
    all_pkls = glob.glob(os.path.join(models_root, "**", "*.pkl"), recursive=True)
    opt = str(getattr(current_args, 'optimizer', '')).lower()
    if opt != 'ridge':
        critical_keys = [
            'run', 'window_size', 'horizon', 'model', 'predict', 
            'optimizer', 'batch_size', 'maxiter', 'learning_rate', 
            'perturbation', 'tolerance', 'initialization', 'weights',
            'ansatz', 'entangle', 'reps', 'encoding', 'map', 'reorder', 
            'heads_config', 'select_features', 'freeze_qnn', 'trainable_encoding', 'use_hadamard', 'hidden_layer'
        ]   
    else:
        critical_keys = [
            'run', 'window_size', 'horizon', 'model', 'predict', 
            'optimizer', 'learning_rate', 'initialization', 'weights',
            'ansatz', 'entangle', 'reps', 'encoding', 'map', 'reorder', 
            'heads_config', 'select_features', 'freeze_qnn', 'trainable_encoding', 'use_hadamard', 'hidden_layer'
        ]

    for pkl_path in all_pkls:
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            saved_config = data.get('config', {})
            test_m = data.get('test_metrics') or data.get('final_eval_metrics') or {}
            
            weights = data.get('final_weights') or data.get('selected_weights')
            if weights is None:
                continue
            
            # History check: SPSA needs it, qelm (Ridge) does not
            history = data.get('train_history') or []
            if opt != 'ridge' and len(history) < 10:
                continue
            match = True
            
            for key in critical_keys:
                val_curr = getattr(current_args, key, None)
                val_saved = saved_config.get(key, None)
                s_curr = str(val_curr).replace(" ", "").replace("'", '"')
                s_save = str(val_saved).replace(" ", "").replace("'", '"')
                if isinstance(val_curr, float):
                    if round(val_curr, 5) != round(val_saved, 5):
                        match = False ; break
                elif s_curr != s_save:
                    match = False
                    break
            
            if match:
                print(f"{C_GREEN}[Cache Search] MATCH FOUND!{C_RESET}")
                print(f"  > Source: {pkl_path}")
                return pkl_path, data

        except Exception:
            continue
            
    print(f"{C_YELLOW}[Cache Search] No identical experiment found in models/ folder.{C_RESET}")
    return None, None
def save_classical_results(args, train_results, val_eval, test_eval, scalers, timestamp, selection_type="Unknown", excel_path=None):
    """
    Saves detailed results specifically for Classical Models (LSTM/RNN).
    Standardized to match Quantum experiment directory structure.
    """
    save_dir = getattr(args, 'save_dir', 'classical_baselines')
    models_dir = os.path.join("models", save_dir)
    logs_dir = os.path.join("logs", save_dir)
    figs_dir = os.path.join("figures", save_dir)

    for folder in [models_dir, logs_dir, figs_dir]:
        os.makedirs(folder, exist_ok=True)
    excel_filename = excel_path if excel_path else os.path.join(logs_dir, "classical_experiments_summary.xlsx")
    model_name = f"{timestamp}_classical_f{len(args.features)}_w{args.window_size}_h{args.horizon}_hidd{args.hidden_size}.pkl"
    model_filename = os.path.join(models_dir, model_name)

    selected_w = train_results.get('selected_weights', train_results['final_weights'])
    total_params = sum(p.numel() for p in selected_w.values())
    save_payload = {
        "config": vars(args),
        "best_weights": train_results.get('best_weights'), 
        "final_weights": train_results.get('final_weights'),
        "selected_weights": selected_w,
        "weight_selection_method": selection_type,
        "train_history": train_results['train_history'],
        "val_history": train_results['val_history'],
        "val_metrics": val_eval['metrics'],
        "test_metrics": test_eval['metrics'],
        "y_scaler": scalers[1],
        "x_scaler": scalers[0]
    }
    
    with open(model_filename, "wb") as f:
        pickle.dump(save_payload, f)

    # 3. Save Summary to Excel
    m_val, m_test = val_eval['metrics'], test_eval['metrics']
    try:
        dt_temp = datetime.datetime.strptime(timestamp, "%m-%d_%H-%M-%S")
        dt_object = dt_temp.replace(year=2026)
    except Exception:
        dt_object = timestamp

    ignore_keys = ['select_features', 'drop_features', 'save_plot', 'show_plot']

    def clean_val(v):
        if isinstance(v, bool): return str(v).lower()
        if isinstance(v, list): return str(v)
        return v

    def normalize_loaded_bools(val):
        if isinstance(val, (bool, np.bool_)): return str(val).lower()
        if isinstance(val, str):
            if val.upper() in ['TRUE', 'VERDADERO']: return 'true'
            if val.upper() in ['FALSE', 'FALSO']: return 'false'
        return val
    # Inside save_classical_results, before creating raw_data
    if getattr(args, 'model', '') == 'multihead' and hasattr(args, 'heads_config'):
        # Just like the quantum version, clean the list for Excel readability
        clean_heads = []
        for h in args.heads_config:
            clean_heads.append({
                'features': h.get('features'),
                'hidden': h.get('hidden_size', args.hidden_size)
            })
        heads_config_str = str(clean_heads)
    else:
        heads_config_str = str([{'features': map_names(args.features, reverse=True), 'hidden': args.hidden_size}])
    raw_data = {}
    for key, value in vars(args).items():
        if key not in ignore_keys:
            if key in ['features', 'targets']:
                short_list = map_names(value, reverse=True)
                raw_data[key] = clean_val(short_list)
            else:
                raw_data[key] = clean_val(value)
    # Add 'heads_config' and 'head_number' to the raw_data dictionary
    raw_data['heads_config'] = heads_config_str
    raw_data['head_number'] = len(args.heads_config) if getattr(args, 'model','') == 'multihead' else 1
    metrics_flat = {
        "date": dt_object,
        "model_id": os.path.basename(model_filename),
        "model": "classical_lstm",
        "weight_selection": selection_type,
        "Val Global Open MSE": m_val.get('Global_open_MSE'),
        "Val Global Open R2": m_val.get('Global_open_R2'),
        "Val Global Closed R2": m_val.get('Global_closed_R2'),
        "step MSE": m_test.get('Step_MSE'),
        "step R2": m_test.get('Step_R2'),
        "local MSE": m_test.get('Local_MSE'),
        "global open MSE": m_test.get('Global_open_MSE'),
        "global open R2": m_test.get('Global_open_R2'),
        "global closed MSE": m_test.get('Global_closed_MSE'),
        "global closed R2": m_test.get('Global_closed_R2'),
        "final val loss": train_results['val_history'][-1] if train_results['val_history'] else None,
        "total params": total_params,
        "iterations": len(train_results['train_history']),
        "recursivity": m_test.get('Recursivity')
    }
    raw_data.update(metrics_flat)

    # Per-Target Mapping
    target_names = ["Surge_Velocity", "Sway_Velocity", "Yaw_Rate", "Yaw_Angle"]
    metric_types = ["Step_MSE", "Step_R2", "Global_open_R2", "Global_closed_R2"]
    
    for tgt in target_names:
        for m_type in metric_types:
            key_in_dict = f"{tgt}_{m_type}"
            if key_in_dict in m_test:
                col_name = key_in_dict.replace("_", " ") 
                raw_data[col_name] = m_test[key_in_dict]
        
    column_order = [
        "date", "model_id","config_file", "save_folder",  "excel_path", "save_dir", "indices","run", "weight_selection", "data", "features", "targets", "predict", "norm", "reconstruct_train", "reconstruct_val",
        
        "window_size", "horizon", "model", "head_number","heads_config","hidden_size", "layers", "total params", "weights",

        "optimizer", "maxiter", "learning_rate", "batch_size", "perturbation",  "tolerance", "initialization","iterations", "features_resolved",
        "final val loss", "Val Global Open R2", "Val Global Closed R2", "step R2", "global open R2", "global closed R2"
    ]
    
    ordered_row = {col: raw_data.get(col, None) for col in column_order}
    for k, v in raw_data.items():
        if k not in ordered_row: ordered_row[k] = v

    df_new = pd.DataFrame([ordered_row])

    if os.path.exists(excel_filename):
        try:
            df_existing = pd.read_excel(excel_filename)
            df_existing = df_existing.map(normalize_loaded_bools)
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
            df_final.to_excel(excel_filename, index=False)
        except Exception as e:
            print(f"[Warning] Excel error: {e}. Saving to CSV backup.")
            df_new.to_csv(os.path.join(logs_dir, f"backup_{timestamp}.csv"), index=False)
    else:
        df_new.to_excel(excel_filename, index=False)

    print(f"\n[Logger] Classical model saved to {model_filename}")
    return model_filename
def load_experiment_results(filepath, final = True):
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
    if 'test_metrics' in data:
        m_final = data['test_metrics']
    else:
        m_final = data.get('final_eval_metrics', data.get('best_eval_metrics', {}))
    if config.get('model') is None:
        if 'classical' in filepath.lower() or 'hidden_size' in config:
            config['model'] = 'classical_lstm'
        else:
            config['model'] = 'N/A'
    def resolve_multi_val(values_list):
        if not values_list: return 'N/A'
        if all(str(x) == str(values_list[0]) for x in values_list):
            return str(values_list[0]) 
        return "|".join(map(str, values_list))
    is_multi = config.get('model') == 'multihead'
    heads = config.get('heads_config', [])
    if is_multi and isinstance(heads, list):
        summary_params = {
            'reps': resolve_multi_val([h.get('reps', config.get('reps')) for h in heads]),
            'ansatz': resolve_multi_val([h.get('ansatz', config.get('ansatz')) for h in heads]),
            'entangle': resolve_multi_val([h.get('entangle', config.get('entangle')) for h in heads]),
            'encoding': resolve_multi_val([h.get('encoding', config.get('encoding')) for h in heads]),
            'map': resolve_multi_val([h.get('map', config.get('map')) for h in heads])
        }
    else:
        summary_params = {k: str(config.get(k, 'N/A')) for k in ['reps', 'ansatz', 'entangle', 'encoding', 'map']}
    print("\n" + "="*120)
    print(f"EXPERIMENT SUMMARY {'(MULTI-HEAD)' if is_multi else ''}")
    print("="*120)
    fname = os.path.basename(filepath)
    try:
        parts = fname.split('_')
        timestamp = f"{parts[0]}_{parts[1]}"
    except:
        timestamp = "Unknown"
        
    print(f"Timestamp: {timestamp}")
    selection_method = data.get('weight_selection_method', 'Unknown')
    print(f"Weights Selected: {selection_method}")
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
        if k in summary_params:
            val = summary_params[k]
        else:
            val = config.get(k, 'N/A')
            if k == 'features' and isinstance(val, list): val = f"{len(val)} features"
        print(f"{k:<15}: {val}")
    if is_multi:
        print(f"{'head_number':<15}: {len(heads)}")
    # --- Performance Table (Aggregate) ---
    print("\n--- Aggregate Performance (Final Weights) ---")
    
    if m_final:
        def get_fmt(metrics, key):
            val = metrics.get(key)
            if val is None: return "N/A"
            return f"{val:.6f}" if isinstance(val, (int, float)) else str(val)
        print(f"Step MSE: {get_fmt(m_final, 'Step_MSE'):<12} | Step R2: {get_fmt(m_final, 'Step_R2'):<12} | Local MSE: {get_fmt(m_final, 'Local_MSE')}")
        print("-" * 120)
        print(f"Global OPEN   -> MSE: {get_fmt(m_final, 'Global_open_MSE'):<10} | R2: {get_fmt(m_final, 'Global_open_R2')}")
        print(f"Global CLOSED -> MSE: {get_fmt(m_final, 'Global_closed_MSE'):<10} | R2: {get_fmt(m_final, 'Global_closed_R2')}")

        # --- DETAILED PER-TARGET TABLE ---
        print("\n--- Detailed Breakdown per Target (All Phases) ---")
        headers = ["TARGET", "Step MSE", "Step R2", "Loc MSE", "Open MSE", "Open R2", "Clos MSE", "Clos R2"]
        header_str = "{:<16} | {:<9} {:<9} | {:<9} | {:<9} {:<9} | {:<9} {:<9}".format(*headers)
        print("-" * len(header_str))
        print(header_str)
        print("-" * len(header_str))

        base_names = ["Surge_Velocity", "Sway_Velocity", "Yaw_Rate", "Yaw_Angle"]        
        for tgt in base_names:
            def t_get(metric_suffix):
                key = f"{tgt}_{metric_suffix}"
                val = m_final.get(key)
                if val is None:
                    key2 = f"delta_{tgt}_{metric_suffix}"
                    val = m_final.get(key2)
                if val is None: return "N/A"
                return f"{val:.5f}" if isinstance(val, (int, float)) else str(val)

            row_vals = [
                tgt.replace("_", " "),
                t_get("Step_MSE"),
                t_get("Step_R2"),
                t_get("Local_MSE"),
                t_get("Global_open_MSE"), t_get("Global_open_R2"), 
                t_get("Global_closed_MSE"), t_get("Global_closed_R2"),
            ]

            print( "{:<16} | {:<9} {:<9} | {:<9} | {:<9} {:<9} | {:<9} {:<9}".format(*row_vals))
            
        print("-" * len(header_str))

    else:
        print("Metric data missing in file.")

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
    fig, ax1 = plt.subplots(figsize=(12, 8))
    iterations = range(1, len(train_loss) + 1)
    
    ax1.set_xlabel(r"$\mathit{Iterations}$", **label_style)
    ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="--", alpha=0.5, linewidth=1.0)
    
    use_dual_axis = (args.reconstruct_train != args.reconstruct_val)
    c_train = '#1f77b4'
    c_val = '#ff7f0e'  
    
    if use_dual_axis:
        ylabel_train = r"$\mathit{Train\ MSE\ (Reconstructed)}$" if args.reconstruct_train else r"$\mathit{Train\ MSE\ (Normalized)}$"
        ax1.set_ylabel(ylabel_train, color=c_train, **label_style)
        ax1.plot(iterations, train_loss, color=c_train, alpha=0.6, linewidth=2.0, label='Train Loss')
        ax1.tick_params(axis='y', labelcolor=c_train)
        t_min, t_max = min(train_loss), max(train_loss)
        ax1.set_ylim([t_min * 0.5, t_max * 2.0])
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
    plt.title(f"Convergence Plot: {args.optimizer.upper()} {title_suffix}", pad=20, **title_style)
    
    fig.tight_layout()
    
    if filename and args.save_plot:
        _ensure_dir_exists(filename)
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
    true_path = data['true_backbone']
    pred_path = data['local']['pred_path'] 
    time_steps = np.arange(len(true_path))

    if horizons is None:
        horizons = [args.horizon]
    elif isinstance(horizons, (int, float)):
        horizons = [int(horizons)]

    max_h = pred_path.shape[1]
    horizons = [min(h, max_h) for h in horizons]
    horizons.sort(reverse=True)

    num_targets = len(args.targets)
    cols = 2
    rows = math.ceil(num_targets / cols)
    
    fig = plt.figure(figsize=(7 * cols, 5 * rows))
    gs = gridspec.GridSpec(rows, cols, hspace=0.3, wspace=0.25)
    unit_map = {"Surge Velocity": "m/s", "Sway Velocity": "m/s", "Yaw Rate": "rad/s", "Yaw Angle": "rad"}
    
    targets = []
    for i, t_name in enumerate(args.targets):
        u = unit_map.get(t_name, "")
        targets.append({"name": t_name, "unit": u, "idx": i})
    
    axes = []
    for i, target in enumerate(targets):
        row, col = i // 2, i % 2
        ax = fig.add_subplot(gs[row, col])
        ax.plot(time_steps, true_path[:, target['idx']], 'k-', linewidth=1.5, alpha=0.3, label='True Path')
        
        ax.set_title(target['name'], **subtitle_style)
        ax.set_ylabel(rf"$\mathit{{{target['name'].split()[0]}}}$ [{target['unit']}]", **label_style)
        if row == 1: ax.set_xlabel(r"$\mathit{Time\ Step}$", **label_style)
        ax.grid(True, linestyle='--', alpha=0.5, linewidth=1.0)
        axes.append(ax)
    num_samples = pred_path.shape[0]
    if 'colors' in globals() and len(colors) > 0:
        color_list = colors
    else:
        color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i in range(0, num_samples, step_interval):
        for h_idx, h in enumerate(horizons):
            t_indices = np.arange(i, i + h + 1)
            if t_indices[-1] >= len(time_steps): continue
            curr_true = true_path[i].reshape(1, -1)
            curr_pred = pred_path[i, :h, :]
            branch_data = np.vstack([curr_true, curr_pred]) # (h+1, 4)
            c = color_list[(h_idx + 1) % len(color_list)]

            for idx, ax in enumerate(axes):
                lbl = f'Pred (H={h})' if (i == 0 and idx == 0) else ""
                ax.plot(t_indices, branch_data[:, idx], color=c, linestyle='-', linewidth=1.5, alpha=0.8, label=lbl if idx==0 else "")
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
    horizon_steps.sort()
    true_data = data['true_backbone'] 
    raw_pred = data['global'][loop]['pred_path']
    num_targets = len(args.targets)
    cols = 2
    rows = math.ceil(num_targets / cols)
    
    fig = plt.figure(figsize=(7*cols, 5*rows))
    gs = gridspec.GridSpec(rows, cols, hspace=0.3, wspace=0.25)
    
    unit_map = {"Surge Velocity": "m/s", "Sway Velocity": "m/s", "Yaw Rate": "rad/s", "Yaw Angle": "rad"}
    targets = [{"name": t, "unit": unit_map.get(t, ""), "idx": i} for i, t in enumerate(args.targets)]
    axes = []
    for i, target in enumerate(targets):
        row, col = i // 2, i % 2
        ax = fig.add_subplot(gs[row, col])
        time_steps_true = np.arange(len(true_data))
        ax.plot(time_steps_true, true_data[:, target['idx']], 'k-', linewidth=1.5, alpha=0.4, label='True')
        
        for h_idx, h in enumerate(horizon_steps):
            k = h - 1
            if k >= raw_pred.shape[1]: 
                continue 
            if k == 0:
                pred_seq = raw_pred[:, k, :]
                true_seq_aligned = true_data
                start_t = 0
            else:
                pred_seq = raw_pred[:-k, k, :]
                true_seq_aligned = true_data[k:]
                start_t = k 

            min_len = min(len(true_seq_aligned), len(pred_seq))
            pred_seq = pred_seq[:min_len]
            t_axis = np.arange(start_t, start_t + min_len)
            if 'colors' in globals(): c = colors[h_idx % len(colors)]
            else: c = ['#D62728', '#1f77b4', '#2ca02c'][h_idx % 3]
            ax.plot(t_axis, pred_seq[:, target['idx']], '--', color=c, 
                    linewidth=1.8, alpha=0.9, label=f'Pred (k={h})')
        ax.set_title(target['name'], **subtitle_style)
        ax.set_ylabel(rf"$\mathit{{{target['name'].split()[0]}}}$ [{target['unit']}]", **label_style)
        if row == 1: ax.set_xlabel(r"$\mathit{Time\ Step}$", **label_style)
        ax.grid(True, linestyle='--', alpha=0.5, linewidth=1.0)
        if '_force_ticks_font' in globals(): _force_ticks_font(ax)
        axes.append(ax)
    axes[0].legend(prop=legend_prop if 'legend_prop' in globals() else None, loc='best')
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
    if not getattr(args, 'save_plot', True):
        return # Exit immediately if saving is disabled
    if mode not in data: return
    if filename:
        _ensure_dir_exists(filename)
    if mode == 'local': pred_obj = data[mode]
    else: pred_obj = data[mode][loop]
    pred_deltas_all_h = pred_obj['pred_deltas_denorm']
    true_deltas_all_h = data['true_deltas_denorm']
    pred_path_all_h = pred_obj['pred_path'] 
    true_path_all_h = data['true_path']
    true_backbone = data['true_backbone']
    num_points = min(pred_deltas_all_h.shape[0], true_deltas_all_h.shape[0])
    pred_deltas_all_h = pred_deltas_all_h[:num_points]
    true_deltas_all_h = true_deltas_all_h[:num_points]
    pred_path_all_h = pred_path_all_h[:num_points]
    true_path_all_h = true_path_all_h[:num_points]
    true_path_flat = true_backbone[:num_points]         
    time_steps = np.arange(num_points)

    unit_map = {"Surge Velocity": "m/s", "Sway Velocity": "m/s", "Yaw Rate": "rad/s", "Yaw Angle": "rad"}
    targets = [{"name": t, "unit": unit_map.get(t, ""), "idx": i} for i, t in enumerate(args.targets)]
    if filename:
        _ensure_dir_exists(filename)
    for tgt in targets:
        idx = tgt['idx']
        t_name = tgt['name']
        t_unit = tgt['unit']
        raw_step_errors = np.abs(true_deltas_all_h[:, :, idx] - pred_deltas_all_h[:, :, idx])
        raw_pos_errors  = np.abs(true_path_all_h[:, :, idx] - pred_path_all_h[:, :, idx])
        max_h = raw_step_errors.shape[1]
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
        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)
        ax_top_left = plt.subplot(gs[0])
        ax_top_right = ax_top_left.twinx()

        num_lines = len(tasks)
        if num_lines == 1:
            colors_acc = ['#D62728'] 
            colors_net = ['#1F77B4'] 
        else:
            colors_acc = [cm.Reds(x) for x in np.linspace(0.5, 1.0, num_lines)]
            colors_net = [cm.Blues(x) for x in np.linspace(0.5, 1.0, num_lines)]

        lines_legend = []
        for i, (label, s_err, p_err) in enumerate(tasks):
            accumulated_error = np.cumsum(s_err)
            l1, = ax_top_left.plot(time_steps, accumulated_error, color=colors_acc[i], alpha=0.9, linewidth=2.5, label=f'Acc Error ({label})')
            l2, = ax_top_right.plot(time_steps, p_err, color=colors_net[i], alpha=0.7, linewidth=2.0, linestyle='--', label=f'Net Error ({label})')
            
            lines_legend.extend([l1, l2])
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
        ax_bot = plt.subplot(gs[1])
        c_path = '#2CA02C' 
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
            clean_name = t_name.replace(" ", "_")
            
            if horizon_mode != ['mean'] and horizon_mode != ['max']:
                h_suffix = "H_" + "_".join([str(h) for h in horizon_mode_list])
            else:
                h_suffix = horizon_mode_list[0]

            if mode == 'local': 
                save_path = f"{filename}_{clean_name}_{mode}_{h_suffix}.png"
            else: 
                save_path = f"{filename}_{clean_name}_{mode}_{loop}_{h_suffix}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
    pred_path = pred_obj['pred_path'] 
    true_path = data['true_path']
    num_samples = min(pred_path.shape[0], true_path.shape[0])
    abs_error = np.abs(true_path[:num_samples] - pred_path[:num_samples])
    horizon_steps = abs_error.shape[1]
    num_targets = len(args.targets)
    cols = 2
    rows = math.ceil(num_targets / cols)
    fig = plt.figure(figsize=(7*cols, 5*rows))
    gs = gridspec.GridSpec(rows, cols, hspace=0.3, wspace=0.25)
    unit_map = {"Surge Velocity": "m/s", "Sway Velocity": "m/s", "Yaw Rate": "rad/s", "Yaw Angle": "rad"}
    targets = [{"name": t, "unit": unit_map.get(t, ""), "idx": i} for i, t in enumerate(args.targets)]
    for i, tgt in enumerate(targets):
        row, col = i // 2, i % 2
        ax = fig.add_subplot(gs[row, col])
        idx = tgt['idx']
        feature_error = abs_error[:, :, idx] 
        plot_data = [feature_error[:, k] for k in range(horizon_steps)]
        step_means = np.mean(feature_error, axis=0)
        box = ax.boxplot(plot_data, patch_artist=True, showfliers=False, widths=0.6,
                         medianprops=dict(linewidth=2.0, color='#000080')) # Navy Median
        c_face = '#ADD8E6' # Light Blue
        c_edge = '#1F77B4' # Dark Blue
        for patch in box['boxes']:
            patch.set_facecolor(c_face)
            patch.set_edgecolor(c_edge)
            patch.set_alpha(0.7)
        x_pos = np.arange(1, horizon_steps + 1)
        ax.plot(x_pos, step_means, marker='D', color='#D62728', linestyle='None', 
                markersize=6, label='Mean Error')
        ax.set_title(tgt['name'], **subtitle_style)
        ax.set_ylabel(rf"$\mathit{{Abs\ Error}}$ [{tgt['unit']}]", **label_style)
        if row == 1: 
            ax.set_xlabel(r"$\mathit{Horizon\ Step}$", **label_style)
        ax.grid(True, linestyle='--', alpha=0.5)
        if '_force_ticks_font' in globals(): _force_ticks_font(ax)


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