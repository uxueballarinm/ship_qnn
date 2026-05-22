import yaml
import argparse
import os
import sys
from joblib import Parallel, delayed
from copy import deepcopy

# IMPORTANT: Change 'main_script' to the actual name of your file (e.g., train_qnn)
import qnn_and_qelm_train_model_complete as main_script 

def str2bool(v):
    """ Helper to handle boolean strings in argparse """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_base_args():
    """ 
    Creates the exact same parser as your main script.
    This ensures 'args' has every attribute the 'run' function needs.
    """
    parser = argparse.ArgumentParser()
    
    # 1. Logistics
    parser.add_argument('--config', type=str, required=True)    
    parser.add_argument('--indices', type=int, nargs=2, default=None, help="START and END index (e.g. 2600 3601)")    
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--save_dir', type=str, default="")
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--data', type=str, default="dataset")

    # 2. Features (Mutually Exclusive Group)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-select', '--select_features', type=str, default=['sv','wv', 'yr','ya','rarad'], nargs='+')
    group.add_argument('-drop', '--drop_features', type=str, nargs='+')

    # 3. Data Processing / Time Series
    parser.add_argument('-ws', '--window_size', type=int, default=5)
    parser.add_argument('-y', '--horizon', type=int, default=5)
    parser.add_argument('--predict', type=str, default='motion')
    parser.add_argument('--custom_targets', type=str, nargs='+')
    parser.add_argument('--norm', type=str2bool, default=True)
    parser.add_argument('-rt', '--reconstruct_train', type=str2bool, default=False)
    parser.add_argument('-rv', '--reconstruct_val', type=str2bool, default=False)

    # 4. Circuit Architecture
    parser.add_argument('--map', type=str, nargs='+')
    parser.add_argument('--reorder', type=str2bool, default=False)
    parser.add_argument('--encoding', type=str, default='compact')
    parser.add_argument('--entangle', type=str, default='reverse_linear')
    parser.add_argument('--ansatz', type=str, default='ugates')
    parser.add_argument('--trainable_encoding', type=str2bool, default=False)
    parser.add_argument('--reps', type=int, default=3)
    parser.add_argument('-init', '--initialization', type=str, default='uniform')
    parser.add_argument('--model', type=str, default='vanilla')
    parser.add_argument('--heads_config', default=None)

    # 5. Optimization
    parser.add_argument('-opt','--optimizer', type=str, default='cobyla')
    parser.add_argument('--maxiter', type=int, default=10000)
    parser.add_argument('-tol', '--tolerance', type=float, default=None)
    parser.add_argument('-lr','--learning_rate', type=float, default=0.01)
    parser.add_argument('-p','--perturbation', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--weights', type=str, default="[1.0, 1.0, 1.0, 1.0]")

    # 6. Logistics / Flags
    parser.add_argument('--show_plot', type=str2bool, default=False)
    parser.add_argument('--save_plot', type=str2bool, default=True)
    parser.add_argument('--check_existing', type=str2bool, default=False)
    parser.add_argument('--save_in_excel', type=str2bool, default=False)
    parser.add_argument('--use_hadamard', type=str2bool, default=False)
    parser.add_argument('--freeze_qnn', type=str2bool, default=False)
    parser.add_argument('--hidden_layer', type=int, default=0)

    # We use parse_known_args([]) to initialize the object with default values
    return parser.parse_args()

def worker_task(idx, config_dict, base_args):
    """ Runs a single experiment from the YAML list """
    # Create a fresh clone of the defaults
    current_args = deepcopy(base_args)
    
    # Update only the attributes present in the YAML for this experiment
    for key, value in config_dict.items():
        setattr(current_args, key, value)
    
    # Specific overrides for local parallel execution
    current_args.indices = [idx + 1]
    current_args.show_plot = False # Crucial: prevents windows from popping up
    
    print(f"\n>>> [Queue] Launching Exp {idx+1} (Save Dir: {current_args.save_dir})")
    
    try:
        main_script.run(current_args)
        print(f"--- [Success] Exp {idx+1} finished. ---")
    except Exception as e:
        print(f"--- [Error] Exp {idx+1} failed: {e} ---")

def main():
    # SETTINGS
    YAML_FILE = "experiment_definitions/optimizer_study_2/1head_optimizer_study.yml" # <--- Change this
    WORKERS = 2                     # <--- Adjust based on RAM
    
    
    # Initialize the base argument template
    base_args = get_base_args()
    
    if not os.path.exists(YAML_FILE):
        print(f"Error: {YAML_FILE} not found.")
        return

    with open(YAML_FILE, 'r') as f:
        config_list = yaml.safe_load(f)

    if base_args.indices:
        start_num, end_val = base_args.indices
        
        # Convert 1-based user input to 0-based Python indexing
        # User 2600 -> Index 2599
        py_start = max(0, start_num - 1)
        
        # User 3601 -> Range(..., 3601) stops at 3600 (the 3601st item)
        py_end = min(len(config_list), end_val)
        
        target_indices = range(py_start, py_end)
        print(f"Filtering: Queueing experiments {start_num} to {end_val} (Total: {len(target_indices)})")
    else:
        target_indices = range(len(config_list))
        print(f"Queueing ALL {len(config_list)} experiments.")

    # Run the parallel loop
    Parallel(n_jobs=WORKERS)(
        delayed(worker_task)(i, config_list[i], base_args) for i in target_indices
    )

if __name__ == "__main__":
    main()