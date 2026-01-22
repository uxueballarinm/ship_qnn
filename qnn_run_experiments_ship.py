#python qnn_run_experiments.py config.yml
import yaml
import sys
from copy import deepcopy
import argparse

try:
    from qnn_time_encoding_ship import run 
except ImportError:
    print("Error: Could not import 'run'. Check the filename in batch_runner.py")
    sys.exit(1)

# --- DEFAULTS (Must match your argparse defaults) ---
DEFAULTS = {
    'data': "datasets\zigzag_11_11_ind_reduced_2_s.csv",
    'select_features': ['wv','sv','yr','ya','rarad'],
    'drop_features': None,
    'window_size': 5,
    'horizon': 5,          
    'testing_fold': 3,
    'predict': 'delta',
    'norm': True,
    'reconstruct_train': False,
    'reconstruct_val': False, 
    'reorder': False,
    'map': None,
    'encoding': 'compact',
    'entangle': 'reverse_linear',
    'ansatz': 'ugates',
    'reps': 3,
    'initialization': 'uniform',
    'model': 'vanilla',
    'optimizer': 'cobyla',
    'maxiter': 10000,
    'tolerance': None,
    'show_plot': False,
    'save_plot': True,   
    'run': 0,
    'target': ["wv","sv","yr","ya"],
}

class ExperimentArgs:
    """Helper to convert dict keys to object attributes (args.variable)"""
    def __init__(self, **entries):
        self.__dict__.update(entries)

def main():
    # --- 1. Parse Command Line Argument for Config File ---
    parser = argparse.ArgumentParser(description="Run a batch of QNN experiments from a YAML file.")
    parser.add_argument("config_file", type=str, help="Path to the .yml configuration file")
    
    # This reads the arguments passed to python batch_runner.py
    args = parser.parse_args()
    config_file = f"{args.config_file}"

    # --- 2. Load YAML ---
    try:
        with open(config_file, 'r') as f:
            experiments = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: File '{config_file}' not found.")
        return
    except Exception as e:
        print(f"Error parsing YAML: {e}")
        return

    print(f"Loaded {len(experiments)} experiments.")

    for i, exp_config in enumerate(experiments):
        print(f"\n" + "="*50)
        print(f"Running Experiment {i+1}/{len(experiments)}")
        print(f"Config: {exp_config}")
        print("="*50)

        # 1. Start with defaults
        current_args_dict = deepcopy(DEFAULTS)

        # 2. Update with YAML values
        for key, value in exp_config.items():
            if key in current_args_dict:
                current_args_dict[key] = value
            else:
                print(f"[Warning] Key '{key}' not found in defaults. It will be added but might be ignored by your code.")
                current_args_dict[key] = value

        # 3. Convert to object (Namespace)
        args_obj = ExperimentArgs(**current_args_dict)

        # 4. Run your existing function
        try:
            run(args_obj)
        except Exception as e:
            print(f"!!! Experiment {i+1} Failed !!!")
            print(e)

if __name__ == "__main__":
    main()