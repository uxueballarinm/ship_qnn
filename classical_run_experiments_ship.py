import yaml
import sys
import argparse
from copy import deepcopy

# Import the run function from the script you just created
# Assumes you saved the previous code as 'classical_run.py'
try:
    from classical_time_encoding_ship import run_classical
except ImportError:
    print("Error: Could not import 'run_classical' from 'classical_run.py'.")
    print("Make sure 'classical_run.py' is in the same folder.")
    sys.exit(1)

# --- CLASSICAL DEFAULTS ---
# These values will be used if parameters are missing in the YAML
DEFAULTS = {
    # Data & Task
    'data': "datasets\zigzag_11_11_ind_reduced_2_s.csv",
    'select_features': ['wv','sv','yr','ya','rarad'],
    'drop_features': None,   
    'window_size': 5,
    'horizon': 5,         
    'testing_fold': 3,
    'predict': 'motion',
    'norm': True,
    'reconstruct_train': False,
    'reconstruct_val': False, 
    
    # Classical Hyperparameters
    'model': 'classical',
    'hidden_size': 16,
    'layers': 1,
    'batch_size': 32,
    'learning_rate': 0.005,
    'optimizer': 'adam',
    'maxiter': 100,      
    'patience': 50,
    
    # Plotting & Misc
    'plot_mode': 'both', 
    'show_plot': False,
    'save_plot': True,  
    'run': 0,
    
    # Dummy args
    'ansatz': 'lstm',
    'entangle': 'none',
    'reps': 1,
    'target': ["wv","sv","yr","ya"],
}

class ExperimentArgs:
    """Helper to convert dict keys to object attributes (args.variable)"""
    def __init__(self, **entries):
        self.__dict__.update(entries)

def main():
    parser = argparse.ArgumentParser(description="Run Classical experiments from YAML config.")
    parser.add_argument('config_file', type=str, help="Path to YAML config file")
    args = parser.parse_args()
    
    # 1. Load YAML
    try:
        with open(f"{args.config_file}", 'r') as f:
            experiments = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: File '{args.config_file}' not found.")
        return
    except Exception as e:
        print(f"Error parsing YAML: {e}")
        return

    if not experiments:
        print("No experiments found in YAML.")
        return

    print(f"Loaded {len(experiments)} experiments.")

    # 2. Run Loop
    for i, exp_config in enumerate(experiments):
        print(f"\n" + "="*60)
        print(f"Running Experiment {i+1}/{len(experiments)}")
        print(f"Config: {exp_config}")
        print("="*60)

        # Merge Defaults with YAML config
        run_args_dict = deepcopy(DEFAULTS)
        
        for key, value in exp_config.items():
            if key in run_args_dict:
                run_args_dict[key] = value
            else:
                print(f"[Warning] Unknown key '{key}' in YAML. Adding it anyway.")
                run_args_dict[key] = value

        # Convert to Namespace object (like argparse returns)
        args_obj = ExperimentArgs(**run_args_dict)

        # Execute
        try:
            run_classical(args_obj)
        except Exception as e:
            print(f"!!! Experiment {i+1} Failed !!!")
            print(e)
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()