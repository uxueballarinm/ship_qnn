import yaml
import os

# --- 1. Define the Shared Base Configuration ---
# These settings are common to all experiments
base_config = {
    'window_size': 5,
    'select_features': ['wv', 'sv', 'yr', 'ya', 'rarad'],
    'model': 'multihead',
    'predict': 'motion',
    'optimizer': 'cobyla',
    'maxiter': 2000,
    'save_plot': True
}

# --- 2. Helper Function to Build Heads ---
def create_heads(ansatz_type, head2_map_strategy):
    """
    Creates the heads_config list based on ansatz and mapping strategy.
    
    Head 1 (Surge) is kept constant.
    Head 2 (Turner) varies based on the map.
    """
    
    # Head 1: Surge (1 Feature: 'wv')
    # 1 Feature -> 1 U-Gate -> 3 Slots. 
    # Standard Map: [0, -1, -1]
    head1 = {
        'features': ['wv'],
        'output_dim': 1,
        'reps': 3,
        'encoding': 'compact',
        'ansatz': ansatz_type,
        'entangle': 'reverse_linear',
        'reorder': False,
        'map': [0, -1, -1]
    }

    # Head 2: Turner (4 Features: 'sv', 'yr', 'ya', 'rarad')
    # 4 Features -> 2 U-Gates -> 6 Slots.
    # Indices relative to this head: 0=sv, 1=yr, 2=ya, 3=rarad
    
    if head2_map_strategy == "standard":
        # Order: [sv, yr, ya, rarad, pad, pad]
        h2_map = [0, 1, 2, 3, -1, -1]
        
    elif head2_map_strategy == "rudder_first":
        # Order: [rarad, sv, yr, ya, pad, pad]
        # Puts Rudder (3) in the very first rotation slot
        h2_map = [3, 0, 1, 2, -1, -1]
        
    elif head2_map_strategy == "rudder_last":
        # Order: [sv, yr, ya, pad, pad, rarad]
        # Puts Rudder (3) in the very last rotation slot
        h2_map = [0, 1, 2, -1, -1, 3]

    head2 = {
        'features': ['sv', 'yr', 'ya', 'rarad'],
        'output_dim': 3,
        'reps': 2,
        'encoding': 'compact',
        'ansatz': ansatz_type,
        'entangle': 'reverse_linear',
        'reorder': False,
        'map': h2_map
    }
    
    return [head1, head2]

# --- 3. Define the 3 Experiments ---

experiments = []

# Experiment 1: Baseline (EfficientSU2 + Standard Map)
exp1 = base_config.copy()
exp1['heads_config'] = create_heads(ansatz_type='efficientsu2', head2_map_strategy='standard')
experiments.append(("experiment_1_baseline.yml", exp1))

# Experiment 2: Map Importance Test (EfficientSU2 + Rudder First)
# Comparing Exp 1 vs Exp 2 tells you if Map Order matters.
exp2 = base_config.copy()
exp2['heads_config'] = create_heads(ansatz_type='efficientsu2', head2_map_strategy='rudder_first')
experiments.append(("experiment_2_map_variation.yml", exp2))

# Experiment 3: Ansatz Test (UGates + Standard Map)
# Comparing Exp 1 vs Exp 3 tells you if UGates is better than EfficientSU2.
exp3 = base_config.copy()
exp3['heads_config'] = create_heads(ansatz_type='ugates', head2_map_strategy='standard')
experiments.append(("experiment_3_ugates.yml", exp3))

# --- 4. Write Files ---
print("Generating configuration files...")

for filename, config in experiments:
    with open(filename, 'w') as f:
        # We wrap the config in a list because your main script 
        # expects a list of experiments (even if there is only 1)
        yaml.dump([config], f, default_flow_style=None, sort_keys=False)
    print(f" -> Created {filename}")

print("\nDone! You can now run these in separate terminals:")
print("python qnn_run_experiments_ship.py experiment_1_baseline.yml")
print("python qnn_run_experiments_ship.py experiment_2_map_variation.yml")
print("python qnn_run_experiments_ship.py experiment_3_ugates.yml")