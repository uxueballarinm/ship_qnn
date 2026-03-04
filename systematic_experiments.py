import yaml
import os

# Define the 4 fundamental map blocks based on your uploaded examples
MAP_BLOCKS = {
    "surge_sway": [0, 1],
    "yaw_rate_angle": [2, 3],
    "rudder": [4],
    "bias": [-1],
    "surge": [0],
    "sway": [1],
    "yaw_rate": [2],
    "yaw_angle": [3],
}

# The 4 systematic permutations of feature re-uploading
MAP_COMBINATIONS = [
    MAP_BLOCKS["surge_sway"] + MAP_BLOCKS["rudder"] + MAP_BLOCKS["yaw_rate_angle"] + MAP_BLOCKS["bias"], 
    MAP_BLOCKS["rudder"] + MAP_BLOCKS["surge_sway"] + MAP_BLOCKS["yaw_rate_angle"] + MAP_BLOCKS["bias"], 
    MAP_BLOCKS["yaw_rate_angle"] + MAP_BLOCKS["rudder"] + MAP_BLOCKS["surge_sway"] + MAP_BLOCKS["bias"], 
    MAP_BLOCKS["rudder"] + MAP_BLOCKS["yaw_rate_angle"] + MAP_BLOCKS["surge_sway"] + MAP_BLOCKS["bias"],

]

SEEDS = list(range(10)) # Runs 0 through 9

def generate_experiment_base(run_id, save_dir):
    return {
        'window_size': 5,
        'data': "data/reduce_row_number_absolutes",
        'save_dir': save_dir,
        'select_features': ['sv', 'wv', 'yr', 'ya', 'rarad'],
        'model': 'multihead',
        'predict': 'motion',
        'optimizer': 'spsa',
        'maxiter': 4000,
        'batch_size': 256,
        'learning_rate': [0.1, 0.001],
        'perturbation': 0.15,
        'weights': [1.0, 1.0, 1.0, 1.0],
        'initialization': 'identity',
        'save_plot': False,
        'run': run_id,
    }

# --- 3-HEAD OPTION A: [Surge] | [Sway] | [YawRate + YawAngle] ---
# In this mode, Head 1 and 2 split the translation block
experiments_3h_A = []
for seed in SEEDS:
    for i, map_cfg in enumerate(MAP_COMBINATIONS):
        exp = generate_experiment_base(seed, "3_head_systematic_A")
        exp['heads_config'] = [
            {'output_dim': 1, 'map': map_cfg, 'reps': 1, 'ansatz': 'efficientsu2', 'entangle': 'linear'}, # SURGE
            {'output_dim': 1, 'map': map_cfg, 'reps': 1, 'ansatz': 'efficientsu2', 'entangle': 'linear'}, # SWAY
            {'output_dim': 2, 'map': map_cfg, 'reps': 1, 'ansatz': 'efficientsu2', 'entangle': 'linear'}  # YR + YA
        ]
        experiments_3h_A.append(exp)

# --- 3-HEAD OPTION B: [Surge + Sway] | [YawRate] | [YawAngle] ---
# In this mode, Head 2 and 3 split the rotational block
experiments_3h_B = []
for seed in SEEDS:
    for i, map_cfg in enumerate(MAP_COMBINATIONS):
        exp = generate_experiment_base(seed, "3_head_systematic_B")
        exp['heads_config'] = [
            {'output_dim': 2, 'map': map_cfg, 'reps': 1, 'ansatz': 'efficientsu2', 'entangle': 'linear'}, # SURGE + SWAY
            {'output_dim': 1, 'map': map_cfg, 'reps': 1, 'ansatz': 'efficientsu2', 'entangle': 'linear'}, # YAW RATE
            {'output_dim': 1, 'map': map_cfg, 'reps': 1, 'ansatz': 'efficientsu2', 'entangle': 'linear'}  # YAW ANGLE
        ]
        experiments_3h_B.append(exp)

# Saving
os.makedirs("systematic_configs", exist_ok=True)
with open("systematic_configs/3_head_OptionA.yml", "w") as f: yaml.dump(experiments_3h_A, f)
with open("systematic_configs/3_head_OptionB.yml", "w") as f: yaml.dump(experiments_3h_B, f)