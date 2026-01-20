import os
import yaml
import itertools

# ==========================================
# 1. CONFIGURATION
# ==========================================

# A. FIXED PARAMS:
# These values are identical for EVERY experiment in EVERY file.
FIXED_PARAMS = {
    "maxiter": 5000,                    
    "optimizer": "cobyla",
    "encoding": "compact",
    "model": "vanilla",
    "predict": "delta",
    "window_size": 5,
    "horizon": 5,                    
    "reconstruct_val": False,         
    "save_plot": True,
    "entangle": "reverse_linear",
    "ansatz": "efficientsu2"
}

# B. SPLIT BY FILE: 
# A separate YAML file will be created for every combination of these.
# The filename will be generated from these values (e.g., "ugates_circular.yml").
SPLIT_BY_FILE = {
    "reps": [1, 3, 5]
}

# C. VARY WITHIN FILE:
# These variations will be listed INSIDE each YAML file.
# e.g., one file will contain experiments for reps=1, reps=3, reps=5.
VARY_WITHIN_FILE = {
    "select_features": [["wv", "sv", "yr", "ya", "radeg", "rarad"],["dwv", "dsv", "dyr", "dya", "radeg", "rarad"]],
}

OUTPUT_DIR = "experiment_definitions/delta_or_position"

# ==========================================
# 2. GENERATOR LOGIC (Do not edit below)
# ==========================================

def get_combinations(param_dict):
    """
    Returns a list of dictionaries representing every combination 
    of parameters in the input dict.
    """
    if not param_dict:
        return [{}]
    
    keys = list(param_dict.keys())
    # Extract values lists: e.g. [ ["a", "b"], [1, 2] ]
    values_lists = list(param_dict.values())
    
    # Cartesian product
    combos = []
    for combination in itertools.product(*values_lists):
        combos.append(dict(zip(keys, combination)))
    return combos

def generate_yaml_files():
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Generating YAML files in '{OUTPUT_DIR}'...\n")

    # 1. Generate the outer configurations (Files)
    file_configs = get_combinations(SPLIT_BY_FILE)

    # 2. Generate the inner configurations (List entries)
    inner_configs = get_combinations(VARY_WITHIN_FILE)

    count = 0
    
    for file_conf in file_configs:
        # Create a filename based on the values of the file-splitting params
        # e.g. If SPLIT_BY_FILE has "ansatz" and "entangle", filename is "ugates_circular.yml"
        # We replace spaces or special chars if necessary, but usually raw values work fine.
        filename_parts = [str(v) for v in file_conf.values()]
        filename = "_".join(filename_parts) + ".yml"
        filepath = os.path.join(OUTPUT_DIR, filename)

        experiment_list = []

        # Combine Fixed + File Specific + Inner Specific
        for inner_conf in inner_configs:
            # Merge all dictionaries (Python 3.9+ syntax could use | operator)
            full_entry = {**FIXED_PARAMS, **file_conf, **inner_conf}
            experiment_list.append(full_entry)

        # Write to YAML
        with open(filepath, "w") as f:
            yaml.dump(experiment_list, f, default_flow_style=False, sort_keys=False)

        print(f" -> Created: {filename} ({len(experiment_list)} experiments)")
        count += 1

    print(f"\nDone! Generated {count} files containing {count * len(inner_configs)} total experiments.")

if __name__ == "__main__":
    generate_yaml_files()