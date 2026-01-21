import os
import yaml

# ==========================================
# 1. CONFIGURATION
# ==========================================

OUTPUT_DIR = "experiment_definitions/split_by_first_feature"

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
    "ansatz": "efficientsu2",
    "select_features": ["wv", "sv", "yr", "ya", "rarad", "map"],
    # CRITICAL: Disable shuffling. 
    # The circuit will process features in the exact order listed below.
    "reorder": False,
    "select_features": ["wv", "sv", "yr", "ya", "rarad"]
}

# B. SPLIT BY FILE: 
SPLIT_BY_FILE = {
    "run": list(range(5))
}

# C. VARY WITHIN FILE:
# Logic: Run all seeds for Set 1, then all seeds for Set 2.
VARY_WITHIN_FILE = {
    "map": [
        [0, 1, 2, 4, 3, -1],  # Order 1: Kinematic
        [4, 2, 1, 0, 3, -1],  # Order 2: Steering
        [3, 0, 1, 2, 4, -1]   # Order 3: State-Space
    ]
}

OUTPUT_DIR = "experiment_definitions/fixed_orders_no_radeg"

# ==========================================
# 2. GENERATOR LOGIC
# ==========================================

def generate_yaml_files():
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Generating YAML files in '{OUTPUT_DIR}'...\n")

    # Group maps by their first element (index 0)
    # The keys will be: -1, 0, 1, 2, 3, 4
    grouped_maps = {}

    for map_config in ALL_MAPS:
        first_val = map_config[0]
        if first_val not in grouped_maps:
            grouped_maps[first_val] = []
        grouped_maps[first_val].append(map_config)

    count = 0
    
    # Iterate over the groups (sorted just for tidy filenames)
    for first_val in sorted(grouped_maps.keys()):
        
        # Define filename: e.g. "map_start_0.yml", "map_start_minus1.yml"
        # We replace -1 with 'neg1' or just keep it as is. 
        # Using string replacement for clarity if desired.
        val_str = str(first_val).replace("-1", "neg1")
        filename = f"map_start_{val_str}.yml"
        filepath = os.path.join(OUTPUT_DIR, filename)

        experiment_list = []
        
        # For every map in this group, create an experiment entry
        for specific_map in grouped_maps[first_val]:
            
            # Combine Fixed params + the specific map
            full_entry = {**FIXED_PARAMS}
            full_entry["map"] = specific_map
            
            experiment_list.append(full_entry)

        # Write to YAML
        with open(filepath, "w") as f:
            yaml.dump(experiment_list, f, default_flow_style=None, sort_keys=False)

        print(f" -> Created: {filename} ({len(experiment_list)} experiments)")
        count += 1

    print(f"\nDone! Generated {count} files containing {len(ALL_MAPS)} total experiments.")

if __name__ == "__main__":
    generate_yaml_files()