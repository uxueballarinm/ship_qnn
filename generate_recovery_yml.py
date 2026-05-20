import pandas as pd
import yaml
import os
import ast
from itertools import permutations

def get_abs_path(path):
    abs_p = os.path.abspath(path)
    return '\\\\?\\' + abs_p if os.name == 'nt' and not abs_p.startswith('\\\\?\\') else abs_p

def generate_exact_recovery():
    # --- 1. SETTINGS ---
    excel_path = r"logs\experiments_systematic\qrc\1_head_model_massive_search\1_head_model_massive_search.xlsx"
    output_yml = r"experiment_definitions/massive_map_study/1_head_recovery_32_missing.yml"
    
    # How many seeds were you SUPPOSED to have per map? (e.g., 0 to 9)
    # If you only ran 1 seed, use range(1). If you ran 10, use range(10).
    expected_seeds = range(30) 
    
    # The pool of indices used for permutations
    pool = [0, 1, 2, 3, 4, -1]
    
    # --- 2. GENERATE ALL POSSIBLE (MAP, SEED) COMBINATIONS ---
    print("Generating theoretical truth (720 maps x seeds)...")
    all_maps = list(permutations(pool))
    
    # The 'theoretical_truth' is a set of tuples: ((map_tuple), seed)
    theoretical_truth = set()
    for m in all_maps:
        for s in expected_seeds:
            theoretical_truth.add((tuple(m), s))
            
    print(f"Total expected experiments: {len(theoretical_truth)}")

    # --- 3. LOAD EXCEL ---
    if not os.path.exists(get_abs_path(excel_path)):
        print(f"File not found: {excel_path}")
        return

    df = pd.read_excel(get_abs_path(excel_path), sheet_name="Sheet1")
    print(f"Loaded {len(df)} rows from Excel.")

    # --- 4. EXTRACT WHAT WE ACTUALLY HAVE ---
    actual_results = set()
    
    for idx, row in df.iterrows():
        try:
            # 1. Get the seed
            seed = int(row['run'])
            
            # 2. Get the map from heads_config string
            # heads_config usually looks like: "[{'features': [...], 'map': [0,1,2,3,4,-1], ...}]"
            raw_config = row['heads_config']
            config_list = ast.literal_eval(raw_config)
            m_tuple = tuple(config_list[0]['map'])
            
            actual_results.add((m_tuple, seed))
        except Exception as e:
            continue

    # --- 5. THE MISSING SEARCH ---
    missing = theoretical_truth - actual_results
    print(f"\n--- ANALYSIS ---")
    print(f"Completed: {len(actual_results)}")
    print(f"Missing:   {len(missing)}")

    # Sanity Check by Seed
    if missing:
        missing_df = pd.DataFrame(list(missing), columns=['map', 'run'])
        print("\nBreakdown of missing runs per seed:")
        print(missing_df['run'].value_counts().sort_index())

    # --- 6. GENERATE RECOVERY YAML ---
    if not missing:
        print("\nNo missing experiments found. You are 100% complete!")
        return

    recovery_exps = []
    for m_tuple, seed in sorted(list(missing), key=lambda x: (x[1], x[0])):
        exp = {
            "window_size": 5,
            "data": "data/reduce_row_number_absolutes",
            "save_dir": "efficientsu2/1_head_model_massive_search_RECOVERY",
            "model": "multihead",
            "predict": "motion",
            "optimizer": "ridge",
            "initialization": "identity",
            "run": seed,
            "heads_config": [{
                "features": ["sv", "wv", "yr", "ya", "rarad"],
                "output_dim": 4,
                "map": list(m_tuple),
                "reps": 1,
                "ansatz": "efficientsu2",
                "encoding": "serial",
                "entangle": "linear"
            }]
        }
        recovery_exps.append(exp)

    os.makedirs(os.path.dirname(get_abs_path(output_yml)), exist_ok=True)
    with open(get_abs_path(output_yml), 'w') as f:
        yaml.dump(recovery_exps, f, default_flow_style=False, sort_keys=False)

    print(f"\nSUCCESS: Created {output_yml} with {len(recovery_exps)} runs.")

if __name__ == "__main__":
    generate_exact_recovery()