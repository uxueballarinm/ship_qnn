# import pandas as pd
# import yaml
# import os
# import ast
# from itertools import permutations

# def get_abs_path(path):
#     abs_p = os.path.abspath(path)
#     return '\\\\?\\' + abs_p if os.name == 'nt' and not abs_p.startswith('\\\\?\\') else abs_p

# def generate_exact_recovery():
#     # --- 1. SETTINGS ---
#     excel_path = r"logs\experiments_systematic\qelm\1_head_model_massive_search\1_head_model_massive_search_compact_identity.xlsx"
#     output_yml = r"logs\experiments_systematic\qelm\1_head_model_massive_search\1_head_recovery_compact.yml"
    
#     # How many seeds were you SUPPOSED to have per map? (e.g., 0 to 9)
#     # If you only ran 1 seed, use range(1). If you ran 10, use range(10).
#     expected_seeds = range(30) 
    
#     # The pool of indices used for permutations
#     pool = [0, 1, 2, 3, 4, -1]
    
#     # --- 2. GENERATE ALL POSSIBLE (MAP, SEED) COMBINATIONS ---
#     print("Generating theoretical truth (720 maps x seeds)...")
#     all_maps = list(permutations(pool))
    
#     # The 'theoretical_truth' is a set of tuples: ((map_tuple), seed)
#     theoretical_truth = set()
#     for m in all_maps:
#         for s in expected_seeds:
#             theoretical_truth.add((tuple(m), s))
            
#     print(f"Total expected experiments: {len(theoretical_truth)}")

#     # --- 3. LOAD EXCEL ---
#     if not os.path.exists(get_abs_path(excel_path)):
#         print(f"File not found: {excel_path}")
#         return

#     df = pd.read_excel(get_abs_path(excel_path), sheet_name="Sheet1")
#     print(f"Loaded {len(df)} rows from Excel.")

#     # --- 4. EXTRACT WHAT WE ACTUALLY HAVE ---
#     actual_results = set()
    
#     for idx, row in df.iterrows():
#         try:
#             # 1. Get the seed
#             seed = int(row['run'])
            
#             # 2. Get the map from heads_config string
#             # heads_config usually looks like: "[{'features': [...], 'map': [0,1,2,3,4,-1], ...}]"
#             raw_config = row['heads_config']
#             config_list = ast.literal_eval(raw_config)
#             m_tuple = tuple(config_list[0]['map'])
            
#             actual_results.add((m_tuple, seed))
#         except Exception as e:
#             continue

#     # --- 5. THE MISSING SEARCH ---
#     missing = theoretical_truth - actual_results
#     print(f"\n--- ANALYSIS ---")
#     print(f"Completed: {len(actual_results)}")
#     print(f"Missing:   {len(missing)}")

#     # Sanity Check by Seed
#     if missing:
#         missing_df = pd.DataFrame(list(missing), columns=['map', 'run'])
#         print("\nBreakdown of missing runs per seed:")
#         print(missing_df['run'].value_counts().sort_index())

#     # --- 6. GENERATE RECOVERY YAML ---
#     if not missing:
#         print("\nNo missing experiments found. You are 100% complete!")
#         return

#     recovery_exps = []
#     for m_tuple, seed in sorted(list(missing), key=lambda x: (x[1], x[0])):
#         exp = {
#             "window_size": 5,
#             "data": "data/reduce_row_number_absolutes",
#             "save_dir": "massive_study_compact_identity",
#             "model": "multihead",
#             "predict": "motion",
#             "optimizer": "ridge",
#             "initialization": "identity",
#             "run": seed,
#             "heads_config": [{
#                 "features": ["sv", "wv", "yr", "ya", "rarad"],
#                 "output_dim": 4,
#                 "map": list(m_tuple),
#                 "reps": 1,
#                 "ansatz": "efficientsu2",
#                 "encoding": "compact",
#                 "entangle": "linear"
#             }]
#         }
#         recovery_exps.append(exp)

#     os.makedirs(os.path.dirname(get_abs_path(output_yml)), exist_ok=True)
#     with open(get_abs_path(output_yml), 'w') as f:
#         yaml.dump(recovery_exps, f, default_flow_style=False, sort_keys=False)

#     print(f"\nSUCCESS: Created {output_yml} with {len(recovery_exps)} runs.")

# if __name__ == "__main__":
#     generate_exact_recovery()
import os
import pandas as pd
import json

def get_best_mappings(head_count):
    if head_count == 1:
        full = [
            {'feats': ['sv', 'wv', 'yr', 'ya', 'rarad'], 'maps': [[-1, 4, 1, 0, 2, 3]], 'dims': [4]},
            {'feats': ['sv', 'wv', 'yr', 'ya', 'rarad'], 'maps': [[-1, 4, 1, 2, 3, 0]], 'dims': [4]},
            {'feats': ['sv', 'wv', 'yr', 'ya', 'rarad'], 'maps': [[1, 4, 0, 2, 3, -1]], 'dims': [4]}
        ]
        reduced = [
            {'feats': ['sv', 'wv', 'yr', 'ya', 'rarad'], 'maps': [[1, 4, 3, 2, 0, -1]], 'dims': [4]},
            {'feats': ['sv', 'wv', 'yr', 'ya', 'rarad'], 'maps': [[3, 0, 4, -1, 2, 1]], 'dims': [4]},
            {'feats': ['sv', 'wv', 'yr', 'ya', 'rarad'], 'maps': [[2, 3, 4, 0, 1, -1]], 'dims': [4]}
        ]
    elif head_count == 2:
        full = [
            {'feats': [['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad']], 'maps': [[2, 3, 4, 0, 1, -1], [1, 4, 0, 2, 3, -1]], 'dims': [2, 2]},
            {'feats':  [['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad']], 'maps': [[-1, 3, 1, 0, 2, 4], [-1, 4, 1, 0, 2, 3]], 'dims': [1, 3]},
            {'feats':  [['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad']], 'maps': [[0, 2, 3, -1, 1, 4], [3, 2, 0, -1, 1, 4]], 'dims': [1, 3]}
        ]
        reduced = [
            {'feats': [['sv', 'yr'], ['wv','yr', 'ya','rarad']], 'maps': [[0, 1, -1], [1, 3, -1, 2, 0, -1]], 'dims': [1, 3]},
            {'feats': [['sv', 'wv', 'rarad'], ['yr', 'ya','rarad']], 'maps': [[0, 1, 2], [0, 1, 2]], 'dims': [2, 2]},
            {'feats': [['sv', 'yr'], ['wv','yr', 'ya','rarad']], 'maps': [[0, 1, -1], [0, 3, -1, 1, 2, -1]], 'dims': [1, 3]}
        ]
    elif head_count == 3:
        full = [
            {'feats': [['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad'],  ['sv', 'wv','yr','ya', 'rarad']], 'maps': [[2, 3, 1, 0, 4, -1], [2, 3, 0, 1, 4, -1], [1, 4, 0, 2, 3, -1]], 'dims': [1, 1, 2]},
            {'feats': [['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad'],  ['sv', 'wv','yr','ya', 'rarad']], 'maps': [[4, 3, 1, 0, 2, -1], [2, 3, 0, 1, 4, -1], [1, 4, 0, 2, 3, -1]], 'dims': [1, 1, 2]},
            {'feats': [['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad'],  ['sv', 'wv','yr','ya', 'rarad']], 'maps': [[-1, 3, 1, 0, 2, 4], [-1, 1, 0, 2, 3, 4], [-1, 1, 0, 3, 2, 4]], 'dims': [1, 1, 2]}
        ]
        reduced = [
            {'feats': [['sv', 'yr'], ['wv', 'rarad'], ['yr', 'ya','rarad']], 'maps': [[0, 1, -1], [0, 1, -1], [0, 1, 2]], 'dims': [1, 1, 2]},
            {'feats': [['sv', 'rarad'], ['wv', 'rarad'], ['yr', 'ya','rarad']], 'maps': [[0, 1, -1], [0, 1, -1], [0, 1, 2]], 'dims': [1, 1, 2]},
            {'feats': [['sv', 'rarad'], ['wv', 'rarad'], ['yr', 'ya','rarad']], 'maps': [[0, 1, -1], [0, 1, -1], [0, 2, -1, 2, 1, -1]], 'dims': [1, 1, 2]}
        ]
    elif head_count == 4:
        full = [
            {'feats': [['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad']], 'maps': [[-1, 3, 1, 0, 2, 4], [-1, 1, 0, 2, 3, 4],[-1, 4, 1, 2, 3, 0], [-1, 0, 4, 1, 3, 2]], 'dims': [1, 1, 1, 1]},
            {'feats': [['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad']], 'maps': [[0, 2, 3, -1, 1, 4], [2, 1, 0, -1, 3, 4],[3, 2, 0, -1, 1, 4], [3, 2, 0, -1, 1, 4]], 'dims': [1, 1, 1, 1]},
            {'feats': [['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad']], 'maps': [[-1, 1, 3, 0, 2, 4], [-1, 0, 3, 1, 4, 2],[-1, 1, 0, 2, 3, 4], [-1, 1, 0, 3, 2, 4]], 'dims': [1, 1, 1, 1]},
        ]
        reduced = [
            {'feats': [['sv', 'yr'], ['wv', 'rarad'], ['yr', 'rarad'], ['ya', 'yr']], 'maps': [[0, 1, -1],[0, 1, -1],[0, 1, -1],[0, 1, -1]], 'dims': [1, 1, 1, 1]},
            {'feats': [['sv', 'rarad'], ['wv', 'rarad'], ['yr', 'rarad'], ['ya', 'rarad']], 'maps': [[0, 1, -1],[0, 1, -1],[0, 1, -1],[0, 1, -1]], 'dims': [1, 1, 1, 1]},
            {'feats': [['sv', 'yr'], ['yr', 'ya'], ['ya', 'yr'], ['ya', 'yr']], 'maps': [[0, 1, -1],[0, 1, -1],[0, 1, -1],[0, 1, -1]], 'dims': [1, 1, 1, 1]},
        ]
    return full + reduced

def clean_string_format(val):
    """Uniformizes string layouts by dropping spaces, braces, and shifting quotes."""
    if pd.isna(val) if isinstance(val, (str, float, int)) else False:
        return ""
    # Strip spaces, unified brackets/quotes, clean out formatting tags
    return str(val).replace(" ", "").replace("'", '"').replace("\n", "").replace("双", '"').strip()
import ast

def parse_heads_config(val):
    """Safely converts an Excel cell value into a clean Python list of dictionaries."""
    if pd.isna(val):
        return None
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            # Cleans up common Excel formatting quirks and converts string to list
            cleaned = val.strip().replace("双", '"').replace("'", '"')
            return ast.literal_eval(cleaned)
        except Exception:
            try:
                import json
                return json.loads(cleaned)
            except Exception:
                return None
    return None

def compare_configs(cfg_list_a, cfg_list_b):
    """Compares two lists of head configurations key-by-key, ignoring order and spacing."""
    if not cfg_list_a or not cfg_list_b:
        return False
    if len(cfg_list_a) != len(cfg_list_b):
        return False
        
    for h_a, h_b in zip(cfg_list_a, cfg_list_b):
        try:
            # Compare the critical elements that uniquely identify the topology structure
            if int(h_a.get('reps')) != int(h_b.get('reps')): return False
            if str(h_a.get('encoding')).strip().lower() != str(h_b.get('encoding')).strip().lower(): return False
            if int(h_a.get('output_dim')) != int(h_b.get('output_dim')): return False
            
            # Standardize and compare feature lists
            feat_a = [str(f).strip().lower() for f in h_a.get('features', [])]
            feat_b = [str(f).strip().lower() for f in h_b.get('features', [])]
            if feat_a != feat_b: return False
            
            # Standardize and compare mapping arrays
            map_a = str(h_a.get('map')).replace(" ", "")
            map_b = str(h_b.get('map')).replace(" ", "")
            if map_a != map_b: return False
            
        except Exception:
            return False
    return True

def generate_recovery_yaml(excel_path, output_yaml_path):
    print(f"Reading experimental logs from: {excel_path}")
    df_log = pd.read_excel(excel_path)
    print(f"Loaded {len(df_log)} data records.")

    # Parse and cache all Excel logs as actual Python objects for accurate lookups
    parsed_logs = []
    for _, row in df_log.iterrows():
        try:
            parsed_hc = parse_heads_config(row['heads_config'])
            if parsed_hc is None:
                continue
            parsed_logs.append({
                'head_number': int(row['head_number']),
                'run': int(row['run']),
                'heads_config': parsed_hc
            })
        except Exception:
            continue

    seeds = 10
    optimizer = 'spsa'
    initialization = 'identity'
    maxiter = 4000
    lr = "[0.1, 0.001]"
    save_dir = "study_qnn"
    
    yaml_lines = []
    total_scanned = 0
    total_missing = 0

    for head_num in [1, 2, 3, 4]:
        mappings = get_best_mappings(head_num)
        for map_config in mappings:
            for reps in [1, 3]:
                for encoding in ['serial', 'compact']:
                    
                    # Build the exact target heads_config list structure for this loop iteration
                    target_heads_list = []
                    for i in range(head_num):
                        f_item = map_config['feats'][i] if head_num > 1 else map_config['feats']
                        m_item = map_config['maps'][i] if head_num > 1 else map_config['maps'][0]
                        d_item = map_config['dims'][i] if head_num > 1 else map_config['dims'][0]
                        
                        head_dict = {
                            "features": f_item,
                            "output_dim": d_item,
                            "map": m_item,
                            "reps": reps,
                            "encoding": encoding,
                            "ansatz": "efficientsu2",
                            "entangle": "linear"
                        }
                        target_heads_list.append(head_dict)
                    
                    for run in range(seeds):
                        total_scanned += 1
                        
                        # Search through our parsed log entries
                        is_finished = False
                        for log_entry in parsed_logs:
                            if log_entry['head_number'] == head_num and log_entry['run'] == run:
                                if compare_configs(log_entry['heads_config'], target_heads_list):
                                    is_finished = True
                                    break
                        
                        if not is_finished:
                            total_missing += 1
                            yaml_lines.append(f"- window_size: 5")
                            yaml_lines.append(f"  data: data/reduce_row_number_absolutes")
                            yaml_lines.append(f"  save_dir: {save_dir}")
                            yaml_lines.append(f"  select_features: [sv, wv, yr, ya, rarad]")
                            yaml_lines.append(f"  model: multihead")
                            yaml_lines.append(f"  predict: motion")
                            yaml_lines.append(f"  optimizer: {optimizer}")
                            yaml_lines.append(f"  maxiter: {maxiter}")
                            yaml_lines.append(f"  batch_size: 256")
                            yaml_lines.append(f"  learning_rate: {lr}")
                            yaml_lines.append(f"  perturbation: 0.15")
                            yaml_lines.append(f"  weights: [1.0, 1.0, 1.0, 1.0]")
                            yaml_lines.append(f"  initialization: {initialization}")
                            yaml_lines.append(f"  save_plot: false")
                            yaml_lines.append(f"  run: {run}")
                            yaml_lines.append(f"  check_existing: true")
                            yaml_lines.append(f"  save_in_excel: true")
                            yaml_lines.append(f"  heads_config:")
                            
                            for h_cfg in target_heads_list:
                                yaml_lines.append(f"  - features: {str(h_cfg['features']).replace(' ', '')}")
                                yaml_lines.append(f"    output_dim: {h_cfg['output_dim']}")
                                yaml_lines.append(f"    map: {str(h_cfg['map']).replace(' ', '')}")
                                yaml_lines.append(f"    reps: {h_cfg['reps']}")
                                yaml_lines.append(f"    encoding: {h_cfg['encoding']}")
                                yaml_lines.append(f"    ansatz: {h_cfg['ansatz']}")
                                yaml_lines.append(f"    entangle: {h_cfg['entangle']}")
                            yaml_lines.append("")

    print(f"\nScan complete: Total baseline combinations analyzed: {total_scanned}")
    print(f"Missing configurations located: {total_missing}")
    
    if total_missing > 0:
        with open(output_yaml_path, "w") as f:
            f.write("\n".join(yaml_lines))
        print(f"SUCCESS: Generated recovery file containing {total_missing} configurations inside '{output_yaml_path}'")
    else:
        print("Excellent! 100% of your targeted experiments are accounted for.")

if __name__ == "__main__":
    log_sheet = "logs/experiments_systematic/correlation_study/correlation_qnn_identity.xlsx"
    recovery_output = "recovered_qnn_experiments.yml"
    
    generate_recovery_yaml(log_sheet, recovery_output)

# import os
# import pandas as pd
# from collections import defaultdict

# def get_best_mappings(head_count):
#     """
#     Returns a list of 6 mappings for each head count matching the quantum topology.
#     3 Full (using all features) and 3 Reduced (minimal overlap splits).
#     """
#     if head_count == 1:
#         full = [{'feats': ['sv', 'wv', 'yr', 'ya', 'rarad'], 'dims': [4]}]
#         reduced = []
#     elif head_count == 2:
#         full = [
#             {'feats': [['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad']], 'dims': [3, 1]},
#             {'feats': [['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad']], 'dims': [2, 2]},
#             {'feats': [['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad']], 'dims': [1, 3]},
#         ]
#         reduced = [
#             {'feats': [['sv', 'yr'], ['wv','yr', 'ya','rarad']], 'dims': [1, 3]},
#             {'feats': [['sv', 'wv', 'rarad'], ['yr', 'ya','rarad']], 'dims': [2, 2]},
#             {'feats': [['sv', 'wv', 'yr','rarad'], ['ya','yr']], 'dims': [3, 1]},
#         ]
#     elif head_count == 3:
#         full = [
#             {'feats': [['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad']], 'dims': [1, 1, 2]},
#             {'feats': [['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad']], 'dims': [2, 1, 1]},
#             {'feats': [['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad']], 'dims': [1, 2, 1]}
#         ]
#         reduced = [
#             {'feats': [['sv', 'yr'], ['wv', 'rarad'], ['yr', 'ya','rarad']], 'dims': [1, 1, 2]},
#             {'feats': [['sv', 'rarad'], ['wv', 'rarad'], ['yr', 'ya','rarad']], 'dims': [1, 1, 2]}
#         ]
#     elif head_count == 4:
#         full = [
#             {'feats': [['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad']], 'dims': [1, 1, 1, 1]},
#         ]
#         reduced = [
#             {'feats': [['sv', 'yr'], ['wv', 'rarad'], ['yr', 'rarad'], ['ya', 'yr']], 'dims': [1, 1, 1, 1]},
#             {'feats': [['sv', 'rarad'], ['wv', 'rarad'], ['yr', 'rarad'], ['ya', 'rarad']], 'dims': [1, 1, 1, 1]},
#             {'feats': [['sv', 'yr'], ['yr', 'ya'], ['ya', 'yr'], ['ya', 'yr']], 'dims': [1, 1, 1, 1]},
#         ]
#     return full + reduced

# def generate_all_expected_configs():
#     """Generates all 14,400 expected configurations from the original grid sweep."""
#     seeds = 10
#     optimizers = ['adam', 'spsa']
#     batch_sizes = [32, 128, 256]
#     spsa_perturbations = [0.05, 0.1, 0.15]
#     adam_lrs = [0.01, 0.001, 0.005]
#     spsa_lrs = ['[0.1, 0.001]', '[0.1, 0.01]']
    
#     all_configs = []
    
#     for head_num in [1, 2, 3, 4]:
#         mappings = get_best_mappings(head_num)
#         for map_config in mappings:
#             for reps in [1, 3]:
#                 hidden_size = 4 if reps == 1 else 16
#                 for opt in optimizers:
#                     for bs in batch_sizes:
#                         strategies = []
#                         if opt == 'adam':
#                             for start_lr in adam_lrs:
#                                 strategies.append({'lr': start_lr, 'sched': False, 'pat': 5, 'pert': 0.05})
#                                 strategies.append({'lr': start_lr, 'sched': True, 'pat': 5, 'pert': 0.05})
#                                 strategies.append({'lr': start_lr, 'sched': True, 'pat': 15, 'pert': 0.05})
#                         elif opt == 'spsa':
#                             for target_range in spsa_lrs:
#                                 for pert in spsa_perturbations:
#                                     strategies.append({'lr': target_range, 'sched': False, 'pat': 5, 'pert': pert})
                        
#                         for strat in strategies:
#                             for run in range(seeds):
#                                 heads_config = []
#                                 for i in range(head_num):
#                                     f = map_config['feats'][i] if head_num > 1 else map_config['feats']
#                                     d = map_config['dims'][i] if head_num > 1 else map_config['dims'][0]
#                                     heads_config.append({
#                                         'features': f,
#                                         'output_dim': d,
#                                         'hidden_size': hidden_size
#                                     })
                                
#                                 config = {
#                                     'head_num': head_num,
#                                     'hidden_size': hidden_size,
#                                     'optimizer': opt,
#                                     'batch_size': bs,
#                                     'learning_rate': strat['lr'],
#                                     'use_scheduler': strat['sched'],
#                                     'scheduler_patience': strat['pat'],
#                                     'perturbation': strat['pert'],
#                                     'run': run,
#                                     'heads_config': heads_config,
#                                     'map_config': map_config
#                                 }
#                                 all_configs.append(config)
#     return all_configs

# def get_heads_config_signature(heads_config):
#     """Creates a standardized string representation of a head config to prevent layout mismatch."""
#     parts = []
#     for h in heads_config:
#         f_str = str(sorted(h['features'])) if isinstance(h['features'], list) else str(h['features'])
#         f_str = f_str.replace(' ', '').replace("'", "").replace('"', '')
#         parts.append(f"f:{f_str}_d:{h['output_dim']}_h:{h['hidden_size']}")
#     return ";".join(parts)

# def parse_excel_completed_runs(excel_path, match_by_exact_heads=False):
#     """Reads the Excel spreadsheet and extracts unique signatures of completed runs."""
#     df = pd.read_excel(excel_path)
    
#     # Normalize column headers to lowercase and strip whitespace for flexible matching
#     df.columns = [str(c).strip().lower() for c in df.columns]
    
#     completed_signatures = set()
    
#     for _, row in df.iterrows():
#         # Extracted parameters with flexible fallbacks for common naming conventions
#         opt = str(row.get('optimizer', '')).strip().lower()
#         bs = int(row.get('batch_size', row.get('bs', 0)))
        
#         lr_val = row.get('learning_rate', row.get('lr', ''))
#         lr = str(lr_val).replace(' ', '')
        
#         sched_val = row.get('use_scheduler', row.get('sched', False))
#         sched = str(sched_val).strip().lower() in ['true', '1', 'yes'] if isinstance(sched_val, str) else bool(sched_val)
        
#         pat = int(row.get('scheduler_patience', row.get('pat', 5)))
#         pert = float(row.get('perturbation', row.get('pert', 0.05)))
#         run = int(row.get('run', row.get('seed', 0)))
#         head_num = int(row.get('head_num', row.get('head_count', row.get('heads', 1))))
#         hidden_size = int(row.get('hidden_size', 4))
        
#         if match_by_exact_heads:
#             # Look for a structural architecture representation column if available
#             h_str = ""
#             for col in ['heads_config', 'features', 'mapping', 'architecture']:
#                 if col in df.columns:
#                     h_str = str(row[col]).strip().replace(' ', '').replace("'", "").replace('"', '')
#                     break
#             sig = (head_num, hidden_size, opt, bs, lr, sched, pat, round(pert, 4), run, h_str)
#         else:
#             # Hyperparameter-only signature (Assumes all topology mappings for a matching hyperparam set are done)
#             sig = (head_num, hidden_size, opt, bs, lr, sched, pat, round(pert, 4), run)
            
#         completed_signatures.add(sig)
        
#     return completed_signatures

# def build_recovery_yaml(missing_configs):
#     """Formats missing experiments exactly matching the original file styling structure."""
#     save_dir = "systematic_classical_study"
#     yaml_lines = []
    
#     # Group missing runs by their main architecture metadata block to print tidy header labels
#     grouped = defaultdict(list)
#     for c in missing_configs:
#         heads_sig = get_heads_config_signature(c['heads_config'])
#         group_key = (
#             c['head_num'], c['hidden_size'], c['optimizer'], c['batch_size'],
#             str(c['learning_rate']).replace(' ', ''), c['use_scheduler'], 
#             c['scheduler_patience'], c['perturbation'], heads_sig
#         )
#         grouped[group_key].append(c)
        
#     for group_key, configs_in_group in grouped.items():
#         head_num, hidden_size, opt, bs, lr, sched, pat, pert, _ = group_key
        
#         # Original block header formatting style preserved
#         yaml_lines.append(f"## Head_{head_num}_Hidden_{hidden_size}_{opt}_BS_{bs}_Pert_{pert}_Sched_{sched}")
        
#         # Sort sequentially by run index
#         configs_in_group.sort(key=lambda x: x['run'])
        
#         for c in configs_in_group:
#             yaml_lines.append("- window_size: 5")
#             yaml_lines.append("  data: data/reduce_row_number_absolutes")
#             yaml_lines.append(f"  save_dir: {save_dir}")
#             yaml_lines.append("  select_features: [sv, wv, yr, ya, rarad]")
#             yaml_lines.append("  model: multihead")
#             yaml_lines.append("  predict: motion")
#             yaml_lines.append(f"  optimizer: {opt}")
#             yaml_lines.append("  maxiter: 200")
#             yaml_lines.append(f"  batch_size: {bs}")
#             yaml_lines.append(f"  learning_rate: {c['learning_rate']}")
#             yaml_lines.append(f"  use_scheduler: {str(sched).lower()}")
#             yaml_lines.append(f"  scheduler_patience: {pat}")
#             yaml_lines.append(f"  perturbation: {pert}")
#             yaml_lines.append("  weights: [1.0, 1.0, 1.0, 1.0]")
#             yaml_lines.append("  initialization: uniform")
#             yaml_lines.append("  save_plot: false")
#             yaml_lines.append(f"  run: {c['run']}")
#             yaml_lines.append("  heads_config:")
            
#             for h in c['heads_config']:
#                 f = h['features']
#                 d = h['output_dim']
#                 yaml_lines.append(f"  - features: {str(f).replace(' ', '')}")
#                 yaml_lines.append(f"    output_dim: {d}")
#                 yaml_lines.append(f"    hidden_size: {h['hidden_size']}")
#             yaml_lines.append("") 
            
#     return "\n".join(yaml_lines)

# def main(excel_path, match_by_exact_heads=False):
#     print("Generating complete grid mapping dictionary...")
#     all_expected = generate_all_expected_configs()
#     total_count = len(all_expected)
#     print(f"Total baseline experiments in grid sweep: {total_count}")
    
#     print(f"Analyzing completed experiments from: '{excel_path}'...")
#     completed_sigs = parse_excel_completed_runs(excel_path, match_by_exact_heads)
    
#     missing_configs = []
#     for c in all_expected:
#         lr_str = str(c['learning_rate']).replace(' ', '')
#         pert_val = round(c['perturbation'], 4)
        
#         if match_by_exact_heads:
#             # Build string signature lookup
#             c_h_str = get_heads_config_signature(c['heads_config'])
#             sig = (c['head_num'], c['hidden_size'], c['optimizer'], c['batch_size'], lr_str, c['use_scheduler'], c['scheduler_patience'], pert_val, c['run'], c_h_str)
#         else:
#             sig = (c['head_num'], c['hidden_size'], c['optimizer'], c['batch_size'], lr_str, c['use_scheduler'], c['scheduler_patience'], pert_val, c['run'])
            
#         if sig not in completed_sigs:
#             missing_configs.append(c)
            
#     print(f"Analysis Complete: Found {len(missing_configs)} missing experiments out of {total_count}.")
    
#     if missing_configs:
#         print("Writing recovery.yml file...")
#         recovery_content = build_recovery_yaml(missing_configs)
#         with open("recovery.yml", "w") as f:
#             f.write(recovery_content)
#         print("Successfully generated recovery.yml with all outstanding items!")
#     else:
#         print("All experiments have been successfully completed! No recovery file required.")

# if __name__ == "__main__":
#     # Change this path to your actual completed experiments log file path
#     EXCEL_FILE_PATH = "logs\systematic_classical_study\classical_experiments_summary.xlsx" 
    
#     # Set to True if your Excel logs the specific feature allocation layout mapping per head
#     # Set to False to evaluate purely on high-level hyperparameter grid blocks completed
#     MATCH_EXACT_HEADS_LAYOUT = False 
    
#     if os.path.exists(EXCEL_FILE_PATH):
#         main(EXCEL_FILE_PATH, match_by_exact_heads=MATCH_EXACT_HEADS_LAYOUT)
#     else:
#         print(f"Error: Could not find the file at '{EXCEL_FILE_PATH}'. Please verify the path.")