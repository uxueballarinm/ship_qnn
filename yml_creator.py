import os

def get_best_mappings(head_count):
    """
    Returns a list of 6 mappings for each head count:
    3 Full (using rarad) and 3 Reduced (minimal overlap).
    """
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

def build_yaml_content(is_qelm=False):
    seeds = 30 if is_qelm else 10
    optimizer = 'ridge' if is_qelm else 'spsa'
    initialization = 'uniform' if is_qelm else 'identity'
    maxiter = 1 if is_qelm else 4000
    lr = "0.001" if is_qelm else "[0.1, 0.001]"
    save_dir = "study_qelm" if is_qelm else "study_qnn"
    
    yaml_lines = []
    
    for head_num in [1, 2, 3, 4]:
        mappings = get_best_mappings(head_num)
        for map_config in mappings:
            for reps in [1, 3]:
                for encoding in ['serial', 'compact']:
                    # We output Run/Combination block
                    yaml_lines.append(f"## Head_{head_num}_Reps_{reps}_{encoding}")
                    
                    for run in range(seeds):
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
                        
                        for i in range(head_num):
                            f = map_config['feats'][i] if head_num > 1 else map_config['feats']
                            m = map_config['maps'][i] if head_num > 1 else map_config['maps'][0]
                            d = map_config['dims'][i] if head_num > 1 else map_config['dims'][0]
                            
                            yaml_lines.append(f"  - features: {str(f).replace(' ', '')}")
                            yaml_lines.append(f"    output_dim: {d}")
                            yaml_lines.append(f"    map: {str(m).replace(' ', '')}")
                            yaml_lines.append(f"    reps: {reps}")
                            yaml_lines.append(f"    encoding: {encoding}")
                            yaml_lines.append(f"    ansatz: efficientsu2")
                            yaml_lines.append(f"    entangle: linear")
                        yaml_lines.append("") # Empty line between experiments
    return "\n".join(yaml_lines)

# Generate and save
with open("study_qnn.yml", "w") as f:
    f.write(build_yaml_content(is_qelm=False))

with open("study_qelm.yml", "w") as f:
    f.write(build_yaml_content(is_qelm=True))

print("Created study_qnn.yml (960 experiments)")
print("Created study_qelm.yml (2880 experiments)")