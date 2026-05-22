import os

def get_best_mappings(head_count):
    """
    Returns a list of 6 mappings for each head count matching the quantum topology.
    3 Full (using all features) and 3 Reduced (minimal overlap splits).
    """
    if head_count == 1:
        full = [
            {'feats': ['sv', 'wv', 'yr', 'ya', 'rarad'], 'dims': [4]},
        ]
        reduced = []
    elif head_count == 2:
        full = [
            {'feats': [['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad']], 'dims': [3, 1]},
            {'feats': [['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad']], 'dims': [2, 2]},
            {'feats': [['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad']], 'dims': [1, 3]},
        ]
        reduced = [
            {'feats': [['sv', 'yr'], ['wv','yr', 'ya','rarad']], 'dims': [1, 3]},
            {'feats': [['sv', 'wv', 'rarad'], ['yr', 'ya','rarad']], 'dims': [2, 2]},
            {'feats': [['sv', 'wv', 'yr','rarad'], ['ya','yr']], 'dims': [3, 1]},
        ]
    elif head_count == 3:
        full = [
            {'feats': [['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad']], 'dims': [1, 1, 2]},
            {'feats': [['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad']], 'dims': [2, 1, 1]},
            {'feats': [['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad']], 'dims': [1, 2, 1]}
        ]
        reduced = [
            {'feats': [['sv', 'yr'], ['wv', 'rarad'], ['yr', 'ya','rarad']], 'dims': [1, 1, 2]},
            {'feats': [['sv', 'rarad'], ['wv', 'rarad'], ['yr', 'ya','rarad']], 'dims': [1, 1, 2]}
        ]
    elif head_count == 4:
        full = [
            {'feats': [['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad'], ['sv', 'wv','yr','ya', 'rarad']], 'dims': [1, 1, 1, 1]},
        ]
        reduced = [
            {'feats': [['sv', 'yr'], ['wv', 'rarad'], ['yr', 'rarad'], ['ya', 'yr']], 'dims': [1, 1, 1, 1]},
            {'feats': [['sv', 'rarad'], ['wv', 'rarad'], ['yr', 'rarad'], ['ya', 'rarad']], 'dims': [1, 1, 1, 1]},
            {'feats': [['sv', 'yr'], ['yr', 'ya'], ['ya', 'yr'], ['ya', 'yr']], 'dims': [1, 1, 1, 1]},
        ]
    return full + reduced

def build_classical_yaml():
    seeds = 10
    save_dir = "systematic_classical_study"
    yaml_lines = []
    
    # Grid sweeps
    optimizers = ['adam', 'spsa']
    batch_sizes = [32, 128, 256]
    spsa_perturbations = [0.05, 0.1, 0.15]
    adam_lrs = [0.01, 0.001, 0.005]          # Initial values for Adam
    spsa_lrs = ['[0.1, 0.001]', '[0.1, 0.01]']
    # Learning rate strategies
    # For SPSA: [dynamic_decay]
    # For Adam: [static, dynamic_patience_5, dynamic_patience_15]
    
    for head_num in [1, 2, 3, 4]:
        mappings = get_best_mappings(head_num)
        for map_config in mappings:
            for reps in [1, 3]:
                # Map reps parameter matching directly to hidden_size
                hidden_size = 4 if reps == 1 else 16
                
                for opt in optimizers:
                    for bs in batch_sizes:
                        
                        # Build strategies per optimizer
                        strategies = []
                        if opt == 'adam':
                            for start_lr in adam_lrs:
                                # Static Setup
                                strategies.append({
                                    'lr': start_lr, 'sched': False, 'pat': 5, 'pert': 0.05
                                })
                                # Dynamic Scheduler Setups (varying patience)
                                strategies.append({
                                    'lr': start_lr, 'sched': True, 'pat': 5, 'pert': 0.05
                                })
                                strategies.append({
                                    'lr': start_lr, 'sched': True, 'pat': 15, 'pert': 0.05
                                })
                        elif opt == 'spsa':
                            for target_range in spsa_lrs:
                                for pert in spsa_perturbations:
                                    # SPSA relies natively on its decay factory loops instead of Plateau schedulers
                                    strategies.append({
                                        'lr': target_range, 'sched': False, 'pat': 5, 'pert': pert
                                    })
                        
                        for strat in strategies:
                            yaml_lines.append(f"## Head_{head_num}_Hidden_{hidden_size}_{opt}_BS_{bs}_Pert_{strat['pert']}_Sched_{strat['sched']}")
                            
                            for run in range(seeds):
                                yaml_lines.append(f"- window_size: 5")
                                yaml_lines.append(f"  data: data/reduce_row_number_absolutes")
                                yaml_lines.append(f"  save_dir: {save_dir}")
                                yaml_lines.append(f"  select_features: [sv, wv, yr, ya, rarad]")
                                yaml_lines.append(f"  model: multihead")
                                yaml_lines.append(f"  predict: motion")
                                yaml_lines.append(f"  optimizer: {opt}")
                                yaml_lines.append(f"  maxiter: 200")
                                yaml_lines.append(f"  batch_size: {bs}")
                                yaml_lines.append(f"  learning_rate: {strat['lr']}")
                                yaml_lines.append(f"  use_scheduler: {str(strat['sched']).lower()}")
                                yaml_lines.append(f"  scheduler_patience: {strat['pat']}")
                                yaml_lines.append(f"  perturbation: {strat['pert']}")
                                yaml_lines.append(f"  weights: [1.0, 1.0, 1.0, 1.0]")
                                yaml_lines.append(f"  initialization: uniform")
                                yaml_lines.append(f"  save_plot: false")
                                yaml_lines.append(f"  run: {run}")
                                yaml_lines.append(f"  heads_config:")
                                
                                for i in range(head_num):
                                    f = map_config['feats'][i] if head_num > 1 else map_config['feats']
                                    d = map_config['dims'][i] if head_num > 1 else map_config['dims'][0]
                                    
                                    yaml_lines.append(f"  - features: {str(f).replace(' ', '')}")
                                    yaml_lines.append(f"    output_dim: {d}")
                                    yaml_lines.append(f"    hidden_size: {hidden_size}")
                                yaml_lines.append("") 
    return "\n".join(yaml_lines)

if __name__ == "__main__":
    with open("study_classical.yml", "w") as f:
        f.write(build_classical_yaml())
    print("Successfully generated study_classical.yml for structural evaluation study.")