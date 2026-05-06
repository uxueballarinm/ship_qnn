import yaml
import numpy as np

def generate_complete_grid(seeds=10, filename="experiment_definitions/baselines/mlp/complete_grid_search.yaml"):
    experiments = []
    summary_path = "logs/baselines/optimizer_hyperparameters_grid.xlsx"
    
    scenarios = {
        "1h_Opt1": {"model": "vanilla", "h_size": 3, "cfg": None},
        "2h_Opt1": {"model": "multihead", "cfg": [
            {"features": ["sv", "wv", "rarad"], "output_dim": 2, "hidden_size": 2},
            {"features": ["yr", "ya", "rarad"], "output_dim": 2, "hidden_size": 2}]},
        "3h_Opt1": {"model": "multihead", "cfg": [
            {"features": ["sv", "yr"], "output_dim": 1, "hidden_size": 2},
            {"features": ["wv", "yr", "rarad"], "output_dim": 2, "hidden_size": 2},
            {"features": ["ya", "yr"], "output_dim": 1, "hidden_size": 2}]},
        "4h_Opt1": {"model": "multihead", "cfg": [
            {"features": ["sv", "yr"], "output_dim": 1, "hidden_size": 2},
            {"features": ["wv", "rarad"], "output_dim": 1, "hidden_size": 2},
            {"features": ["yr", "rarad"], "output_dim": 1, "hidden_size": 2},
            {"features": ["ya", "yr"], "output_dim": 1, "hidden_size": 2}]}}

    # SPSA GRID: Explicitly defining start and end pairs
    # This allows you to test 0.1 -> 0.01 AND 0.1 -> 0.001
    spsa_lrs = [
        [1.0, 0.1], [1.0, 0.01],
        [0.1, 0.01], [0.1, 0.001],
        [0.05, 0.005], [0.05, 0.0005]
    ]
    spsa_perturbations = [0.01, 0.05, 0.1]
    batch_sizes = [32, 64, 128, 256, 512, 1024]

    # Adam GRID (Remains static as Adam is internally adaptive)
    adam_lrs = [0.1, 0.01, 0.001, 0.0005]

    for name, arch in scenarios.items():
        # --- SPSA: Dynamic LR Combinations ---
        for lr_pair in spsa_lrs:
            for pert in spsa_perturbations:
                for batch in batch_sizes:
                    for seed in range(seeds):
                        exp = {
                            "run": seed,
                            "model": arch["model"],
                            "optimizer": "spsa",
                            "learning_rate": lr_pair, # [start, end] triggers dynamic decay
                            "perturbation": pert,
                            "batch_size": batch,
                            "maxiter": 4000,
                            "save_dir": f"grid/{name}/spsa/lr{lr_pair[0]}_to_{lr_pair[1]}_p{pert}_b{batch}",
                            "excel_path": summary_path
                        }
                        if arch["model"] == "multihead": exp["heads_config"] = arch["cfg"]
                        else: exp["hidden_size"] = arch["h_size"]
                        experiments.append(exp)

        # --- ADAM: Static Base LR ---
        for lr in adam_lrs:
            for batch in batch_sizes:
                for seed in range(seeds):
                    exp = {
                        "run": seed,
                        "model": arch["model"],
                        "optimizer": "adam",
                        "learning_rate": lr,
                        "batch_size": batch,
                        "maxiter": 300,
                        "save_dir": f"grid/{name}/adam/lr{lr}_b{batch}",
                        "excel_path": summary_path
                    }
                    if arch["model"] == "multihead": exp["heads_config"] = arch["cfg"]
                    else: exp["hidden_size"] = arch["h_size"]
                    experiments.append(exp)

    with open(filename, 'w') as f:
        yaml.dump(experiments, f, default_flow_style=False)
    
    print(f"Generated {len(experiments)} experiments in {filename}")

if __name__ == "__main__":
    generate_complete_grid(seeds=10)