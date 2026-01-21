import pandas as pd
import numpy as np

# 1. Load Data
data_path = "datasets\zigzag_11_11_ind_reduced_2_s.csv"
df = pd.read_csv(data_path)

# 2. Define Inputs (X) and Targets (Y)
# Adjust these lists based on what your model actually uses!
feature_names = ["Position X", "Position Y", "Surge Velocity", "Sway Velocity", "Yaw Rate", "Yaw Angle", "Speed U", "Rudder Angle (deg)", "Rudder Angle (rad)"]

# We usually predict the 'Next Step', so let's shift the targets
# to measure predictive power.
targets = ["Yaw Angle"]# "Surge Velocity","Sway Velocity", "Yaw Rate", "Yaw Angle"] # Or 'delta x', 'delta y'
df_features = df[feature_names].iloc[:-1].reset_index(drop=True)
df_targets = df[targets].iloc[1:].reset_index(drop=True)

# 3. Calculate "Importance" (Correlation with Target)
# We take the max correlation with ANY target variable
target_corr = {}
for feat in feature_names:
    corrs = [abs(df_features[feat].corr(df_targets[t])) for t in targets]
    target_corr[feat] = max(corrs)

print("--- Feature Importance (Correlation with Target) ---")
sorted_importance = sorted(target_corr.items(), key=lambda x: x[1], reverse=True)
for name, score in sorted_importance:
    print(f"{name:<20}: {score:.4f}")

# 4. Calculate "Grouping" (Correlation with Each Other)
feat_corr_matrix = df_features.corr().abs()

# 5. GENERATE OPTIMAL ORDER
remaining = set(feature_names)
final_indices = []
final_names = []

# We create groups of 3 (for U-gate slots: Theta, Phi, Lambda)
while remaining:
    # A. Pick the "Captain" for this group: The most important remaining feature
    # (We assume the most important feature deserves the Theta slot)
    # Filter 'sorted_importance' for only remaining keys
    candidates = [f for f in sorted_importance if f[0] in remaining]
    if not candidates: break
    
    captain = candidates[0][0] # Highest importance
    
    group = [captain]
    remaining.remove(captain)
    
    # B. Find the 2 best "Wingmen" (Most correlated with the Captain)
    # We sort remaining features by their correlation to the Captain
    if remaining:
        # Get correlations of remaining features with the captain
        wingmen_scores = feat_corr_matrix[captain][list(remaining)].sort_values(ascending=False)
        
        # Take up to 2 wingmen
        num_to_take = min(2, len(remaining))
        best_wingmen = wingmen_scores.index[:num_to_take].tolist()
        
        group.extend(best_wingmen)
        for w in best_wingmen:
            remaining.remove(w)
            
    # C. Add this group to the final list
    # The 'group' list is already sorted: [Captain (Theta), Wingman1 (Phi), Wingman2 (Lambda)]
    final_names.extend(group)

# 6. Output for Config
print("\n--- Optimized 'Physics' Order ---")
print(f"Feature Order: {final_names}")
indices = [feature_names.index(f) for f in final_names]
print(f"Indices: {indices}")