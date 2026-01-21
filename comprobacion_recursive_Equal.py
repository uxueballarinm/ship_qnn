import numpy as np

# --- 1. MOCK CLASSES (To simulate your environment) ---
class MockArgs:
    def __init__(self, predict_mode):
        self.horizon = 10
        self.features = [
            'Surge Velocity', 'Sway Velocity', 'Yaw Rate', 'Yaw Angle',
            'delta Surge Velocity', 'delta Sway Velocity', 'delta Yaw Rate', 'delta Yaw Angle'
        ]
        self.predict = predict_mode
        
        # NOTE: The hardcoded function requires targets in this EXACT order
        if predict_mode == 'motion':
            self.targets = ['Surge Velocity', 'Sway Velocity', 'Yaw Rate', 'Yaw Angle']
        else:
            self.targets = ['delta Surge Velocity', 'delta Sway Velocity', 'delta Yaw Rate', 'delta Yaw Angle']

class MockModel:
    def forward(self, x, params):
        # A deterministic dummy function: output = mean of input * 0.01
        # Returns shape (Batch, Horizon, Targets)
        # We simulate 4 targets
        val = np.mean(x) * 0.01
        return np.full((x.shape[0], 10, 4), val)

class MockScaler:
    def transform(self, x): return x * 0.5  # Dummy scaling
    def inverse_transform(self, x): return x * 2.0  # Dummy un-scaling
    
def recursive_generic(args, model, best_params, x_test, x_scaler, y_scaler):
    num_samples = x_test.shape[0]; horizon = args.horizon        
    num_targets = len(args.targets)
    
    # 1. Setup Update Rules (Support Multiple Updates per Target)
    # Structure: {target_idx: [(feature_idx, mode), (feature_idx, mode), ...]}
    update_rules = {} 

    for t_idx, t_name in enumerate(args.targets):
        update_rules[t_idx] = []
        is_target_delta = "delta" in t_name.lower() or args.predict == 'delta'
        
        # --- LOGIC 1: Primary Update (Direct Match) ---
        if t_name in args.features:
            f_idx = args.features.index(t_name)
            # If target is Delta and Feature is Delta -> DIRECT
            # If target is Motion and Feature is Motion -> DIRECT
            update_rules[t_idx].append((f_idx, "DIRECT"))

        # --- LOGIC 2: Secondary Physics Update (Coupling) ---
        # If we predicted 'Motion', we should also update 'Delta' (Differentiation)
        # If we predicted 'Delta', we should also update 'Motion' (Integration)
        
        secondary_name = None
        mode = None
        
        if is_target_delta:
            # We have Delta, look for Motion to Integrate
            clean = t_name.replace("delta ", "").strip()
            if clean in args.features: 
                secondary_name = clean
                mode = "INTEGRATE"
        else:
            # We have Motion, look for Delta to Differentiate
            delta_ver = f"delta {t_name}"
            if delta_ver in args.features: 
                secondary_name = delta_ver
                mode = "DIFF"
        
        if secondary_name:
            f_idx_sec = args.features.index(secondary_name)
            update_rules[t_idx].append((f_idx_sec, mode))

    # 2. Initialization
    recursive_preds = np.zeros((num_samples, horizon, num_targets))
    curr_window = x_test[0:1, :, :].copy()
    last_feat_real = x_scaler.inverse_transform(curr_window[:, -1, :])
    
    # Trackers for differentiation
    last_pred_values = np.zeros(num_targets) 

    # 3. Main Loop
    for i in range(num_samples):
        preds_full = model.forward(curr_window, best_params)
        recursive_preds[i] = preds_full[0]

        next_gt_idx = min(i + 1, num_samples - 1)
        next_gt_features_norm = x_test[next_gt_idx:next_gt_idx+1, -1, :] 
        
        # Start with Ground Truth
        next_input_real = x_scaler.inverse_transform(next_gt_features_norm)
        
        pred_step_norm = preds_full[:, 0, :]
        if y_scaler: pred_step_real = y_scaler.inverse_transform(pred_step_norm)
        else: pred_step_real = pred_step_norm

        # Apply Updates
        for t_idx, updates in update_rules.items():
            pred_val = pred_step_real[0, t_idx]
            
            for f_idx, mode in updates:
                if mode == "DIRECT":
                    next_input_real[0, f_idx] = pred_val
                elif mode == "INTEGRATE":
                    # New Pos = Old Pos + Pred Delta
                    old_val = last_feat_real[0, f_idx]
                    next_input_real[0, f_idx] = old_val + pred_val
                elif mode == "DIFF":
                    # New Delta = Pred Pos - Old Pos (approx as Pred Pos - Last Pred Pos)
                    # Note: To match hardcoded logic exactly, we use (Current Pred - Last Iteration Pred)
                    if i == 0: diff_val = pred_val 
                    else: diff_val = pred_val - last_pred_values[t_idx]
                    next_input_real[0, f_idx] = diff_val

        # Update trackers
        last_pred_values = pred_step_real[0]
        last_feat_real = next_input_real   
        
        new_row_norm = x_scaler.transform(next_input_real)
        curr_window = np.concatenate([curr_window[:, 1:, :], new_row_norm.reshape(1, 1, -1)], axis=1)

    return recursive_preds#, 0.0

# --- 3. THE HARDCODED FUNCTION (Your Fixed Code) ---
def recursive_hardcoded(args, model, best_params, x_test, x_scaler, y_scaler):
        
    num_samples = x_test.shape[0]; horizon = args.horizon
    feat_names = args.features 
    def get_idx(name): return feat_names.index(name) if name in feat_names else None
    idx_wv, idx_sv, idx_yr, idx_ya = get_idx('Surge Velocity'), get_idx('Sway Velocity'), get_idx('Yaw Rate'), get_idx('Yaw Angle')
    idx_dwv, idx_dsv, idx_dyr, idx_dya = get_idx('delta Surge Velocity'), get_idx('delta Sway Velocity'), get_idx('delta Yaw Rate'), get_idx('delta Yaw Angle')

    direct_updates = []
    physics_updates = []
    if args.predict == 'delta':
        if idx_dwv is not None: direct_updates.append('delta Surge Velocity')
        if idx_dsv is not None: direct_updates.append('delta Sway Velocity')
        if idx_dyr is not None: direct_updates.append('delta Yaw Rate')
        if idx_dya is not None: direct_updates.append('delta Yaw Angle')
        if idx_wv is not None: physics_updates.append('Surge Velocity(integrated from delta Surge Velocity)')
        if idx_sv is not None: physics_updates.append('Sway Velocity(integrated from delta Sway Velocity)')
        if idx_yr is not None: physics_updates.append('Yaw Rate(integrated from delta Yaw Rate)')
        if idx_ya is not None: physics_updates.append('Yaw Angle(integrated from delta Yaw Angle)')
    else: # motion
        if idx_wv is not None: direct_updates.append('Surge Velocity')
        if idx_sv is not None: direct_updates.append('Sway Velocity')
        if idx_yr is not None: direct_updates.append('Yaw Rate')
        if idx_ya is not None: direct_updates.append('Yaw Angle')
        if idx_dwv is not None: physics_updates.append('delta Surge Velocity (differentiated from Surge Velocity)')
        if idx_dsv is not None: physics_updates.append('delta Sway Velocity (differentiated from Sway Velocity)')
        if idx_dyr is not None: physics_updates.append('delta Yaw Rate (differentiated from Yaw Rate)')
        if idx_dya is not None: physics_updates.append('delta Yaw Angle (differentiated from Yaw Angle)')

    num_driven = len(direct_updates) + len(physics_updates)
    total_feats = len(feat_names)

    if num_driven == 0:
        print("  > Fully Open-Loop (No recursion).")
        preds_full = model.forward(x_test, best_params)
        return preds_full, 0.0

    print(f"  > Recursive Loop Active ({num_driven}/{total_feats} features)")
    if direct_updates:
        print(f"    Direct Feedback: {', '.join(direct_updates)}")
    if physics_updates:
        print(f"    Physics Feedback: {', '.join(physics_updates)}")

    recursive_ratio = num_driven / total_feats
    
    curr_window = x_test[0:1, :, :] 
    last_step_real = x_scaler.inverse_transform(curr_window[:, -1, :])
    recursive_preds = np.zeros((num_samples, horizon, len(args.targets)))
    last_wv, last_sv, last_yr, last_ya = 0.0, 0.0, 0.0, 0.0

    for i in range(num_samples):
        preds_full = model.forward(curr_window, best_params)
        recursive_preds[i] = preds_full[0]
        pred_step = preds_full[:, 0, :] 
        
        next_gt_idx = min(i + 1, num_samples - 1)
        next_gt_features_norm = x_test[next_gt_idx:next_gt_idx+1, -1, :] 

        if y_scaler: pred_step_real = y_scaler.inverse_transform(pred_step)
        else: pred_step_real = pred_step

        next_gt_real = x_scaler.inverse_transform(next_gt_features_norm)
        new_row_real = np.copy(next_gt_real)
        
        if args.predict == 'delta':
            dwv, dsv,dyr,dya = pred_step_real[0, 0], pred_step_real[0, 1], pred_step_real[0, 2], pred_step_real[0, 3]
            if idx_dwv is not None: new_row_real[0, idx_dwv] = dwv
            if idx_dsv is not None: new_row_real[0, idx_dsv] = dsv
            if idx_dyr is not None: new_row_real[0, idx_dyr] = dyr
            if idx_dya is not None: new_row_real[0, idx_dya] = dya
            if idx_wv is not None: new_row_real[0, idx_wv] = last_step_real[0, idx_wv] + dwv
            if idx_sv is not None: new_row_real[0, idx_sv] = last_step_real[0, idx_sv] + dsv
            if idx_yr is not None: new_row_real[0, idx_yr] = last_step_real[0, idx_yr] + dyr
            if idx_ya is not None: new_row_real[0, idx_ya] = last_step_real[0, idx_ya] + dya
        else: 
            wv, sv, yr, ya = pred_step_real[0, 0], pred_step_real[0, 1], pred_step_real[0, 2], pred_step_real[0, 3]
            if idx_wv is not None: new_row_real[0, idx_wv] = wv
            if idx_sv is not None: new_row_real[0, idx_sv] = sv
            if idx_yr is not None: new_row_real[0, idx_yr] = yr
            if idx_ya is not None: new_row_real[0, idx_ya] = ya
            if idx_dwv is not None: 
                new_row_real[0, idx_dwv] = wv- last_wv
                last_wv = wv
            if idx_dsv is not None:
                new_row_real[0, idx_dsv] = sv - last_sv 
                last_sv = sv
            if idx_dyr is not None:
                new_row_real[0, idx_dyr] = yr - last_yr 
                last_yr = yr
            if idx_dya is not None:
                new_row_real[0, idx_dya] = ya - last_ya 
                last_ya = ya

        last_step_real = new_row_real
        new_row_norm = x_scaler.transform(new_row_real)
        curr_window = np.concatenate([curr_window[:, 1:, :], new_row_norm.reshape(1, 1, -1)], axis=1)

    return recursive_preds#, recursive_ratio
    

# --- 4. RUN COMPARISON TEST ---

# Create Data
x_test = np.random.rand(50, 10, 8) # (Samples, Window, Features)

# TEST A: Motion Prediction (Integration Logic inside generic, Diff logic inside hardcoded)
print("--- TEST A: PREDICT MOTION ---")
args_motion = MockArgs('motion')
res_gen_m = recursive_generic(args_motion, MockModel(), None, x_test, MockScaler(), MockScaler())
res_hard_m = recursive_hardcoded(args_motion, MockModel(), None, x_test, MockScaler(), MockScaler())

if np.allclose(res_gen_m, res_hard_m):
    print("SUCCESS: Motion predictions are identical.")
else:
    print("FAIL: Motion predictions differ.")
    print("Diff:", np.max(np.abs(res_gen_m - res_hard_m)))


# TEST B: Delta Prediction (Integration Logic inside hardcoded)
print("\n--- TEST B: PREDICT DELTA ---")
args_delta = MockArgs('delta')
res_gen_d = recursive_generic(args_delta, MockModel(), None, x_test, MockScaler(), MockScaler())
res_hard_d = recursive_hardcoded(args_delta, MockModel(), None, x_test, MockScaler(), MockScaler())

if np.allclose(res_gen_d, res_hard_d):
    print("SUCCESS: Delta predictions are identical.")
else:
    print("FAIL: Delta predictions differ.")
    print("Diff:", np.max(np.abs(res_gen_d - res_hard_d)))