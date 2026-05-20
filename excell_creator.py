import pandas as pd
import glob
import os
import ast

def get_abs_path(path):
    abs_p = os.path.abspath(path)
    if os.name == 'nt' and not abs_p.startswith('\\\\?\\'):
        return '\\\\?\\' + abs_p
    return abs_p

# --- 1. COLUMN DEFINITION (Strict 79 Columns) ---
TARGET_COLUMNS = [
    "date", "model_id", "run", "weight_selection", "data", "data_n", "data_dt", 
    "features", "targets", "window_size", "horizon", "predict", "norm", 
    "reconstruct_train", "reconstruct_val", "model", "heads_config", "head_number", 
    "encoding", "ansatz", "entangle", "reps", "map", "reorder", "optimizer", 
    "maxiter", "iterations", "tolerance", "batch_size", "learning_rate", 
    "perturbation", "weights", "freeze_qnn", "trainable_encoding", "use_hadamard", 
    "total params", "q params", "c params", "final val loss", "Val Global Open MSE", 
    "Val Global Open R2", "Val Global Closed R2", "step MSE", "step R2", "local MSE", 
    "global open MSE", "global open R2", "global closed MSE", "global closed R2", 
    "recursivity", "initialization", "Surge Velocity Step MSE", "Surge Velocity Step R2", 
    "Surge Velocity Local MSE", "Surge Velocity Global open MSE", "Surge Velocity Global open R2", 
    "Surge Velocity Global closed MSE", "Surge Velocity Global closed R2", "Sway Velocity Step MSE", 
    "Sway Velocity Step R2", "Sway Velocity Local MSE", "Sway Velocity Global open MSE", 
    "Sway Velocity Global open R2", "Sway Velocity Global closed MSE", "Sway Velocity Global closed R2", 
    "Yaw Rate Step MSE", "Yaw Rate Step R2", "Yaw Rate Local MSE", "Yaw Rate Global open MSE", 
    "Yaw Rate Global open R2", "Yaw Rate Global closed MSE", "Yaw Rate Global closed R2", 
    "Yaw Angle Step MSE", "Yaw Angle Step R2", "Yaw Angle Local MSE", "Yaw Angle Global open MSE", 
    "Yaw Angle Global open R2", "Yaw Angle Global closed MSE", "Yaw Angle Global closed R2"
]

def parse_to_list(val):
    if isinstance(val, list): return val
    if val is None or pd.isna(val): return []
    if isinstance(val, str):
        val = val.strip()
        if (val.startswith('[') and val.endswith(']')) or (val.startswith('{') and val.endswith('}')):
            try: return ast.literal_eval(val)
            except: return [x.strip().strip("'").strip('"') for x in val[1:-1].split(',')]
    return []

def clean_bool(val, default=False):
    if val is None or pd.isna(val): return default
    if isinstance(val, bool): return val
    s = str(val).strip().upper()
    if s in ["TRUE", "VERDADERO", "1", "1.0"]: return True
    if s in ["FALSE", "FALSO", "0", "0.0"]: return False
    return default

def robust_numeric(val):
    """Handles Spanish decimals (0,001) and precision errors."""
    if val is None or pd.isna(val): return 0.0
    try:
        if isinstance(val, str):
            val = val.replace(',', '.')
        return float(val)
    except:
        return 0.0

def fix_sv_wv_normalization(val):
    """
    Normalizes Surge/Sway order. 
    Forces 'sv' (Surge) to always come before 'wv' (Sway) in strings 
    to group old experiments (reversed) with new ones.
    """
    if val is None or pd.isna(val): return val
    s = str(val).lower().replace(" ", "")
    # Standardize long names to codes
    s = s.replace('surgevelocity', 'sv').replace('swayvelocity', 'wv')
    s = s.replace('surge', 'sv').replace('sway', 'wv')
    
    # If it's a list string like ['wv','sv'], we swap to ['sv','wv']
    if "'wv','sv'" in s: s = s.replace("'wv','sv'", "'sv','wv'")
    if "wv,sv" in s: s = s.replace("wv,sv", "sv,wv")
    
    return s

def get_arch_fp(row):
    """Unique ID for an architecture (ignores seed/run)."""
    m_raw = row.get('heads_config', row.get('map', '[]'))
    m_str = str(parse_to_list(m_raw)).replace(" ", "").lower()
    ws = row.get('window_size', 5)
    reps = row.get('reps', 1)
    enc = str(row.get('encoding', 'serial')).lower()
    return f"W{ws}_R{reps}_E{enc}_M{m_str}"

def run_curation_study():
    search_root = r"logs" 
    output_dir = r"logs\curated_studies"
    os.makedirs(output_dir, exist_ok=True)

    print("Step 1: Loading files (ignoring Power Query junk)...")
    files = glob.glob(os.path.join(search_root, "**/*.[xc][ls][sv]*"), recursive=True)
    all_data = []
    
    for f in files:
        if "QRC_VS_SPSA" in f.upper() or "curated_studies" in f: continue
        try:
            f_abs = get_abs_path(f)
            if f.lower().endswith(('.xlsx', '.xls')):
                xl = pd.ExcelFile(f_abs)
                for sheet in xl.sheet_names:
                    # Only skip explicit Power Query "Consulta" sheets
                    if "CONSULTA" in sheet.upper(): continue
                    df = xl.parse(sheet)
                    if not df.empty: all_data.append(df)
            else:
                # Try reading with semicolon for Spanish CSVs if comma fails
                try: df = pd.read_csv(f_abs, sep=',')
                except: df = pd.read_csv(f_abs, sep=';')
                if not df.empty: all_data.append(df)
        except: continue

    full_df = pd.concat(all_data, ignore_index=True)
    print(f"Total rows scanned: {len(full_df)}")

    # --- 2. CONSOLIDATION & NORMALIZATION ---
    # Consolidate columns
    if 'freeze' in full_df.columns:
        full_df['freeze_qnn'] = full_df['freeze_qnn'].fillna(full_df['freeze'])
    if 'total_params' in full_df.columns:
        full_df['total params'] = full_df['total params'].fillna(full_df['total_params'])

    # Fix SV/WV Swap across all config columns
    for col in ['features', 'targets', 'custom_targets', 'select_features', 'heads_config', 'map']:
        if col in full_df.columns:
            full_df[col] = full_df[col].apply(fix_sv_wv_normalization)

    def clean_str(x): return str(x).lower().strip()
    
    # --- 3. FILTERING ---
    mask_base = (
        (full_df['predict'].fillna('').apply(clean_str) == 'motion') &
        (full_df['ansatz'].fillna('').apply(clean_str).str.contains('eff')) &
        (full_df['entangle'].fillna('').apply(clean_str).str.startswith('lin')) &
        (full_df['norm'].apply(lambda x: clean_bool(x, True)) == True) &
        (full_df.get('trainable_encoding', pd.Series([None]*len(full_df))).apply(lambda x: clean_bool(x, False)) == False)
    )
    df_filtered = full_df[mask_base].copy()

    # SPSA (QNN)
    qnn_full = df_filtered[
        (df_filtered['optimizer'].fillna('').apply(clean_str) == 'spsa') &
        (df_filtered['freeze_qnn'].apply(lambda x: clean_bool(x, False)) == False) &
        (df_filtered['batch_size'].apply(lambda x: robust_numeric(x) == 256)) &
        (df_filtered['learning_rate'].apply(lambda x: [float(i) for i in parse_to_list(x)] == [0.1, 0.001]))
    ].copy()

    # Ridge (QELM)
    qelm_full = df_filtered[
        (df_filtered['optimizer'].fillna('').apply(clean_str) == 'ridge') &
        (df_filtered['freeze_qnn'].apply(lambda x: clean_bool(x, False)) == True) &
        (df_filtered['learning_rate'].apply(lambda x: robust_numeric(x) == 0.001))
    ].copy()

    # --- 4. MINIMUM 10 RUNS LOGIC ---
    qnn_full['arch_fp'] = qnn_full.apply(get_arch_fp, axis=1)
    qelm_full['arch_fp'] = qelm_full.apply(get_arch_fp, axis=1)

    # Filter individual lists for 10 runs
    qnn_counts = qnn_full.groupby('arch_fp')['run'].nunique()
    valid_qnn_archs = qnn_counts[qnn_counts >= 10].index
    qnn_final = qnn_full[qnn_full['arch_fp'].isin(valid_qnn_archs)]

    qelm_counts = qelm_full.groupby('arch_fp')['run'].nunique()
    valid_qelm_archs = qelm_counts[qelm_counts >= 10].index
    qelm_final = qelm_full[qelm_full['arch_fp'].isin(valid_qelm_archs)]

    # Filter Matched (Both must have >= 10)
    matched_archs = set(valid_qnn_archs).intersection(set(valid_elm_archs)) if 'valid_elm_archs' in locals() else set(valid_qnn_archs).intersection(set(valid_qelm_archs))
    
    matched_long = pd.concat([
        qnn_full[qnn_full['arch_fp'].isin(matched_archs)],
        qelm_full[qelm_full['arch_fp'].isin(matched_archs)]
    ], ignore_index=True)

    # --- 5. EXPORT ---
    # Ensure all target columns exist
    for col in TARGET_COLUMNS:
        if col not in qnn_final.columns: qnn_final[col] = None
        if col not in qelm_final.columns: qelm_final[col] = None
        if col not in matched_long.columns: matched_long[col] = None

    qnn_final[TARGET_COLUMNS].to_excel(os.path.join(output_dir, "study_qnn_spsa.xlsx"), index=False)
    qelm_final[TARGET_COLUMNS].to_excel(os.path.join(output_dir, "study_qelm_ridge.xlsx"), index=False)
    matched_long[TARGET_COLUMNS].to_excel(os.path.join(output_dir, "study_qnn_and_qelm_matched_long.xlsx"), index=False)

    print(f"\nFinal Summary:")
    print(f"- QNN Runs (>=10): {len(qnn_final)}")
    print(f"- QELM Runs (>=10): {len(qelm_final)}")
    print(f"- Matched Study Rows (Vertical): {len(matched_long)}")
    print(f"- Matched Architectures: {len(matched_archs)}")

if __name__ == "__main__":
    run_curation_study()