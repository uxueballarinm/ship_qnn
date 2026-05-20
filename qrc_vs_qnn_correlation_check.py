import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import ast
import os
from adjustText import adjust_text
import plotly.express as px
import plotly.graph_objects as go

# --- ADVANCED STATS HELPERS ---
def calculate_ccc(x, y):
    if len(x) < 2: return np.nan
    cor = np.corrcoef(x, y)[0, 1]
    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x), np.var(y)
    sd_x, sd_y = np.std(x), np.std(y)
    denom = var_x + var_y + (mean_x - mean_y)**2
    return (2 * cor * sd_x * sd_y) / denom if denom != 0 else 0

def pearson_no_outliers(x, y, threshold=3):
    df = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(df) < 5: return np.nan
    z_scores = np.abs(stats.zscore(df.astype(float)))
    filtered = df[(z_scores < threshold).all(axis=1)]
    return stats.pearsonr(filtered['x'], filtered['y'])[0] if len(filtered) > 3 else np.nan

def parse_config(config_str):
    if pd.isna(config_str): return []
    try:
        val = ast.literal_eval(str(config_str))
        return val if isinstance(val, list) else [val]
    except: return []

def format_heads_vertical(config_list):
    if not config_list: return "N/A"
    lines = ["<b>Circuit Architecture:</b>"]
    for i, head in enumerate(config_list):
        ansatz = head.get('ansatz', 'N/A')
        qubits = head.get('output_dim', 'N/A')
        reps = head.get('reps', 'N/A')
        h_map = head.get('map', 'N/A')
        lines.append(f"• Head {i+1}: {ansatz} (Q:{qubits}, L:{reps})")
        lines.append(f"  Map: {h_map}")
    return "<br>".join(lines)

class QNNvsQELMFullSuite:
    def __init__(self, qnn_file, qelm_file, output_dir='correlation_results'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # EXACT column names from your list
        self.param_col_name = "total params"
        self.match_keys = [
            "features", "targets", "window_size", "horizon", "predict", "norm", 
            "reconstruct_train", "reconstruct_val", "model", "heads_config", 
            "head_number", "encoding", "ansatz", "entangle", "reps", "map", 
            "reorder", self.param_col_name
        ]
        self.metrics = ['Val Global Open R2', 'Val Global Open MSE']
        
        print("Reading datasets (Memory Optimized)...")
        cols_to_load = self.match_keys + self.metrics
        
        self.df_qnn = pd.read_excel(qnn_file, usecols=lambda c: c in cols_to_load, engine='openpyxl')
        self.df_qelm = pd.read_excel(qelm_file, usecols=lambda c: c in cols_to_load, engine='openpyxl')
        self._preprocess()

    def _preprocess(self):
        for df in [self.df_qnn, self.df_qelm]:
            df.columns = [str(c).strip() for c in df.columns]
            for col in self.match_keys:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.strip().str.replace(".0", "", regex=False)

    def _get_aggregates(self, df):
        aggs = []
        keys = [k for k in self.match_keys if k in df.columns]
        for m in self.metrics:
            if m not in df.columns: continue
            df[m] = pd.to_numeric(df[m], errors='coerce')
            is_mse = 'MSE' in m.upper()
            g = df.groupby(keys)[m].agg([('mean','mean'),('median','median'),('best','min' if is_mse else 'max')]).reset_index()
            g['metric_source'] = m
            aggs.append(g)
        return pd.concat(aggs) if aggs else pd.DataFrame()

    def run_analysis(self):
        print("1. Aggregating and matching...")
        qnn_agg = self._get_aggregates(self.df_qnn)
        qelm_agg = self._get_aggregates(self.df_qelm)
        
        common_keys = list(set(qnn_agg.columns) & set(qelm_agg.columns) - {'mean', 'median', 'best'})
        merged = pd.merge(qnn_agg, qelm_agg, on=common_keys, suffixes=('_qnn', '_qelm')).dropna()
        
        if merged.empty:
            print("ERROR: No matching experiments found."); return

        print("2. Calculating stats and Excel report...")
        global_results, head_results, reps_results = [], {}, {}

        for metric in merged['metric_source'].unique():
            subset = merged[merged['metric_source'] == metric]
            for agg in ['mean', 'median', 'best']:
                res_g = self._calc_stats_row(subset, metric, agg, "All")
                if res_g: global_results.append(res_g)
                
                for h in subset['head_number'].unique():
                    h_sub = subset[subset['head_number'] == h]
                    res_h = self._calc_stats_row(h_sub, metric, agg, h)
                    if res_h:
                        if h not in head_results: head_results[h] = []
                        head_results[h].append(res_h)

                if 'reps' in subset.columns:
                    for r in subset['reps'].unique():
                        r_sub = subset[subset['reps'] == r]
                        res_r = self._calc_stats_row(r_sub, metric, agg, "All", r)
                        if res_r:
                            if r not in reps_results: reps_results[r] = []
                            reps_results[r].append(res_r)

        self._save_excel(global_results, head_results, reps_results)

        print("3. Generating Plots and HTML explorers...")
        self._generate_plots(merged)

    def _calc_stats_row(self, df, metric, agg, head, reps="All"):
        x, y = df[f'{agg}_qnn'].values, df[f'{agg}_qelm'].values
        if len(x) < 3: return None
        spearman, _ = stats.spearmanr(x, y)
        pearson, _ = stats.pearsonr(x, y)
        p_no_out = pearson_no_outliers(x, y)
        return {
            'Metric': metric, 'Head': head, 'Reps': reps, 'Aggregation': agg, 'N': len(x),
            'Spearman_Rho': round(spearman, 4), 'Pearson_With_Outliers': round(pearson, 4),
            'Pearson_No_Outliers': round(p_no_out, 4) if not np.isnan(p_no_out) else "N/A",
            'Pearson_Fisher_Z': round(np.arctanh(pearson), 4) if abs(pearson) < 1 else "N/A",
            'CCC': round(calculate_ccc(x, y), 4)
        }

    def _generate_plots(self, merged_all):
        viz_r2 = merged_all[merged_all['metric_source'] == 'Val Global Open R2'].copy()
        viz_mse = merged_all[merged_all['metric_source'] == 'Val Global Open MSE'].copy()
        
        # DYNAMIC detection of the parameter column after merge
        actual_param_col = self.param_col_name
        if actual_param_col not in viz_r2.columns:
            if f"{self.param_col_name}_qnn" in viz_r2.columns: actual_param_col = f"{self.param_col_name}_qnn"
            elif f"{self.param_col_name}_qelm" in viz_r2.columns: actual_param_col = f"{self.param_col_name}_qelm"

        for df in [viz_r2, viz_mse]:
            df['parsed_heads'] = df['heads_config'].apply(parse_config)
            df['vertical_config'] = df['parsed_heads'].apply(format_heads_vertical)
            df['mapping_id'] = range(len(df))
            if actual_param_col in df.columns:
                df[actual_param_col] = pd.to_numeric(df[actual_param_col], errors='coerce')

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(32, 10))
        
        # Subplot 1: R2
        x1, y1 = viz_r2['median_qelm'], viz_r2['median_qnn']
        ax1.scatter(x1, y1, alpha=0.6, s=100, edgecolors='k')
        ax1.plot([min(x1.min(), y1.min()), max(x1.max(), y1.max())], [min(x1.min(), y1.min()), max(x1.max(), y1.max())], 'k--', label='y=x')
        ax1.set_title("R2 Consistency (Median)"); ax1.set_xlabel("QELM"); ax1.set_ylabel("QNN")

        # Subplot 2: MSE
        x2, y2 = viz_mse['median_qelm'], viz_mse['median_qnn']
        ax2.scatter(x2, y2, alpha=0.6, s=100, color='seagreen', edgecolors='k')
        ax2.plot([0, max(x2.max(), y2.max())], [0, max(x2.max(), y2.max())], 'k--')
        ax2.set_title("MSE Consistency (Median)")

        # Subplot 3: Efficiency Frontier
        viz_r2['r2_diff'] = viz_r2['median_qnn'] - viz_r2['median_qelm']
        if actual_param_col in viz_r2.columns:
            sns.scatterplot(data=viz_r2, x=actual_param_col, y='r2_diff', ax=ax3, s=120, hue='r2_diff', palette='RdYlGn')
            ax3.axhline(0, color='black', linestyle='-')
            ax3.set_title(f"Efficiency Frontier\n({actual_param_col})")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'ranking_mapping_plots_labeled.png'))

        for mode, df_mode in [('R2', viz_r2), ('MSE', viz_mse)]:
            fig_int = px.scatter(
                df_mode, x='median_qelm', y='median_qnn', 
                hover_name="head_number", 
                hover_data=["predict", "vertical_config"],
                title=f"Interactive {mode} Anomaly Explorer (QNN vs QELM)"
            )
            fig_int.add_trace(go.Scatter(x=[df_mode['median_qelm'].min(), df_mode['median_qelm'].max()], 
                                         y=[df_mode['median_qelm'].min(), df_mode['median_qelm'].max()], 
                                         mode='lines', name='y=x', line=dict(color='black', dash='dash')))
            fig_int.write_html(os.path.join(self.output_dir, f'interactive_{mode}_explorer.html'))

    def _save_excel(self, g, h, r):
        path = os.path.join(self.output_dir, 'Comprehensive_Correlation_Report.xlsx')
        with pd.ExcelWriter(path) as writer:
            pd.DataFrame(g).to_excel(writer, sheet_name='Summary_Global', index=False)
            for h_id in sorted(h.keys(), key=lambda x: str(x)):
                pd.DataFrame(h[h_id]).to_excel(writer, sheet_name=f"Head_{h_id}"[:31], index=False)
            for r_id in sorted(r.keys(), key=lambda x: str(x)):
                pd.DataFrame(r[r_id]).to_excel(writer, sheet_name=f"Reps_{r_id}"[:31], index=False)

if __name__ == "__main__":
    # Update paths as needed
    qnn_f = r"logs\curated_studies\study_qnn_spsa.xlsx"
    qelm_f = r"logs\curated_studies\study_qelm_ridge.xlsx"
    suite = QNNvsQELMFullSuite(qnn_f, qelm_f)
    suite.run_analysis()