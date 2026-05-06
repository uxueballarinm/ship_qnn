import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import ast
import os
from adjustText import adjust_text # Preserving your specific import
import plotly.express as px
import plotly.graph_objects as go

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

def run_full_analysis():
    file_path = r'logs\QRC_VS_SPSA.xlsx'
    output_dir = 'correlation_results'
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(file_path):
        print(f"ERROR: File not found at {file_path}"); return

    # 1. Data Loading & Matching
    print(f"Reading data...")
    df = pd.read_excel(file_path, sheet_name='Hoja1')
    df.columns = [str(c).strip() for c in df.columns]
    param_col = 'total params'

    spsa_df = df[df['optimizer'].str.lower() == 'spsa'].copy()
    ridge_df = df[df['optimizer'].str.lower() == 'ridge'].copy()
    match_cols = ['heads_config', 'window_size', 'horizon', 'predict']
    
    merged = pd.merge(
        spsa_df[match_cols + ['global open MSE_Mean', 'global open R2_Mean', param_col]],
        ridge_df[match_cols + ['global open MSE_Mean', 'global open R2_Mean']],
        on=match_cols, suffixes=('_spsa', '_qrc')
    ).reset_index(drop=True)
    
    merged['mapping_id'] = merged.index
    merged['parsed_heads'] = merged['heads_config'].apply(parse_config)
    merged['vertical_config'] = merged['parsed_heads'].apply(format_heads_vertical)

    # --- CALCULATIONS ---
    # R2 Difference (Efficiency Metric)
    merged['r2_diff'] = merged['global open R2_Mean_spsa'] - merged['global open R2_Mean_qrc']
    
    # R2: 10% Outlier Logic
    merged['is_r2_outlier'] = np.abs(merged['r2_diff']) > 0.1
    
    # MSE: 10% Outlier Logic
    max_mse_observed = max(merged['global open MSE_Mean_spsa'].max(), merged['global open MSE_Mean_qrc'].max())
    mse_threshold = 0.1 * max_mse_observed
    merged['mse_diff'] = merged['global open MSE_Mean_spsa'] - merged['global open MSE_Mean_qrc']
    merged['is_mse_outlier'] = np.abs(merged['mse_diff']) > mse_threshold

    # Regression lines for static plots
    m1, b1 = np.polyfit(merged['global open R2_Mean_qrc'], merged['global open R2_Mean_spsa'], 1)
    m2, b2 = np.polyfit(merged['global open MSE_Mean_qrc'], merged['global open MSE_Mean_spsa'], 1)

    # 2. Static Heatmap (Preserved)
    plt.figure(figsize=(9, 7))
    corr_cols = ['global open R2_Mean_spsa', 'global open R2_Mean_qrc', 'global open MSE_Mean_spsa', 'global open MSE_Mean_qrc']
    sns.heatmap(merged[corr_cols].corr(method='spearman'), annot=True, cmap='RdYlGn', fmt='.3f')
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()

    # 3. STATIC LOCAL PLOTS (Now 3 Subplots)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(32, 10))
    
    # --- R2 Subplot ---
    x1, y1 = merged['global open R2_Mean_qrc'], merged['global open R2_Mean_spsa']
    ax1.scatter(x1, y1, c=['red' if o else 'royalblue' for o in merged['is_r2_outlier']], alpha=0.5, s=100)
    ax1.plot([min(x1.min(), y1.min()), max(x1.max(), y1.max())], [min(x1.min(), y1.min()), max(x1.max(), y1.max())], 'k--', label='Ideal (y=x)')
    ax1.plot(x1, m1*x1 + b1, color='firebrick', alpha=0.3, label='Actual Trend')
    texts1 = [ax1.text(x1[i], y1[i], f"#{i}", fontsize=8) for i in range(len(merged))]
    adjust_text(texts1, ax=ax1, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
    ax1.set_title("R2 Consistency (Ideal Line y=x)"); ax1.legend()

    # --- MSE Subplot ---
    x2, y2 = merged['global open MSE_Mean_qrc'], merged['global open MSE_Mean_spsa']
    ax2.scatter(x2, y2, c=['orange' if o else 'seagreen' for o in merged['is_mse_outlier']], alpha=0.5, s=100)
    ax2.plot([0, max_mse_observed], [0, max_mse_observed], 'k--', label='Ideal (y=x)')
    ax2.plot(x2, m2*x2 + b2, color='firebrick', alpha=0.3, label='Actual Trend')
    texts2 = [ax2.text(x2[i], y2[i], f"#{i}", fontsize=8) for i in range(len(merged))]
    adjust_text(texts2, ax=ax2, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
    ax2.set_title(f"MSE Consistency (Threshold: {mse_threshold:.2f})"); ax2.legend()

    # --- NEW: Efficiency Frontier Subplot ---
    # Colors: Green where Ridge wins (Negative Diff), Red where SPSA wins (Positive Diff)
    eff_colors = ['#2ecc71' if d < 0 else '#e74c3c' for d in merged['r2_diff']]
    ax3.scatter(merged[param_col], merged['r2_diff'], c=eff_colors, s=120, edgecolors='k', alpha=0.7)
    ax3.axhline(0, color='black', linestyle='-', linewidth=1.5)
    ax3.set_title("Efficiency Frontier\n(Below 0 = Ridge is Better)")
    ax3.set_xlabel("Total Parameters (Q + C)"); ax3.set_ylabel("R2 Delta (SPSA - Ridge)")
    texts3 = [ax3.text(merged[param_col][i], merged['r2_diff'][i], f"#{i}", fontsize=8) for i in range(len(merged))]
    adjust_text(texts3, ax=ax3, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    for ax in [ax1, ax2, ax3]: ax.margins(0.2); ax.grid(alpha=0.1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ranking_mapping_plots_labeled.png'))

    # 4. INTERACTIVE HTMLs (3 Files)
    # R2 and MSE Explorers
    for mode in ['R2', 'MSE']:
        is_r2 = (mode == 'R2')
        out_col = 'is_r2_outlier' if is_r2 else 'is_mse_outlier'
        cx, cy = (x1, y1) if is_r2 else (x2, y2)
        cm, cb = (m1, b1) if is_r2 else (m2, b2)
        
        fig_int = px.scatter(
            merged, x=cx, y=cy, color=out_col, hover_name="mapping_id",
            hover_data={param_col: True, "predict": True, "vertical_config": True, out_col: False},
            title=f"Interactive {mode} Anomaly Explorer",
            color_discrete_map={True: "red" if is_r2 else "orange", False: "royalblue" if is_r2 else "seagreen"}
        )
        fig_int.add_trace(go.Scatter(x=[cx.min(), cx.max()], y=[cx.min(), cx.max()], mode='lines', name='Ideal (y=x)', line=dict(color='black', dash='dash')))
        fig_int.add_trace(go.Scatter(x=[cx.min(), cx.max()], y=[cm*cx.min()+cb, cm*cx.max()+cb], mode='lines', name='Actual Trend', line=dict(color='firebrick', width=1)))
        fig_int.update_layout(hoverlabel=dict(align="left", bgcolor="white"))
        fig_int.write_html(os.path.join(output_dir, f'interactive_{mode}_explorer.html'))

    # NEW: Efficiency Explorer HTML
    fig_eff = px.scatter(
        merged, x=param_col, y="r2_diff", color="r2_diff", 
        color_continuous_scale="RdYlGn_r", # Visual spectrum from QRC win (Green) to SPSA win (Red)
        hover_name="mapping_id",
        hover_data={param_col: True, "predict": True, "vertical_config": True, "r2_diff": ":.4f"},
        title="Efficiency Frontier Explorer: Impact of Complexity on Optimizer Success",
        labels={"r2_diff": "R2 Delta (SPSA-Ridge)", param_col: "Total Params"}
    )
    fig_eff.add_hline(y=0, line_dash="solid", line_color="black")
    fig_eff.update_layout(hoverlabel=dict(align="left", bgcolor="white"))
    fig_eff.write_html(os.path.join(output_dir, 'interactive_efficiency_explorer.html'))

    # 5. Final Output
    merged.to_csv(os.path.join(output_dir, 'spsa_vs_qrc_comprehensive_analysis.csv'), index=False)
    print(f"\nFinal Success! Created Heatmap, Labeled PNG (3 subplots), and 3 Interactive HTML Explorers.")

if __name__ == "__main__":
    run_full_analysis()