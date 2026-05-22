# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

def compute_fisher(qnn, qelm):
    z_qnn, z_qelm = np.arctanh(np.sqrt(np.clip(qnn, a_min=0, a_max=None))), np.arctanh(np.sqrt(np.clip(qelm, a_min=0, a_max=None)))
    return z_qnn.corr(z_qelm)

def compute_ccc(qnn, qelm):
    qnn, qelm = np.array(qnn), np.array(qelm)
    mu_true, mu_pred = np.mean(qnn), np.mean(qelm)
    var_true, var_pred = np.var(qnn), np.var(qelm)
    covariance = np.cov(qnn, qelm, ddof=0)[0, 1]
    num = 2 * covariance
    den = var_true + var_pred + (mu_true - mu_pred)**2
    return num / den

def compute_correlations(df, mode, qnn_agg, qelm_agg, threshold=None):

    qnn_metric, qelm_metric = 'qnn_'+mode+'_'+qnn_agg, 'qelm_'+mode+'_'+qelm_agg
    if threshold:
        if mode=='r2':
            selected = df[qnn_metric] >= threshold 
            df_selected = df[selected].sort_values(by=qnn_metric).copy()
        else: 
            selected = df[qnn_metric] <= threshold
            df_selected = df[selected].sort_values(by=qnn_metric, ascending=False).copy()
    else: 
        df_selected = df.sort_values(by=qnn_metric).copy() if mode=='r2' else df.sort_values(by=qnn_metric, ascending=False).copy()
    qnn, qelm = df_selected[qnn_metric], df_selected[qelm_metric]

    num_configs = len(qnn)
    qnn_mean, qelm_mean = qnn.mean(), qelm.mean()
    qnn_worse, qelm_worse = qnn.iloc[0], qelm.iloc[0]
    qnn_std, qelm_std = df_selected['qnn_'+mode+'_std'].mean(), df_selected['qelm_'+mode+'_std'].mean()

    pearson = qelm.corr(qnn)
    spearman = qelm.corr(qnn, method='spearman')
    pearson_fisher = compute_fisher(qnn, qelm) if mode=='r2' else None
    ccc = compute_ccc(qnn, qelm)

    return {'num_configs':num_configs, 'r':pearson, 'r_fisher':pearson_fisher, 'ccc':ccc, 'rs':spearman, 'mean':(qnn_mean, qelm_mean), 'worse':(qnn_worse, qelm_worse), 'std': (qnn_std, qelm_std)}


def compare_by(df, mode, qnn_agg, qelm_agg, group_cols, by='reps', only=None, outliers=None, grid=True, distribution_plot=False, save=False):

    qnn_metric, qelm_metric = 'qnn_'+mode+'_'+qnn_agg, 'qelm_'+mode+'_'+qelm_agg
    if only: df = df[df[only[0]]==only[1]].copy()

    if grid:
        other_cols = [col for col in group_cols if col != by]
        expected_count = df[by].nunique()
        df = df.groupby(other_cols).filter(lambda x: x[by].nunique() == expected_count)
        if df.empty or len(df)<3:
            print(f"Warning: Not enough valid grid configurations found across all '{by}' values.")
            if not df.empty and qnn_agg==qelm_agg:
                for uniq in df[by].unique().tolist():
                    reduced = df[df[by]==uniq].copy()
                    print(f"{uniq}: {float((reduced['qelm_'+mode+'_'+qnn_agg]-reduced['qnn_'+mode+'_'+qnn_agg])/reduced['qnn_'+mode+'_'+qnn_agg]):.4f} %")
            return
        
    unique_vals = sorted(df[by].unique().tolist())
    results = {'val': [], 'r': [], 'rs': [], 'ccc': [], 'r_fisher': [], 'r_select': [], 'rs_select': [], 'ccc_select': [], 'r_fisher_select': []}

    for val in unique_vals:
        df_reduced = df[df[by] == val].copy() 
        metrics = compute_correlations(df_reduced, mode, qnn_agg, qelm_agg)

        if outliers[0] == 'threshold':
            threshold = outliers[1]
        elif outliers[0] == 'percent':
            threshold = df_reduced[qnn_metric].quantile(outliers[1])
        elif outliers[0] == 'num':
            threshold = df_reduced.sort_values(by=qnn_metric)[qnn_metric].iloc[int(outliers[1])] if mode=='r2' else df_reduced.sort_values(by=qnn_metric)[qnn_metric].iloc[-int(outliers[1])]

        metrics_select = compute_correlations(df_reduced, mode, qnn_agg, qelm_agg, threshold)
        print(f"Computing correlations for {by} = {val}:",
              f"\n    > all {metrics['num_configs']:2d} configurations ({metrics['mean'][0]:.4f}, {metrics['mean'][1]:.4f}) ({metrics['std'][0]:.4f}, {metrics['std'][1]:.4f}) ({metrics['worse'][0]:.4f}, {metrics['worse'][1]:.4f})",
              f"\n    > top {metrics_select['num_configs']:2d} configurations ({metrics_select['mean'][0]:.4f}, {metrics_select['mean'][1]:.4f}) ({metrics_select['std'][0]:.4f}, {metrics_select['std'][1]:.4f}) ({metrics_select['worse'][0]:.4f}, {metrics_select['worse'][1]:.4f})")

        if distribution_plot == True:
            plt.figure(figsize=(5, 4))
            plt.scatter(df_reduced[qnn_metric], df_reduced[qelm_metric], color='blue', alpha=0.7)
            mask = df_reduced[qnn_metric] >= threshold if mode=='r2' else df_reduced[qnn_metric] <= threshold
            plt.scatter(df_reduced[qnn_metric][mask], df_reduced[qelm_metric][mask], color='red', alpha=0.7)
            plt.title(f"Rank Spread for {val}")
            plt.xlabel("QNN Median R2")
            plt.ylabel("QELM Median R2")
            plt.axis('equal')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.show()

        results['val'].append(val)
        results['r'].append(metrics['r'])
        results['rs'].append(metrics['rs'])
        results['ccc'].append(metrics['ccc'])
        results['r_fisher'].append(metrics['r_fisher'])
        results['r_select'].append(metrics_select['r'])
        results['rs_select'].append(metrics_select['rs'])
        results['ccc_select'].append(metrics_select['ccc'])
        results['r_fisher_select'].append(metrics_select['r_fisher'])

    # 3. Generate the Plot
    plt.figure(figsize=(10, 6))

    # Plot each metric
    plt.plot(results['val'], results['r'], marker='o', linewidth=2, label='Pearson (r)', color='C0')
    plt.plot(results['val'], results['rs'], marker='s', linewidth=2, label='Spearman (rs)', color='C1')
    plt.plot(results['val'], results['ccc'], marker='^', linewidth=2, label='CCC (x=y)', color='C2')
    # plt.plot(results['val'], results['r_fisher'], marker='*', linewidth=2, label='Pearson (Fisher)', color='C3')
    plt.plot(results['val'], results['r_select'], marker='o', linestyle='--', linewidth=2, color='C0')
    plt.plot(results['val'], results['rs_select'], marker='s', linestyle='--', linewidth=2, color='C1')
    plt.plot(results['val'], results['ccc_select'], marker='^', linestyle='--', linewidth=2, color='C2')
    # plt.plot(results['val'], results['r_fisher_select'], marker='*', linewidth=2, color='C3')
    
    # Formatting the plot
    plt.xlabel(by.capitalize(), fontsize=12)
    plt.ylabel('Correlation / Agreement Score', fontsize=12)
    
    # Dynamic title based on inputs
    additional = ""
    if grid: additional+=" (symmetric)"
    if outliers: additional+=f" ({outliers[0]} {outliers[1]})"
    if only: additional+=f" ({only[0]}={only[1]})"
    title = (f"QNN vs QELM Correlation across '{by}' ({mode.upper()}){additional}\n"
             f"QNN Aggregation: {qnn_agg} | QELM Aggregation: {qelm_agg}")
    plt.title(title, fontsize=14, pad=15)
    
    # X-axis ticks (ensuring it only shows actual values if they are integers)
    if all(isinstance(x, (int, np.integer)) for x in results['val']):
        plt.xticks(results['val'])
    plt.ylim([-0.05, 1.05])
        
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    os.makedirs('figures/correlation', exist_ok=True)
    name = f"corr_by_{by}_{mode}_{qnn_agg}_{qelm_agg}"
    if grid: name+="_symmetric"
    if outliers: name+=f"_{outliers[0]}{outliers[1]}"
    if only: name+=f"_{only[0]}{only[1]}"
    if save==True: plt.savefig(f'figures/correlation/'+name+'.png')
    plt.show()

def plot_correlation_heatmap(df, filters=None, method='spearman', save=False):
    """
    Plots a heatmap showing correlations between metrics.
    filters: dict containing column filtering pairs, e.g. {'reps': 1, 'encoding': 'compact'}
    method: 'spearman' or 'pearson'
    """
    df_filtered = df.copy()
    
    # Apply dynamic filters if provided
    filter_title_parts = []
    if filters:
        for col, val in filters.items():
            if col in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[col] == val]
                filter_title_parts.append(f"{col}={val}")
            else:
                print(f"Warning: Filter column '{col}' not found in DataFrame.")
                
    if df_filtered.empty:
        print("Error: The data selection is empty with the provided filters. Heatmap skipped.")
        return

    # Select columns to check correlation between QNN and QELM metrics
    corr_cols = [
        'qnn_r2_mean', 'qelm_r2_mean',
        'qnn_r2_median', 'qelm_r2_median',
        'qnn_mse_mean', 'qelm_mse_mean',
        'qnn_mse_median', 'qelm_mse_median'
    ]
    
    # Ensure they exist in the dataframe before correlating
    corr_cols = [c for c in corr_cols if c in df_filtered.columns]
    
    # Calculate correlation matrix
    corr_matrix = df_filtered[corr_cols].corr(method=method)
    
    # Plotting the Heatmap
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', fmt='.3f', vmin=-1, vmax=1, square=True)
    
    filter_str = f" ({', '.join(filter_title_parts)})" if filter_title_parts else " (All Data)"
    plt.title(f"QNN vs QELM metric correlation heatmap{filter_str}\nMethod: {method.capitalize()}", fontsize=13, pad=12)
    plt.tight_layout()
    
    if save:
        os.makedirs('figures/correlation', exist_ok=True)
        fname = f"heatmap_{method}_" + "_".join(filter_title_parts).replace('=', '') + ".png"
        plt.savefig(f'figures/correlation/{fname}')
    plt.show()
def plot_regression_with_threshold(df, mode='r2', agg='median', filters=None, threshold=None, tolerance=None, save=False):
    """
    Plots individual configurations against an ideal x=y line and an actual linear trend regression.
    Maintains standard axis scopes, highlights the REJECTED zone in red based on threshold rules,
    and applies a gray-shaded performance equivalence band around the identity line.
    """
    df_filtered = df.copy()
    filter_strings = []
    if filters:
        for col, val in filters.items():
            df_filtered = df_filtered[df_filtered[col] == val]
            filter_strings.append(f"{col}={val}")
            
    if df_filtered.empty:
        print("Error: Data selection is empty after applying filters.")
        return
        
    x_col, y_col = f'qnn_{mode}_{agg}', f'qelm_{mode}_{agg}'
    
    if mode == 'r2':
        limits = [0.0, 1.0]
    else:
        limits = [0.0, 100.0]

    # Dynamically set an indifference tolerance range if not explicitly specified
    if tolerance is None:
        tolerance = 0.05 if mode == 'r2' else 5.0

    plt.figure(figsize=(9, 8))
    
    # Map configuration settings to names
    def assign_config(row):
        enc = row.get('encoding', '')
        rep = row.get('reps', '')
        if enc == 'serial' and rep == 1: return 'serial1'
        if enc == 'serial' and rep == 3: return 'serial3'
        if enc == 'compact' and rep == 1: return 'compact1'
        if enc == 'compact' and rep == 3: return 'compact3'
        return f"{enc}{rep}"

    def assign_head(row):
        hn = row.get('head_number', '')
        return f"head{hn}"

    df_filtered['config_cat'] = df_filtered.apply(assign_config, axis=1)
    df_filtered['head_cat'] = df_filtered.apply(assign_head, axis=1)

    # Shaded region for threshold (Now highlights the REJECTED zone in red)
    if threshold is not None:
        if mode == 'r2':
            # For R2, smaller than threshold is rejected
            plt.axvspan(limits[0], threshold, color='#ffcccc', alpha=0.4)
            accepted_df = df_filtered[df_filtered[x_col] >= threshold]
        else:
            # For MSE, larger than threshold is rejected
            plt.axvspan(threshold, limits[1], color='#ffcccc', alpha=0.4)
            accepted_df = df_filtered[df_filtered[x_col] <= threshold]
        data_to_correlate = accepted_df
        title_context = f"Accepted Points Zone (N={len(accepted_df)})"
    else:
        data_to_correlate = df_filtered
        title_context = "All Data Points"

    # Plot the indifference tolerance band around the ideal y=x identity line
    x_ref = np.linspace(limits[0], limits[1], 100)
    plt.plot(x_ref, x_ref + tolerance, color='gray', linestyle=':', linewidth=1.2)
    plt.plot(x_ref, x_ref - tolerance, color='gray', linestyle=':', linewidth=1.2)
    plt.fill_between(x_ref, x_ref - tolerance, x_ref + tolerance, color='gray', alpha=0.15)

    # Style Definition Mappings
    color_map = {
        'serial1': '#1f77b4',   # tab:blue
        'serial3': '#ff7f0e',   # tab:orange
        'compact1': '#2ca02c',  # tab:green
        'compact3': '#d62728'   # tab:red
    }
    marker_map = {
        'head1': 'o',
        'head2': 's',
        'head3': '^',
        'head4': 'D'
    }

    # Plot structured point categories
    for (config_val, head_val), sub_df in df_filtered.groupby(['config_cat', 'head_cat']):
        color = color_map.get(config_val, '#7f7f7f')
        marker = marker_map.get(head_val, 'x')
        plt.scatter(sub_df[x_col], sub_df[y_col], color=color, marker=marker, alpha=0.8, s=55, edgecolors='none')

    # Compute correlation properties for the chosen targeted area dataframe
    if len(data_to_correlate) > 1:
        r_pearson = data_to_correlate[x_col].corr(data_to_correlate[y_col])
        r_spearman = data_to_correlate[x_col].corr(data_to_correlate[y_col], method='spearman')
        
        m, b = np.polyfit(data_to_correlate[x_col], data_to_correlate[y_col], 1)
        x_trend = np.linspace(limits[0], limits[1], 100)
        trend_line, = plt.plot(x_trend, m * x_trend + b, color='darkorange', linestyle='-', linewidth=2)
    else:
        r_pearson, r_spearman = np.nan, np.nan
        trend_line = None

    # Ideal Identity Reference line trace
    ideal_line, = plt.plot(limits, limits, color='black', linestyle='--', linewidth=1.5)
        
    # Manual Custom Legend Reconstruction
    legend_handles = [ideal_line]
    if trend_line is not None:
        trend_line.set_label(f'Trend (y = {m:.2f}x + {b:.2f})')
        legend_handles.append(trend_line)
        
    legend_handles.append(mpatches.Patch(color='gray', alpha=0.15, label=f'Tolerance Band (±{tolerance})'))
    
    if threshold is not None:
        lbl = f'Rejected Region (QNN < {threshold})' if mode == 'r2' else f'Rejected Region (QNN > {threshold})'
        legend_handles.append(mpatches.Patch(color='#ffcccc', alpha=0.4, label=lbl))

    # Incorporate configuration labels into legend
    for cv in ['serial1', 'serial3', 'compact1', 'compact3']:
        if cv in df_filtered['config_cat'].values:
            legend_handles.append(Line2D([0], [0], marker='o', color='w', label=cv, markerfacecolor=color_map[cv], markersize=8))
            
    # Incorporate unique head markers into legend
    for hv in ['head1', 'head2', 'head3', 'head4']:
        if hv in df_filtered['head_cat'].values:
            legend_handles.append(Line2D([0], [0], marker=marker_map[hv], color='w', label=hv, markerfacecolor='gray', markersize=8))

    filter_str = f" [{', '.join(filter_strings)}]" if filter_strings else ""
    title_text = (f"QNN vs QELM {mode.upper()} ({agg.capitalize()}){filter_str}\n"
                  f"Correlation for {title_context} -> Pearson r: {r_pearson:.3f} | Spearman rs: {r_spearman:.3f}")
    
    plt.title(title_text, fontsize=11, pad=12)
    plt.xlabel(f"QNN {mode.upper()}", fontsize=11)
    plt.ylabel(f"QELM {mode.upper()}", fontsize=11)
    plt.xlim(limits)
    plt.ylim(limits)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(handles=legend_handles, loc='upper left', fontsize=9)
    plt.tight_layout()
    
    if save:
        os.makedirs('figures/correlation', exist_ok=True)
        name = f"regression_{mode}_{agg}_" + "_".join(filter_strings).replace('=', '') + ".png"
        plt.savefig(f'figures/correlation/{name}')
    plt.show()
def generate_interactive_html(df, mode='r2', agg='median', filters=None, threshold=None, tolerance=None, filename="interactive_explorer.html"):
    """
    Creates an interactive Plotly HTML visualization. 
    Maintains standard axis boundaries, applies custom color configurations, custom symbols per head number,
    shades the REJECTED zone in red based on threshold rules, and visualizes a gray-shaded performance equivalence 
    tolerance band around the ideal y=x identity line.
    """
    df_filtered = df.copy()
    filter_strings = []
    if filters:
        for col, val in filters.items():
            df_filtered = df_filtered[df_filtered[col] == val]
            filter_strings.append(f"{col}={val}")
            
    if df_filtered.empty:
        print("Error: Empty subset for Interactive Viewer.")
        return

    x_col, y_col = f'qnn_{mode}_{agg}', f'qelm_{mode}_{agg}'
    x_vals, y_vals = df_filtered[x_col], df_filtered[y_col]
    
    # Calculate live empirical matching coefficients for display in the interactive title
    r_pearson = x_vals.corr(y_vals)
    r_spearman = x_vals.corr(y_vals, method='spearman')
    
    # Handle specific range limits (0 to 1 for R2, 0 to 100 for MSE)
    if mode == 'r2':
        limits = [0.0, 1.0]
    else:
        limits = [0.0, 100.0]

    # Set indifference tolerance range if not specified
    if tolerance is None:
        tolerance = 0.05 if mode == 'r2' else 5.0
    
    # 1. Map Configurations to Color Categories & Heads to Symbols
    def assign_config(row):
        enc = row.get('encoding', '')
        rep = row.get('reps', '')
        if enc == 'serial' and rep == 1: return 'serial1'
        if enc == 'serial' and rep == 3: return 'serial3'
        if enc == 'compact' and rep == 1: return 'compact1'
        if enc == 'compact' and rep == 3: return 'compact3'
        return f"{enc}{rep}"

    def assign_head(row):
        hn = row.get('head_number', '')
        return f"head{hn}"

    df_filtered['Configuration'] = df_filtered.apply(assign_config, axis=1)
    df_filtered['Heads Config'] = df_filtered.apply(assign_head, axis=1)
    
    df_filtered['Hover text'] = (
        "<b>Ansatz:</b> " + df_filtered['ansatz'].astype(str) + "<br>" +
        "<b>Encoding:</b> " + df_filtered['encoding'].astype(str) + "<br>" +
        "<b>Reps:</b> " + df_filtered['reps'].astype(str) + "<br>" +
        "<b>Heads Config:</b> " + df_filtered['head_number'].astype(str) + "<br>" +
        "<b>Features:</b> " + df_filtered['features'].astype(str) + "<br>" +
        "<b>Map:</b> " + df_filtered['map'].astype(str)
    )
    
    filter_str = f" ({', '.join(filter_strings)})" if filter_strings else ""
    title_text = (f"Interactive QNN vs QELM Explorer{filter_str}<br>"
                  f"Live Correlation Score -> Pearson r: {r_pearson:.3f} | Spearman rs: {r_spearman:.3f}")

    # Color and Symbol configurations matching matplotlib script changes
    color_map = {
        'serial1': '#1f77b4',
        'serial3': '#ff7f0e',
        'compact1': '#2ca02c',
        'compact3': '#d62728'
    }
    symbol_map = {
        'head1': 'circle',
        'head2': 'square',
        'head3': 'triangle-up',
        'head4': 'diamond'
    }

    fig = px.scatter(
        df_filtered, x=x_col, y=y_col, 
        color='Configuration',
        symbol='Heads Config',
        color_discrete_map=color_map,
        symbol_map=symbol_map,
        title=title_text,
        labels={x_col: f"QNN {mode.upper()} ({agg})", y_col: f"QELM {mode.upper()} ({agg})"},
        hover_name='model_id' if 'model_id' in df_filtered.columns else None
    )
    
    fig.update_traces(text=df_filtered['Hover text'], hovertemplate="%{text}<br><br><b>X (QNN):</b> %{x:.4f}<br><b>Y (QELM):</b> %{y:.4f}")
    
    # 2. Add Gray Tolerance Band Region around y=x
    fig.add_trace(go.Scatter(
        x=[limits[0], limits[1], limits[1], limits[0]],
        y=[limits[0]-tolerance, limits[1]-tolerance, limits[1]+tolerance, limits[0]+tolerance],
        fill='toself',
        fillcolor='rgba(128, 128, 128, 0.15)',
        line=dict(color='rgba(128, 128, 128, 0.4)', width=1, dash='dot'),
        name=f'Tolerance Band (±{tolerance})',
        showlegend=True
    ))

    # Ideal Identity Reference Line Trace
    fig.add_trace(go.Scatter(x=limits, y=limits, mode='lines', name='Ideal (x=y)', line=dict(color='black', dash='dash')))
    
    # Empirical trend regression line trace tracking
    if len(df_filtered) > 1:
        m, b = np.polyfit(x_vals, y_vals, 1)
        fig.add_trace(go.Scatter(x=limits, y=[m*limits[0]+b, m*limits[1]+b], 
                                 mode='lines', name=f'Trend (y={m:.2f}x+{b:.2f})', line=dict(color='orange')))
        
    # 3. Add Shaded Region Highlighting the REJECTED area in light red
    if threshold is not None:
        if mode == 'r2':
            # For R2, smaller than threshold is rejected
            fig.add_vrect(x0=limits[0], x1=threshold, fillcolor="red", opacity=0.08, layer="below", line_width=0, annotation_text="Rejected Region")
        else:
            # For MSE, larger than threshold is rejected
            fig.add_vrect(x0=threshold, x1=limits[1], fillcolor="red", opacity=0.08, layer="below", line_width=0, annotation_text="Rejected Region")

    fig.update_layout(xaxis=dict(range=limits), yaxis=dict(range=limits), width=900, height=800)
    
    os.makedirs('correlation_results_1', exist_ok=True)
    full_path = os.path.join('correlation_results_1', filename)
    fig.write_html(full_path)
    print(f"Success! Interactive HTML explorer generated at: {full_path}")
# %%
if __name__ == "__main__":

    group_cols = ['map', 'reps', 'ansatz', 'use_hadamard', 'encoding', 'features', 'head_number']
    qelm_file = 'logs\experiments_systematic\correlation_study\correlation_qelm_uniform.xlsx'
    qnn_file = 'logs\experiments_systematic\correlation_study\correlation_qnn_identity.xlsx'
    df_qelm = pd.read_excel(qelm_file, sheet_name='Sheet1')
    df_qnn = pd.read_excel(qnn_file, sheet_name='Sheet1')

    qelm_agg = df_qelm.groupby(group_cols).agg(
        qelm_mse_mean=('Val Global Open MSE', 'mean'),
        qelm_r2_mean=('Val Global Open R2', 'mean'),
        qelm_mse_median=('Val Global Open MSE', 'median'),
        qelm_r2_median=('Val Global Open R2', 'median'),
        qelm_mse_best=('Val Global Open MSE', 'min'),
        qelm_r2_best=('Val Global Open R2', 'max'),
        qelm_r2_std=('Val Global Open R2', 'std'), 
        count_qelm=('Val Global Open R2', 'count')
    ).reset_index()

    qnn_agg = df_qnn.groupby(group_cols).agg(
        qnn_mse_mean=('Val Global Open MSE', 'mean'),
        qnn_r2_mean=('Val Global Open R2', 'mean'),
        qnn_mse_median=('Val Global Open MSE', 'median'),
        qnn_r2_median=('Val Global Open R2', 'median'),
        qnn_mse_best=('Val Global Open MSE', 'min'),
        qnn_r2_best=('Val Global Open R2', 'max'),
        qnn_r2_std=('Val Global Open R2', 'std'), 
        count_qnn=('Val Global Open R2', 'count')
    ).reset_index()

    # Filter by the minimum sample requirements
    qelm_filtered = qelm_agg[qelm_agg['count_qelm'] >= 30]
    qnn_filtered = qnn_agg[qnn_agg['count_qnn'] >= 10]
    merged = pd.merge(qelm_filtered, qnn_filtered, on=group_cols)

    # %%
    compare_by(
        df=merged, 
        mode='r2', 
        qnn_agg='median', 
        qelm_agg='median',  # <-- Change this from 'median' to 'best'
        group_cols=group_cols, 
        by='reps',
        outliers=('percent', 0), # Or your preferred threshold
        only=('encoding', 'compact'),
        grid=False,
        save=False
    )
    # %%
    # Example 1: Heatmap filtering only configs with reps = 1 and encoding = 'compact'
    plot_correlation_heatmap(
        df=merged, 
        # filters={'reps': 1, 'encoding': 'serial', 'head_number': 1}, 
        method='pearson'


    )
    # %%
    # Example 2: Heatmap filtering only configs with head_number = 1 and encoding = 'compact'
    plot_correlation_heatmap(
        df=merged, 
        filters={'head_number': 1}, 
        method='spearman'
    )
    # %%
    plot_regression_with_threshold(
        df=merged,
        mode='r2',
        agg='mean',
        # filters={'reps': 1, 'encoding': 'serial'},
        threshold=0.5,  # Example threshold for MSE

    )
    # %%
    generate_interactive_html(
        df=merged,
        mode='mse',
        agg='mean',
        # filters={'head_number': 1},
        filename="head1_mse_explorer.html"
        
    )