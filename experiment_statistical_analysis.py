import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
import warnings
import os
import argparse
from openpyxl import load_workbook
from openpyxl.styles import PatternFill
from openpyxl.formatting.rule import CellIsRule
from statsmodels.stats.multitest import multipletests
import openpyxl.utils as utils
import statsmodels.api as sm
import patsy
from scipy import linalg
from statsmodels.formula.api import ols
# 1. STYLE & FONT CONFIGURATION
warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Cambria", "Times New Roman"],
    "axes.titleweight": "bold",
    "font.size": 11
})

class QNNAnalyzer:
    def __init__(self, file_path, sheet_name=0, save_enabled=False):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.save_enabled = save_enabled
        self.df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
        
        self.my_colors = ["#4DB6AC", "#00695C", "#004D40", "#B2DFDB", "#80CBC4"]
        self.line_color = "#002B2E" 
        
        if self.save_enabled:
            base_dir = os.path.dirname(os.path.abspath(file_path))
            self.out_dir = os.path.join(base_dir, "statistical_analysis_results")
            os.makedirs(self.out_dir, exist_ok=True)
            self.log_file = open(os.path. join(self.out_dir, "results_log.txt"), "a", encoding="utf-8")
            
            # FIX: Initialize the writer and ensure at least one sheet exists
            report_path = os.path.join(self.out_dir, "analysis_report.xlsx")
            self.writer = pd.ExcelWriter(report_path, engine='openpyxl')
            
            # Create a blank sheet to prevent the IndexError
            temp_df = pd.DataFrame({"Status": ["Analysis Started"]})
            temp_df.to_excel(self.writer, sheet_name="Summary_Overview", index = False)
        else:
            self.log_file, self.writer = None, None

        self._preprocess()
    
    def _preprocess(self):
        if 'map' in self.df.columns:
            self.df['map'] = self.df['map'].astype(str).str.replace(" ", "")
        categorical_cols = ['reps', 'encoding', 'ansatz', 'entangle', 'head_number']
        def safe_format(val):
            if pd.isna(val): return "NaN"
            try:
                # If it's a number, check if it's effectively an integer (like 1.0)
                num = float(val)
                if num.is_integer():
                    return str(int(num))
                return str(num) # Keeps 0.01 as '0.01'
            except (ValueError, TypeError):
                return str(val).strip()

        for col in categorical_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(safe_format)

    def logger(self, text):
        print(text)
        if self.log_file: self.log_file.write(text + "\n")
    def check_normality_full(self, data, label):
        self.logger(f"\n>>> Normality Check for {label} (N={len(data)})")
        N = len(data)
        if N < 3:
            self.logger("   STATUS: NOT NORMAL (Insufficient data points)")
            return False
        if N < 50:
            # Standard Shapiro-Wilk for smaller groups (Config-level)
            stat, p = stats.shapiro(data)
        else:
            # D'Agostino's K^2 for larger groups (Study-level)
            stat, p = stats.normaltest(data)
        alpha = 0.05
        is_gaussian = p > alpha
        self.logger(f"   Normality Test ({label}={N}): p={p:.4f} -> {'NORMAL' if is_gaussian else 'NOT NORMAL'}")
        return is_gaussian
    def run_study(self, group_by, metric):
        header = f"\n{'='*75}\nSTUDY: {metric} by {group_by}\n{'='*75}"
        self.logger(header)
        
        self.df[metric] = pd.to_numeric(self.df[metric], errors='coerce')
        clean_df = self.df.dropna(subset=[metric]).copy()
        is_error_metric = any(x in metric.upper() for x in ["MSE", "LOSS", "MAE"])
        unique_groups = clean_df[group_by].unique()
        norm_results = {idx: self.check_normality_full(clean_df[clean_df[group_by]==idx][metric], idx) for idx in unique_groups}
        
        pct_normal = sum(norm_results.values()) / len(norm_results)
        use_parametric = sum(norm_results.values()) / len(norm_results) > 0.95  # Strict 95% threshold
        test_type = "Parametric (T-Test)" if use_parametric else "Non-Parametric (Mann-Whitney)"
        self.logger(f"\n>>> GLOBAL STRATEGY: {test_type} (Normality Pass Rate: {pct_normal:.1%})")
        groups_data = [group[metric].values for name, group in clean_df.groupby(group_by)]
        if use_parametric:
            stat, p_omnibus = stats.f_oneway(*groups_data)
            omnibus_name = "ANOVA"
        else:
            stat, p_omnibus = stats.kruskal(*groups_data)
            omnibus_name = "Kruskal-Wallis"
        self.logger(f">>> OMNIBUS {omnibus_name}: p={p_omnibus:.4e}")
        if p_omnibus > 0.05:
            self.logger("!!! WARNING: No significant variance across groups. Proceed with caution.")

        # 4. Aggregation with Confidence Intervals (Replacing Quality Score)
        def get_ci(data):
            # 95% CI using T-distribution (Standard for N=10)
            if len(data) < 2: return 0
            se = stats.sem(data)
            return se * stats.t.ppf((1 + 0.95) / 2., len(data) - 1)
        stats_table = clean_df.groupby(group_by)[metric].agg(['mean', 'std', 'count', 'median']).copy()
        stats_table['95%_CI_Error'] = clean_df.groupby(group_by)[metric].apply(get_ci)
        
        # Ranking: Simply by mean (or median for non-parametric)
        rank_col = 'mean' if use_parametric else 'median'
        stats_table = stats_table.sort_values(rank_col, ascending=is_error_metric)

        # 5. Best-vs-Rest Analysis with Holm Correction
        if len(stats_table) > 1:
            best_idx = stats_table.index[0]
            d_best = clean_df[clean_df[group_by] == best_idx][metric]
            
            raw_p_values, effect_sizes = [], []
            for other_idx in stats_table.index[1:]:
                d_other_df = clean_df[clean_df[group_by] == other_idx]#.sort_values(by='run')
                d_other = d_other_df[metric].values
                if len(d_best) == len(d_other):
                    if use_parametric:
                        # stats.ttest_rel is the PAIRED T-test
                        _, p = stats.ttest_rel(d_best, d_other)
                    else:
                        # stats.wilcoxon is the PAIRED non-parametric test
                        _, p = stats.wilcoxon(d_best, d_other, alternative='two-sided')
                else:
                    # Fallback to independent if N doesn't match (unlikely in your setup)
                    self.logger(f"!!! Warning: Unmatched samples for {other_idx}. Using independent test.")
                    _, p = stats.mannwhitneyu(d_best, d_other) if not use_parametric else stats.ttest_ind(d_best, d_other)
                g = self.calculate_hedges_g(d_best, d_other)
                effect_sizes.append(round(g, 3))
                raw_p_values.append(p)
                # Apply Multiple Comparison Correction
            reject, p_adjusted, _, _ = multipletests(raw_p_values, alpha=0.05, method='fdr_bh')                       
            stats_table['Hedges_g_vs_Best'] = [0.0] + effect_sizes # Best vs itself is 0
            stats_table['p-value (Adjusted)'] = [1.0] + p_adjusted.tolist()
            stats_table['Is_Significant'] = ["N/A (BEST)"] + ["YES" if r else "NO" for r in reject]
            def interpret_g(g):
                g = abs(g)
                if g < 0.2: return "Negligible"
                if g < 0.5: return "Small"
                if g < 0.8: return "Medium"
                return "Large"

            stats_table['Effect_Magnitude'] = ["N/A"] + [interpret_g(g) for g in effect_sizes]

        # 6. Saving & Visualization
        if self.writer:
            sheet_name = f"BestVsRest_{metric[:20]}"
            stats_table.to_excel(self.writer, sheet_name=sheet_name)
            self.run_pairwise_analysis(clean_df, stats_table.index, metric, group_by, use_parametric)
         
        self._visualize_results(clean_df, stats_table, group_by, metric, use_parametric)
    def run_pairwise_analysis(self, data_df, ordered_indices, metric, group_by, use_parametric):
        """Generates a p-value matrix using Holm-Bonferroni adjusted values."""
        self.logger(f"\n>>> Generating Pairwise Matrix (Global Strategy: {'Parametric' if use_parametric else 'Non-Parametric'})")
        
        limit = 50
        indices = ordered_indices[:limit]
        matrix = pd.DataFrame(index=indices, columns=indices)
        comparison_records = []        
        for i_idx, i in enumerate(indices):
            for j_idx, j in enumerate(indices):
               if i_idx < j_idx:
                    # Align by 'run'
                    d1 = data_df[data_df[group_by] == i].sort_values('run')[metric].values
                    d2 = data_df[data_df[group_by] == j].sort_values('run')[metric].values
                    
                    if len(d1) == len(d2):
                        if use_parametric:
                            _, p = stats.ttest_rel(d1, d2)
                        else:
                            _, p = stats.wilcoxon(d1, d2, alternative='two-sided')
                    else:
                        _, p = stats.mannwhitneyu(d1, d2) if not use_parametric else stats.ttest_ind(d1, d2)
                    
                    g = self.calculate_hedges_g(d1, d2)
                    comparison_records.append({'i': i, 'j': j, 'p_raw': p, 'effect_size': g})
        # 2. Apply Multiple Comparison Correction (Holm-Bonferroni)
        if comparison_records:
            raw_p_list = [comp['p_raw'] for comp in comparison_records]
            reject, p_adjusted, _, _ = multipletests(raw_p_list, alpha=0.05, method='fdr_bh')
            
            # 3. Map adjusted p-values back into the matrix
            for idx, comp in enumerate(comparison_records):
                adj_val = p_adjusted[idx]
                matrix.loc[comp['i'], comp['j']] = adj_val
                matrix.loc[comp['j'], comp['i']] = adj_val  # Ensure symmetry
        
        # Fill diagonal
        np.fill_diagonal(matrix.values, 1.0)
        sheet_name = f"Pairwise_{metric[:20]}"
        matrix.to_excel(self.writer, sheet_name=sheet_name)
        self._apply_excel_formatting(sheet_name)
    def _visualize_results(self, clean_df, stats_table, group_by, metric, use_parametric):
        # 5. Visualization (Top/Mid/Bottom Subset)
        plot_indices = self.get_smart_subset(stats_table)
        viz_df = clean_df[clean_df[group_by].isin(plot_indices)].copy()
        viz_df[group_by] = pd.Categorical(viz_df[group_by], categories=plot_indices, ordered=True)
        show_mean = True if use_parametric else False
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        sns.boxplot(x=group_by, y=metric, data=viz_df, palette=self.my_colors, ax=axes[0],showmeans=show_mean, showfliers=True, meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black"} if show_mean else None)
        sns.swarmplot(x=group_by, y=metric, data=viz_df, color=self.line_color, alpha=0.5, ax=axes[0])
        axes[0].set_title(f"Comparison: {metric} (Subset)")
        axes[0].tick_params(axis='x', rotation=45)
        
        qqplot(clean_df[clean_df[group_by] == stats_table.index[0]][metric], line='s', ax=axes[1])
        axes[1].set_title(f"Q-Q Plot: {stats_table.index[0]} (Best)")
        
        plt.tight_layout()
        if self.save_enabled:
            plt.savefig(os.path.join(self.out_dir, f"plot_{group_by}_{metric.replace(' ','_')}.png"))
        plt.show()

        # 6. Histograms
        n_plots = min(len(stats_table), 4)
        fig_h, axes_h = plt.subplots(2, 2, figsize=(12, 10))
        axes_h_flat = axes_h.flatten()
        for i in range(n_plots):
            idx = stats_table.index[i]
            data = clean_df[clean_df[group_by] == idx][metric]
            sns.histplot(data, kde=True, ax=axes_h_flat[i], color="#4DB6AC")
            axes_h_flat[i].set_title(f"Rank {i+1}: {idx}")
        plt.suptitle(f"Histograms: {metric}")
        plt.tight_layout()
        if self.save_enabled:
            plt.savefig(os.path.join(self.out_dir, f"hist_grid_{group_by}_{metric.replace(' ','_')}.png"))
        plt.show()
    def calculate_hedges_g(self, group1, group2):
        """Calculates Hedges' g for small sample sizes (N=10)."""
        n1, n2 = len(group1), len(group2)
        if n1 < 2 or n2 < 2: return 0
        
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled_std == 0: return 0
        
        d = (mean1 - mean2) / pooled_std
        # Hedges' g correction factor for small samples
        correction = 1 - (3 / (4 * (n1 + n2) - 9))
        return d * correction
    
    def _apply_excel_formatting(self, sheet_name):
        ws = self.writer.sheets[sheet_name]
        green = PatternFill(start_color='C6EFCE', end_color='C6EFCE', fill_type='solid')
        red = PatternFill(start_color='FFC7CE', end_color='FFC7CE', fill_type='solid')
        cell_range = f"B2:{utils.get_column_letter(ws.max_column)}{ws.max_row}"
        ws.conditional_formatting.add(cell_range, CellIsRule(operator='lessThan', formula=['0,05'], fill=green))
        ws.conditional_formatting.add(cell_range, CellIsRule(operator='greaterThanOrEqual', formula=['0.05'], fill=red))
    def _perform_art_interaction(self, df, factor1, factor2, metric):
        self.logger(">>> Running Aligned Rank Transform (ART) for Interaction Analysis...")
        grand_mean = df[metric].mean()
        mean_a = df.groupby(factor1)[metric].transform('mean')
        mean_b = df.groupby(factor2)[metric].transform('mean')
        
        df['art_aligned'] = df[metric] - (mean_a - grand_mean) - (mean_b - grand_mean)
        df['art_rank'] = stats.rankdata(df['art_aligned'])
        
        formula = f"art_rank ~ C({factor1}) * C({factor2})"
        model = ols(formula, data=df).fit()
        table = sm.stats.anova_lm(model, typ=2)
        
        inter_key = f"C({factor1}):C({factor2})"
        p_inter = table.loc[inter_key, "PR(>F)"]
        return table, p_inter
    def run_interaction_study(self, factor_x, factor_hue, metric):
        header = f"\n{'='*75}\nINTERACTION STUDY: {metric} ({factor_x} x {factor_hue})\n{'='*75}"
        self.logger(header)
        self.df[metric] = pd.to_numeric(self.df[metric], errors='coerce')
        clean_df = self.df.dropna(subset=[metric]).copy()
        
        # 1. ALWAYS run the statistical tests on the FULL dataset for complete accuracy
        pivot_mean = clean_df.pivot_table(index=factor_x, columns=factor_hue, values=metric, aggfunc='mean')
        
        groups = [g[metric].values for n, g in clean_df.groupby([factor_x, factor_hue])]
        norm_passes = [self.check_normality_full(g, "group") for g in groups if len(g) >= 3]
        pct_normal = sum(norm_passes) / len(norm_passes) if norm_passes else 0
        use_parametric = pct_normal > 0.95 

        if use_parametric:
            formula = f"Q('{metric}') ~ C({factor_x}) * C({factor_hue})"
            model = ols(formula, data=clean_df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            p_inter = anova_table.loc[f"C({factor_x}):C({factor_hue})", "PR(>F)"]
        else:
            anova_table, p_inter = self._perform_art_interaction(clean_df, factor_x, factor_hue, metric)

        if self.writer:
            sheet_name = f"I_{factor_x[:10]}_{factor_hue[:10]}_{metric}".replace(" ", "_")
            anova_table.reset_index().to_excel(self.writer, sheet_name=sheet_name, index=False)

        # --- SMART PLOT FILTERING BLOCK ---
        # Create a separate DataFrame purely for a clean visualization
        plot_df = clean_df.copy()
        max_options_to_plot = 15
        
        # Check if the primary variable (e.g., 'map') has too many variations
        if plot_df[factor_x].nunique() > max_options_to_plot:
            # Find the top performing categories based on the mean metric score
            top_performers = plot_df.groupby(factor_x)[metric].mean().sort_values(ascending=False).index[:max_options_to_plot].tolist()
            # Filter the plot data to show only these top options
            plot_df = plot_df[plot_df[factor_x].isin(top_performers)]
            self.logger(f"[Visual Filter] Limit reached for '{factor_x}'. Plotting top {max_options_to_plot} options to ensure readability.")
            
        # Check if the secondary variable has too many variations
        if plot_df[factor_hue].nunique() > max_options_to_plot:
            top_hues = plot_df.groupby(factor_hue)[metric].mean().sort_values(ascending=False).index[:max_options_to_plot].tolist()
            plot_df = plot_df[plot_df[factor_hue].isin(top_hues)]
            self.logger(f"[Visual Filter] Limit reached for '{factor_hue}'. Plotting top {max_options_to_plot} options to ensure readability.")
        # ----------------------------------

        # --- RENDER CLEAN VISUALIZATION ---
        plt.figure(figsize=(12, 6))
        
        # Use the filtered plot_df instead of clean_df
        sns.pointplot(x=factor_x, y=metric, hue=factor_hue, data=plot_df, capsize=.05, dodge=0.2)
        
        plt.title(f"Interaction: {metric} (Top {max_options_to_plot} Overview)\nSignificance: p = {p_inter:.4e}")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.xticks(rotation=15)
        plt.tight_layout()
        
        if self.save_enabled:
            plt.savefig(os.path.join(self.out_dir, f"int_{factor_x}_{factor_hue}_{metric[:10].replace(' ','_')}.png"))
        plt.show()
    def _create_config_id(self):
        """Creates short IDs and saves a mapping table."""
        group_keys = [
            "features", "targets", "window_size", "horizon", "predict", "norm", 
            "reconstruct_train", "reconstruct_val", "model", "heads_config", 
            "head_number", "encoding", "ansatz", "entangle", "reps", "map", 
            "reorder", "optimizer", "maxiter", "iterations", "tolerance", 
            "batch_size", "learning_rate", "perturbation", "initialization", "output_feat"
        ]
        existing_keys = [k for k in group_keys if k in self.df.columns]
        
        # 1. Generate the long string
        self.df['Full_Config_Long'] = self.df[existing_keys].astype(str).agg(' | '.join, axis=1)
        
        # 2. Create Unique Mapping
        unique_configs = self.df['Full_Config_Long'].unique()
        mapping_dict = {long: f"Config {i+1}" for i, long in enumerate(unique_configs)}
        
        # 3. Apply short ID
        self.df['Config_ID'] = self.df['Full_Config_Long'].map(mapping_dict)
        
        # 4. Save the Mapping Table
        mapping_df = self.df[existing_keys + ['Config_ID']].drop_duplicates().sort_values('Config_ID')
        if self.save_enabled:
            mapping_df.to_excel(os.path.join(self.out_dir, "config_mapping_key.xlsx"), index=False)
            self.logger(f">>> Mapping table saved: config_mapping_key.xlsx")
            
        return 'Config_ID'

    def get_smart_subset(self, stats_table, n_total=30):
        """Returns a list of indices for Top, Middle, and Bottom performers."""
        if len(stats_table) <= n_total:
            return stats_table.index.tolist()
        
        top_n = 15
        bot_n = 10
        mid_n = n_total - top_n - bot_n
        
        top_idx = stats_table.index[:top_n].tolist()
        bot_idx = stats_table.index[-bot_n:].tolist()
        
        # Take evenly spaced samples from the middle
        remaining = stats_table.index[top_n:-bot_n]
        mid_indices = np.linspace(0, len(remaining)-1, mid_n, dtype=int)
        mid_idx = remaining[mid_indices].tolist()
        
        return top_idx + mid_idx + bot_idx
    def close(self):
        """Safely closes the log and saves the Excel report."""
        if self.log_file: 
            self.log_file.close()
            
        if self.writer:
            try:
                # This performs the actual .save() and avoids the IndexError 
                # because we created the 'Summary_Overview' sheet in __init__
                self.writer.close()
                print(f"Report saved successfully in {self.out_dir}")
            except Exception as e:
                print(f"Warning: Could not save Excel report. {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    parser.add_argument("--studies", nargs='+')
    parser.add_argument("--metric", nargs='+')
    parser.add_argument("--interactions", nargs='+')
    parser.add_argument("--interaction_metrics", nargs='+')
    parser.add_argument("--save", action="store_true")
    
    args = parser.parse_args()
    analyzer = QNNAnalyzer(args.file, save_enabled=args.save)

    if not args.studies and not args.interactions:
        config_column = analyzer._create_config_id()
        default_metric = args.metric if args.metric else ["Val Global Open R2"]
        analyzer.logger(f"\n[AUTOMATIC MODE] Running global studies across configuration sets.")
        for m in default_metric:
            analyzer.run_study(config_column, m)
    else:
        # 2. Main Factor Independent Study Mode
        if args.studies:
            metrics = args.metric if args.metric else ["Val Global Open R2"]
            # If user provides one metric but multiple studies, extend to evaluate them all
            if len(metrics) == 1 and len(args.studies) > 1:
                metrics = metrics * len(args.studies)
            for s, m in zip(args.studies, metrics):
                analyzer.run_study(s, m)

        # 3. Fully Crossed Interaction Study Mode (Evaluates ALL targets across ALL pairs)
        if args.interactions:
            i_metrics = args.interaction_metrics if args.interaction_metrics else ["Val Global Open R2"]
            for pair in args.interactions:
                x, hue = pair.split(':')
                for m in i_metrics:
                    analyzer.run_interaction_study(x, hue, m)

    analyzer.close()