import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import lilliefors as lilliefors_test
import warnings
import os
import argparse

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
            self.out_dir = os.path.join(base_dir, "statistical_analysis")
            os.makedirs(self.out_dir, exist_ok=True)
            self.log_file = open(os.path.join(self.out_dir, "results_log.txt"), "a", encoding="utf-8")
            
            # FIX: Initialize the writer and ensure at least one sheet exists
            report_path = os.path.join(self.out_dir, "analysis_report.xlsx")
            self.writer = pd.ExcelWriter(report_path, engine='openpyxl')
            
            # Create a blank sheet to prevent the IndexError
            temp_df = pd.DataFrame({"Status": ["Analysis Started"]})
            temp_df.to_excel(self.writer, sheet_name="Summary_Overview")
        else:
            self.log_file, self.writer = None, None

        self._preprocess()

    def _preprocess(self):
        if 'map' in self.df.columns:
            self.df['map'] = self.df['map'].astype(str).str.replace(" ", "")
        categorical_cols = ['reps', 'encoding', 'ansatz', 'entangle', 'head_number']
        for col in categorical_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.replace(".0", "", regex=False).str.strip()

    def logger(self, text):
        print(text)
        if self.log_file: self.log_file.write(text + "\n")
    def check_normality_full(self, data, label):
        """Performs the 3-test consensus suite (Shapiro, D'Agostino, Anderson)."""
        self.logger(f"\n>>> Normality Check for {label} (N={len(data)})")
        if len(data) < 3:
            self.logger("   STATUS: NOT NORMAL (Insufficient data points)")
            return False
        
        alpha = 0.05
        results = []

        # 1. Shapiro-Wilk
        stat_sw, p_sw = stats.shapiro(data)
        is_sw_normal = p_sw > alpha
        results.append(is_sw_normal)
        self.logger(f"   Shapiro-Wilk: p={p_sw:.4f} -> {'NORMAL' if is_sw_normal else 'NOT NORMAL'}")

        # 2. Lillieford (Requires N >= 3)
        if len(data) >= 4:
            stat_da, p_da = lilliefors_test(data)    
            is_da_normal = p_da > alpha
            results.append(is_da_normal)
            self.logger(f"   Lillieford: p={p_da:.4f} -> {'NORMAL' if is_da_normal else 'NOT NORMAL'}")

        # 3. Anderson-Darling (5% significance level)
        if len(data) >= 5:
            ad_res = stats.anderson(data, dist='norm')
            is_ad_normal = ad_res.statistic < ad_res.critical_values[2]
            results.append(is_ad_normal)
            self.logger(f"   Anderson-Darling: Stat={ad_res.statistic:.3f} (CV 5%={ad_res.critical_values[2]}) -> {'NORMAL' if is_ad_normal else 'NOT NORMAL'}")
        # Consensus: 2 out of 3 pass
        num_passed = sum(results)
        num_run = len(results)
        is_gaussian = num_passed > (num_run / 2)
        final_status = "NORMAL (Gaussian)" if is_gaussian else "NOT NORMAL (Non-Gaussian)"
        self.logger(f"   FINAL STATUS: {final_status} ({num_passed}/{num_run} passed)")
        return is_gaussian

    def run_study(self, group_by, metric):
        header = f"\n{'='*75}\nSTUDY: {metric} by {group_by}\n{'='*75}"
        self.logger(header)
        
        # Ensure numeric and NO FILTERING
        self.df[metric] = pd.to_numeric(self.df[metric], errors='coerce')
        clean_df = self.df.dropna(subset=[metric]).copy()
        
        is_error_metric = any(x in metric.upper() for x in ["MSE", "LOSS", "MAE"])
        
        # Sort so index[0] is always the mathematically "Best"
        stats_table = clean_df.groupby(group_by)[metric].agg(['mean', 'std', 'count', 'max']).copy()
        # Calculate CV as a percentage
        stats_table['CV%'] = (stats_table['std'] / stats_table['mean']) * 100
        stats_table = stats_table.sort_values('mean', ascending=is_error_metric)
        self.logger("\nSummary Statistics:\n" + stats_table.to_string())
        if self.writer:
            sheet_name = f"Gen_{group_by}_{metric}"[:30].replace(" ", "_")
            stats_table.to_excel(self.writer, sheet_name=sheet_name)
        # Normality
        norm_results = {idx: self.check_normality_full(clean_df[clean_df[group_by]==idx][metric], idx) for idx in stats_table.index}

        # Significance
        if len(stats_table) > 1:
            best_idx = stats_table.index[0]
            d_best = clean_df[clean_df[group_by] == best_idx][metric]
            self.logger(f"\n--- PHASE 2: SIGNIFICANCE vs Best ({best_idx}) ---")
            for other_idx in stats_table.index[1:]:
                d_other = clean_df[clean_df[group_by] == other_idx][metric]
                if norm_results[best_idx] and norm_results[other_idx]:
                    _, p = stats.ttest_ind(d_best, d_other, equal_var=False)
                    test = "T-Test (Para)"
                else:
                    _, p = stats.mannwhitneyu(d_best, d_other)
                    test = "Mann-U (Non-Para)"
                self.logger(f"{str(other_idx):<20} | p: {p:.4f} | {test:<15} | {'SIGNIFICANT' if p < 0.05 else 'not sig.'}")

        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        sns.boxplot(x=group_by, y=metric, data=clean_df, palette=self.my_colors, ax=axes[0], showfliers=True)
        sns.swarmplot(x=group_by, y=metric, data=clean_df, color=self.line_color, alpha=0.5, ax=axes[0])
        axes[0].set_title(f"Comparison: {metric}")
        
        qqplot(clean_df[clean_df[group_by] == stats_table.index[0]][metric], line='s', ax=axes[1])
        axes[1].set_title(f"Q-Q Plot: {stats_table.index[0]}")
        plt.tight_layout()
        if self.save_enabled:
            plt.savefig(os.path.join(self.out_dir, f"plot_{group_by}_{metric.replace(' ','_')}.png"))
        plt.show()

        # 2x2 Histogram Grid
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

    def run_interaction_study(self, factor_x, factor_hue, metric):
        header = f"\n{'='*75}\nINTERACTION STUDY: {metric} ({factor_x} x {factor_hue})\n{'='*75}"
        self.logger(header)
        self.df[metric] = pd.to_numeric(self.df[metric], errors='coerce')
        clean_df = self.df.dropna(subset=[metric]).copy()
        
        pivot_mean = clean_df.pivot_table(index=factor_x, columns=factor_hue, values=metric, aggfunc='mean')
        self.logger("\nInteraction Table (Means):\n" + pivot_mean.round(5).to_string())
        
        self.logger("\n--- PHASE 1: NORMALITY VERIFICATION (Interaction Pairs) ---")
        for x_val in pivot_mean.index:
            for hue_val in pivot_mean.columns:
                pair_data = clean_df[(clean_df[factor_x] == x_val) & (clean_df[factor_hue] == hue_val)][metric]
                self.check_normality_full(pair_data, f"{x_val} x {hue_val}")

        plt.figure(figsize=(14, 7))
        sns.boxplot(x=factor_x, y=metric, hue=factor_hue, data=clean_df, palette=self.my_colors, showfliers=True)
        plt.title(f"Interaction: {metric}")
        plt.tight_layout()
        if self.save_enabled:
            plt.savefig(os.path.join(self.out_dir, f"int_{factor_x}_{factor_hue}_{metric.replace(' ','_')}.png"))
        plt.show()
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
    parser.add_argument("--study_metrics", nargs='+')
    parser.add_argument("--interactions", nargs='+')
    parser.add_argument("--interaction_metrics", nargs='+')
    parser.add_argument("--save", action="store_true")
    
    args = parser.parse_args()
    analyzer = QNNAnalyzer(args.file, save_enabled=args.save)

    if args.studies:
        metrics = args.study_metrics if args.study_metrics else ["global open R2"]
        if len(metrics) == 1: metrics = metrics * len(args.studies)
        for s, m in zip(args.studies, metrics):
            analyzer.run_study(s, m)

    if args.interactions:
        i_metrics = args.interaction_metrics if args.interaction_metrics else ["global open R2"]
        if len(i_metrics) == 1: i_metrics = i_metrics * len(args.interactions)
        for pair, m in zip(args.interactions, i_metrics):
            x, hue = pair.split(':')
            analyzer.run_interaction_study(x, hue, m)

    analyzer.close()