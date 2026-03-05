import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import warnings
import os
import argparse
import re

# 1. STYLE & FONT CONFIGURATION
warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Cambria", "Times New Roman"],
    "axes.titleweight": "bold",
    "font.size": 11
})

class QNNAnalyzer:
    def __init__(self, file_path, sheet_name=0, fail_threshold=-1.0, save_enabled=False):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.save_enabled = save_enabled
        self.fail_threshold = fail_threshold
        self.df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
        
        self.my_colors = [
            "#B2EBF2", "#4DB6AC", "#00695C", # <--- YOUR PRIORITIZED TEALS
            "#E57373", "#9575CD", "#64B5F6", # Muted Red, Muted Purple, Sky Blue
            "#81C784", "#FFD54F", "#FF8A65", # Sage Green, Amber, Terracotta
            "#A1887F", "#4DD0E1", "#F06292", # Taupe, Bright Cyan, Soft Rose
            "#AED581", "#D4E157", "#FFB74D", # Olive, Citron, Tangerine
            "#BA68C8", "#4FC3F7", "#90A4AE", # Orchid, Light Blue, Blue Grey
            "#CE93D8", "#80CBC4"             # Plum, Seafoam
        ]
        self.line_color = "#002B2E" 
        
        if self.save_enabled:
            base_dir = os.path.dirname(os.path.abspath(file_path))
            self.out_dir = os.path.join(base_dir, "statistical_analysis")
            os.makedirs(self.out_dir, exist_ok=True)
            self.log_file = open(os.path.join(self.out_dir, "study_results_log.txt"), "a", encoding="utf-8")
            self.writer = pd.ExcelWriter(os.path.join(self.out_dir, "analysis_summary_report.xlsx"), engine='openpyxl')
        else:
            self.log_file, self.writer = None, None

        self._preprocess()

    def _preprocess(self):
        # Clean the map column
        if 'map' in self.df.columns:
            self.df['map'] = self.df['map'].astype(str).str.replace(" ", "")
            if 'Combination' not in self.df.columns:
                self.df['Combination'] = self.df['map']
        
        # 2. Fallback for Combination identification
        if 'Combination' not in self.df.columns:
            if 'save_dir' in self.df.columns:
                self.df['Combination'] = self.df['save_dir']
            else:
                self.df['Combination'] = "Group_1"
    def logger(self, text):
        print(text)
        if self.log_file: self.log_file.write(text + "\n")
    def _get_sheet_name(self, prefix, factors, metric):
        """Excel sheet name helper (max 31 chars)."""
        name = f"{prefix}_{factors}_{metric}"[:31]
        return "".join(x for x in name if x.isalnum() or x == "_")
    def check_normality(self, data):
        if len(data) < 3 or np.ptp(data) == 0: return False
        _, p = stats.shapiro(data)
        return p > 0.05

    def run_study(self, group_by, metric):
        header = f"\n{'='*75}\nSTUDY: {metric} by {group_by}\n{'='*75}"
        self.logger(header)
        if metric not in self.df.columns:
            self.logger(f"Error: Metric '{metric}' not found in file.")
            return
        self.df[metric] = pd.to_numeric(self.df[metric], errors='coerce')
        if "R2" in metric:
            clean_df = self.df[self.df[metric] > self.fail_threshold].dropna(subset=[metric]).copy()
            sort_ascending = False  # Higher R2 is better
        else:
            # For MSE, a 'fail' is a massive error. Adjust threshold if needed.
            clean_df = self.df.dropna(subset=[metric]).copy()
            # Only keep values that are somewhat realistic (adjust if your MSE is higher than 1000)
            clean_df = clean_df[clean_df[metric] < 1000.0]
            sort_ascending = True   #
        
        # 1. Advanced Statistics (including SEM for sufficiency check)
        stats_table = clean_df.groupby(group_by)[metric].agg(['count', 'mean', 'median', 'std', 'max'])
        stats_table['SEM'] = stats_table['std'] / np.sqrt(stats_table['count'])
        stats_table['Precision_Err_%'] = np.where(
            stats_table['mean'] != 0, 
            (1.96 * stats_table['SEM']) / stats_table['mean'].abs() * 100, 
            0
        )
        stats_table = stats_table.sort_values('median', ascending=sort_ascending)
        
        self.logger("\nSummary Statistics & Sufficiency (N=10 check):\n" + stats_table.round(4).to_string())
        # 2. SAVE TO EXCEL (Fixes the IndexError)
        if self.writer:
            sheet_id = self._get_sheet_name("Gen", str(group_by), str(metric))
            stats_table.to_excel(self.writer, sheet_name=sheet_id)
        # 2. Automated Test Selection (Parametric vs Non-Parametric)
        if len(stats_table) > 1:
            best_group = stats_table.index[0]
            d_best = clean_df[clean_df[group_by] == best_group][metric]
            is_best_normal = self.check_normality(d_best)
            
            self.logger(f"\n--- Significance Tests vs. Best ({best_group}) ---")
            self.logger(f"Best Group Normality: {'Normal' if is_best_normal else 'NOT Normal'}")
            
            for other in stats_table.index[1:]:
                d_other = clean_df[clean_df[group_by] == other][metric]
                if len(d_other) < 2: continue
                
                # Selection logic
                if is_best_normal and self.check_normality(d_other):
                    _, p_val = stats.ttest_ind(d_best, d_other, equal_var=False)
                    test_type = "(T-Test)"
                else:
                    _, p_val = stats.mannwhitneyu(d_best, d_other)
                    test_type = "(Mann-U)"
                
                sig = "SIGNIFICANT" if p_val < 0.05 else "not sig."
                self.logger(f"{str(other):<20} | p: {p_val:.4f} {test_type:<8} | {sig}")

        # 3. Visualization
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=group_by, y=metric, data=clean_df, order=stats_table.index, palette=self.my_colors)
        sns.stripplot(x=group_by, y=metric, data=clean_df, order=stats_table.index, color=self.line_color, alpha=0.4)
        plt.title(f"1-Head Analysis: {metric} by {group_by}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        if self.save_enabled:
            plt.savefig(os.path.join(self.out_dir, f"plot_{group_by}_{metric.replace(' ','_')}.png"))
        plt.show()

    def close(self):
        if self.log_file: self.log_file.close()
        if self.writer:
            try:
                self.writer.close()
            except IndexError:
                # This catches the case where run_study was never called or failed
                pass
# Main remains largely same as yours, connecting arguments to the class
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QNN Physics-Informed Statistical Tool")
    parser.add_argument("--file", required=True)
    parser.add_argument("--studies", nargs='+', help="Grouping factor: usually 'Combination' or 'map'")
    parser.add_argument("--study_metrics", nargs='+', help="e.g. 'global closed R2'")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--threshold", type=float, default=-10.0) # Lowered to capture QML noise
    
    args = parser.parse_args()
    analyzer = QNNAnalyzer(args.file, fail_threshold=args.threshold, save_enabled=args.save)

    if args.studies:
        count = len(args.studies)
        metrics = args.study_metrics if args.study_metrics else ["global open R2"]
        if len(metrics) == 1: metrics = metrics * count
        
        for factor, metric in zip(args.studies, metrics):
            analyzer.run_study(factor, metric)
    
    analyzer.close()