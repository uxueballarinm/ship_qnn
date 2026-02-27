import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import warnings
import os
import argparse

# 1. STYLE & FONT CONFIGURATION (Cambria)
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
        
        # High-Contrast Teal spectrum
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
            try:
                self.writer = pd.ExcelWriter(os.path.join(self.out_dir, "analysis_summary_report.xlsx"), mode = 'a', engine='openpyxl')
            except FileNotFoundError:
                self.writer = pd.ExcelWriter(os.path.join(self.out_dir, "analysis_summary_report.xlsx"), engine='openpyxl')
        else:
            self.log_file = None
            self.writer = None

        self._preprocess()

    def _preprocess(self):
        # 1. Clean the map column
        if 'map' in self.df.columns:
            self.df['map'] = self.df['map'].astype(str).str.replace(" ", "")
            
        # 2. Fix the "Duplicate Categories" issue by converting mixed types to pure strings
        categorical_cols = ['reps', 'encoding', 'ansatz', 'entangle', 'head_number']
        for col in categorical_cols:
            if col in self.df.columns:
                # Convert to string, remove decimal points if they were saved as 3.0, and strip spaces
                self.df[col] = self.df[col].astype(str).str.replace(".0", "", regex=False).str.strip()

    def logger(self, text):
        print(text)
        if self.log_file:
            self.log_file.write(text + "\n")
    def _get_sheet_name(self, prefix, factors, metric):
        """
        Smart abbreviator to keep Excel sheet names under 31 chars 
        while preserving the full meaning of the metric.
        """
        # Dictionary to abbreviate long words
        subs = {
            "global": "gl", "open": "op", "closed": "cl",
            "Surge": "Srg", "Sway": "Swy", "Velocity": "Vel",
            "Yaw": "Yw", "Rate": "Rt", "Angle": "Ang",
            " ": "" # Strip spaces
        }
        short_metric = metric
        for k, v in subs.items():
            short_metric = short_metric.replace(k, v)

        # Build raw name: e.g., Gen_map_glopR2
        raw_name = f"{prefix}_{factors}_{short_metric}"
        clean_name = "".join(x for x in raw_name if x.isalnum() or x in "_")

        # Handle the Excel strict 31-character limit
        if len(clean_name) > 31:
            allowed_factor_len = 31 - len(prefix) - len(short_metric) - 2
            if allowed_factor_len > 1:
                clean_name = f"{prefix}_{factors[:allowed_factor_len]}_{short_metric}"
            else:
                clean_name = clean_name[:31]

        return clean_name

    def _clean_filename(self, text):
        """Replaces spaces with underscores for clean file naming."""
        return text.replace(" ", "_")
    def run_study(self, group_by, metric):
        header = f"\n{'='*75}\nGENERAL STUDY: {metric} by {group_by}\n{'='*75}"
        self.logger(header)
        self.df[metric] = pd.to_numeric(self.df[metric], errors='coerce')
        clean_df = self.df[self.df[metric] > self.fail_threshold].copy()
        stats_table = clean_df.groupby(group_by)[metric].agg(['mean', 'std', 'count', 'max']).sort_values('mean', ascending=False)
        self.logger("\nSummary Statistics Table:\n" + stats_table.round(5).to_string())

        if self.writer:
            sheet_id = self._get_sheet_name("Gen", group_by, metric)
            stats_table.to_excel(self.writer, sheet_name=sheet_id)

        if len(stats_table) > 1:
            best_group = stats_table.index[0]
            self.logger(f"\n--- Significance Tests (vs. Best: {best_group}) ---")
            for other in stats_table.index[1:]:
                d_best, d_other = clean_df[clean_df[group_by] == best_group][metric], clean_df[clean_df[group_by] == other][metric]
                if len(d_best) > 1 and len(d_other) > 1:
                    _, p_val = stats.ttest_ind(d_best, d_other, equal_var=False)
                    sig = "SIGNIFICANT" if p_val < 0.05 else "not sig."
                    self.logger(f"{str(best_group):<15} vs {str(other):<20} | p: {p_val:.4f} | {sig}")

        plt.figure(figsize=(10, 6))
        sns.boxplot(x=group_by, y=metric, data=clean_df, hue=group_by, palette=self.my_colors, showfliers=False, legend=False,
                    boxprops=dict(edgecolor=self.line_color), medianprops=dict(color=self.line_color),
                    whiskerprops=dict(color=self.line_color), capprops=dict(color=self.line_color))
        sns.swarmplot(x=group_by, y=metric, data=clean_df, color=self.line_color, alpha=0.5)
        plt.title(f"General Study: {metric} per {group_by}", fontname='Cambria')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', color=self.line_color, linestyle=':', alpha=0.15)
        plt.tight_layout()
        if self.save_enabled:
            fname = f"plot_{group_by}_{self._clean_filename(metric)}.png"
            plt.savefig(os.path.join(self.out_dir, fname), dpi=300)
        plt.show()

    def run_interaction_study(self, factor_x, factor_hue, metric):
        header = f"\n{'='*75}\nINTERACTION STUDY: {metric} ({factor_x} x {factor_hue})\n{'='*75}"
        self.logger(header)

        clean_df = self.df[self.df[metric] > self.fail_threshold].copy()
        pivot_mean = clean_df.pivot_table(index=factor_x, columns=factor_hue, values=metric, aggfunc='mean')
        self.logger("\nMean Interaction Table:\n" + pivot_mean.round(5).to_string())
        
        if self.writer:
            sheet_id = self._get_sheet_name("Int", f"{factor_x}_{factor_hue}", metric)
            pivot_mean.to_excel(self.writer, sheet_name=sheet_id)

        plt.figure(figsize=(14, 7))
        ax = sns.boxplot(x=factor_x, y=metric, hue=factor_hue, data=clean_df, palette=self.my_colors, showfliers=False,
                         boxprops=dict(edgecolor=self.line_color), medianprops=dict(color=self.line_color),
                         whiskerprops=dict(color=self.line_color), capprops=dict(color=self.line_color))
        sns.swarmplot(x=factor_x, y=metric, hue=factor_hue, data=clean_df, color=self.line_color, dodge=True, alpha=0.4, legend=False)
        
        xticks = ax.get_xticks()
        for i in range(len(xticks) - 1):
            plt.axvline(x=(xticks[i] + xticks[i+1]) / 2, color=self.line_color, linestyle='--', alpha=0.2)
        plt.title(f"Interaction: {metric} ({factor_x} by {factor_hue})", fontname='Cambria')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title=factor_hue, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        if self.save_enabled:
            fname = f"int_{factor_x}_{factor_hue}_{self._clean_filename(metric)}.png"
            plt.savefig(os.path.join(self.out_dir, fname), dpi=300)
        plt.show()

    def close(self):
        if self.log_file: self.log_file.close()
        if self.writer: self.writer.close()

def parse_metrics(metrics_list, count, label):
    """If 1 metric is provided, duplicate it. If N are provided, keep them. Else Error."""
    if metrics_list is None: return ["global open R2"] * count
    if len(metrics_list) == 1: return metrics_list * count
    if len(metrics_list) == count: return metrics_list
    raise ValueError(f"Number of metrics for {label} must be 1 or match the number of studies ({count}).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QNN Research Analysis Terminal Tool")
    parser.add_argument("--file", required=True, help="Path to .xlsx file")
    parser.add_argument("--studies", nargs='+', help="Grouping factors (e.g. encoding map)")
    parser.add_argument("--study_metrics", nargs='+', help="Metrics for studies (1 for all or 1 per study)")
    parser.add_argument("--interactions", nargs='+', help="Interaction pairs (e.g. map:reps)")
    parser.add_argument("--interaction_metrics", nargs='+', help="Metrics for interactions (1 for all or 1 per interaction)")
    parser.add_argument("--save", action="store_true", help="Save files in statistical_analysis folder")
    parser.add_argument("--threshold", type=float, default=-1.0)
    
    args = parser.parse_args()
    analyzer = QNNAnalyzer(args.file, fail_threshold=args.threshold, save_enabled=args.save)

    if args.studies:
        s_metrics = parse_metrics(args.study_metrics, len(args.studies), "studies")
        for factor, metric in zip(args.studies, s_metrics):
            analyzer.run_study(factor, metric)

    if args.interactions:
        i_metrics = parse_metrics(args.interaction_metrics, len(args.interactions), "interactions")
        for pair, metric in zip(args.interactions, i_metrics):
            x, hue = pair.split(':')
            analyzer.run_interaction_study(x, hue, metric)

    analyzer.close()