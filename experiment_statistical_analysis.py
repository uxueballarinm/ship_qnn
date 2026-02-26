import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import warnings
import os
import sys

# 1. SILENCE WARNINGS
warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Cambria"],
    "axes.titleweight": "bold",
    "axes.labelweight": "normal",
    "font.size": 12
})
class QNNAnalyzer:
    def __init__(self, file_path, sheet_name=0, fail_threshold=-1.0):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Setup Directories
        self.base_dir = os.path.dirname(file_path)
        self.out_dir = os.path.join(self.base_dir, "statistical_analysis")
        os.makedirs(self.out_dir, exist_ok=True)
        
        # Setup Logging
        self.log_file = open(os.path.join(self.out_dir, "study_results_log.txt"), "w", encoding="utf-8")
        
        # Load Data
        self.df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
        self.fail_threshold = fail_threshold
        
        # Palette & Style
        self.my_colors = ["#B2EBF2", "#4DB6AC", "#00695C"] 
        self.line_color = "#002B2E" 
        
        self._preprocess()

    def _preprocess(self):
        if 'map' in self.df.columns:
            self.df['map'] = self.df['map'].astype(str).str.replace(" ", "")

    def logger(self, text):
        """Prints to terminal and writes to the .txt log file."""
        print(text)
        self.log_file.write(text + "\n")

    def run_study(self, group_by, metric='global open R2'):
        header = f"\n{'='*75}\nGENERAL STUDY: {metric} by {group_by}\n{'='*75}"
        self.logger(header)

        clean_df = self.df[self.df[metric] > self.fail_threshold].copy()
        
        # Stats Table
        stats_table = clean_df.groupby(group_by)[metric].agg(['mean', 'std', 'count', 'max']).sort_values('mean', ascending=False)
        self.logger("\nSummary Statistics Table:")
        self.logger(stats_table.round(5).to_string())

        # Save table to CSV for Excel use
        stats_table.to_csv(os.path.join(self.out_dir, f"stats_{group_by}_{metric[:10]}.csv"))

        # Significance
        if len(stats_table) > 1:
            best_group = stats_table.index[0]
            self.logger(f"\n--- Significance Tests (vs. Best: {best_group}) ---")
            for other in stats_table.index[1:]:
                data_best = clean_df[clean_df[group_by] == best_group][metric]
                data_other = clean_df[clean_df[group_by] == other][metric]
                if len(data_best) > 1 and len(data_other) > 1:
                    t_stat, p_val = stats.ttest_ind(data_best, data_other, equal_var=False)
                    sig = "SIGNIFICANT" if p_val < 0.05 else "not sig."
                    self.logger(f"{str(best_group):<15} vs {str(other):<20} | p: {p_val:.4f} | {sig}")

        # Plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=group_by, y=metric, data=clean_df, hue=group_by, palette=self.my_colors, 
                    showfliers=False, legend=False, boxprops=dict(edgecolor=self.line_color))
        sns.swarmplot(x=group_by, y=metric, data=clean_df, color=self.line_color, alpha=0.5)
        plt.title(f"General Study: {metric} per {group_by}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save Plot
        plt.savefig(os.path.join(self.out_dir, f"plot_{group_by}_{metric[:10]}.png"), dpi=300)
        plt.show()

    def run_interaction_study(self, factor_x, factor_hue, metric='global open R2'):
        header = f"\n{'='*75}\nINTERACTION STUDY: {metric} ({factor_x} x {factor_hue})\n{'='*75}"
        self.logger(header)

        clean_df = self.df[self.df[metric] > self.fail_threshold].copy()
        
        pivot_mean = clean_df.pivot_table(index=factor_x, columns=factor_hue, values=metric, aggfunc='mean')
        self.logger("\nMean Interaction Table:")
        self.logger(pivot_mean.round(5).to_string())
        
        # Save pivot table
        pivot_mean.to_csv(os.path.join(self.out_dir, f"interaction_{factor_x}_{factor_hue}.csv"))

        # Plot
        plt.figure(figsize=(14, 7))
        ax = sns.boxplot(x=factor_x, y=metric, hue=factor_hue, data=clean_df, palette=self.my_colors, 
                         showfliers=False, boxprops=dict(edgecolor=self.line_color))
        sns.swarmplot(x=factor_x, y=metric, hue=factor_hue, data=clean_df, color=self.line_color, 
                      dodge=True, alpha=0.4, legend=False)
        
        xticks = ax.get_xticks()
        for i in range(len(xticks) - 1):
            plt.axvline(x=(xticks[i] + xticks[i+1]) / 2, color=self.line_color, linestyle='--', alpha=0.2)

        plt.title(f"Interaction: {metric} ({factor_x} by {factor_hue})")
        plt.xticks(rotation=45, ha='right')
        plt.legend(title=factor_hue, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Save Plot
        plt.savefig(os.path.join(self.out_dir, f"interaction_{factor_x}_{factor_hue}.png"), dpi=300)
        plt.show()

    def close(self):
        self.log_file.close()
        print(f"\n[Done] All results saved in: {self.out_dir}")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import warnings
import os

# 1. SILENCE WARNINGS
warnings.filterwarnings("ignore")

class QNNAnalyzer:
    def __init__(self, file_path, sheet_name=0, fail_threshold=-1.0):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Setup Directories
        self.base_dir = os.path.dirname(file_path)
        self.out_dir = os.path.join(self.base_dir, "statistical_analysis")
        os.makedirs(self.out_dir, exist_ok=True)
        
        # Setup Files
        self.log_path = os.path.join(self.out_dir, "study_results_log.txt")
        self.excel_out_path = os.path.join(self.out_dir, "analysis_summary_report.xlsx")
        
        # Open Log and Excel Writer
        self.log_file = open(self.log_path, "w", encoding="utf-8")
        self.writer = pd.ExcelWriter(self.excel_out_path, engine='openpyxl')
        
        # Load Data
        self.df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
        self.fail_threshold = fail_threshold
        
        # Palette & Style (High contrast Blue-Greens)
        self.my_colors = ["#B2EBF2", "#4DB6AC", "#00695C"] 
        self.line_color = "#002B2E" # Deep Navy-Teal
        
        self._preprocess()

    def _preprocess(self):
        if 'map' in self.df.columns:
            self.df['map'] = self.df['map'].astype(str).str.replace(" ", "")

    def logger(self, text):
        """Prints to terminal and writes to the .txt log file."""
        print(text)
        self.log_file.write(text + "\n")

    def _get_sheet_name(self, name):
        """Ensures sheet name is under 31 characters for Excel."""
        clean_name = "".join(x for x in name if x.isalnum() or x in " _")
        return clean_name[:31]

    def run_study(self, group_by, metric='global open R2'):
        header = f"\n{'='*75}\nGENERAL STUDY: {metric} by {group_by}\n{'='*75}"
        self.logger(header)

        clean_df = self.df[self.df[metric] > self.fail_threshold].copy()
        
        # Stats Table
        stats_table = clean_df.groupby(group_by)[metric].agg(['mean', 'std', 'count', 'max']).sort_values('mean', ascending=False)
        self.logger("\nSummary Statistics Table:")
        self.logger(stats_table.round(5).to_string())

        # Save to its own sheet in the single Excel file
        sheet_id = self._get_sheet_name(f"Gen_{group_by}_{metric.split()[-1]}")
        stats_table.to_excel(self.writer, sheet_name=sheet_id)

        # Significance
        if len(stats_table) > 1:
            best_group = stats_table.index[0]
            self.logger(f"\n--- Significance Tests (vs. Best: {best_group}) ---")
            for other in stats_table.index[1:]:
                data_best = clean_df[clean_df[group_by] == best_group][metric]
                data_other = clean_df[clean_df[group_by] == other][metric]
                if len(data_best) > 1 and len(data_other) > 1:
                    t_stat, p_val = stats.ttest_ind(data_best, data_other, equal_var=False)
                    sig = "SIGNIFICANT" if p_val < 0.05 else "not sig."
                    self.logger(f"{str(best_group):<15} vs {str(other):<20} | p: {p_val:.4f} | {sig}")

        # Plot
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=group_by, y=metric, data=clean_df, hue=group_by, palette=self.my_colors, 
                    showfliers=False, legend=False, boxprops=dict(edgecolor=self.line_color))
        sns.swarmplot(x=group_by, y=metric, data=clean_df, color=self.line_color, alpha=0.5)
        plt.title(f"General Study: {metric} per {group_by}")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', color=self.line_color, linestyle=':', alpha=0.15)
        plt.tight_layout()
        
        # Save Plot image
        plt.savefig(os.path.join(self.out_dir, f"plot_{group_by}_{metric.replace(' ', '_')}.png"), dpi=300)
        plt.show()

    def run_interaction_study(self, factor_x, factor_hue, metric='global open R2'):
        header = f"\n{'='*75}\nINTERACTION STUDY: {metric} ({factor_x} x {factor_hue})\n{'='*75}"
        self.logger(header)

        clean_df = self.df[self.df[metric] > self.fail_threshold].copy()
        
        # Pivot Table
        pivot_mean = clean_df.pivot_table(index=factor_x, columns=factor_hue, values=metric, aggfunc='mean')
        self.logger("\nMean Interaction Table:")
        self.logger(pivot_mean.round(5).to_string())
        
        # Save to its own sheet in the same Excel file
        sheet_id = self._get_sheet_name(f"Int_{factor_x}_{factor_hue}")
        pivot_mean.to_excel(self.writer, sheet_name=sheet_id)

        # Plot
        plt.figure(figsize=(14, 7))
        ax = sns.boxplot(x=factor_x, y=metric, hue=factor_hue, data=clean_df, palette=self.my_colors, 
                         showfliers=False, boxprops=dict(edgecolor=self.line_color))
        sns.swarmplot(x=factor_x, y=metric, hue=factor_hue, data=clean_df, color=self.line_color, 
                      dodge=True, alpha=0.4, legend=False)
        
        # Dotted separators between categories
        xticks = ax.get_xticks()
        for i in range(len(xticks) - 1):
            plt.axvline(x=(xticks[i] + xticks[i+1]) / 2, color=self.line_color, linestyle='--', alpha=0.2)

        plt.title(f"Interaction: {metric} ({factor_x} by {factor_hue})")
        plt.xticks(rotation=45, ha='right')
        plt.legend(title=factor_hue, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', color=self.line_color, linestyle=':', alpha=0.15)
        plt.tight_layout()
        
        # Save Plot image
        plt.savefig(os.path.join(self.out_dir, f"int_{factor_x}_{factor_hue}_{metric.replace(' ', '_')}.png"), dpi=300)
        plt.show()

    def close(self):
        """Closes files and saves the Excel workbook."""
        self.log_file.close()
        self.writer.close()
        print(f"\n[Done] Results saved in folder: {self.out_dir}")
        print(f"  > Summary Excel: analysis_summary_report.xlsx")
        print(f"  > Terminal Log: study_results_log.txt")
        print(f"  > Plots: [.png files]")


# ==============================================================================
# MAIN CONFIGURATION
# ==============================================================================
if __name__ == "__main__":
    # Your Excel path
    FILE_PATH = r"logs\experiments_mapping_order_encoding\experiments_summary.xlsx"
    
    # Initialize analyzer
    analyzer = QNNAnalyzer(FILE_PATH, sheet_name=0, fail_threshold=-1.0)

    # 1. Run General Study (Shows full terminal stats + Best vs Rest)
    analyzer.run_study(group_by='encoding', metric='global open R2')
    analyzer.run_study(group_by='map', metric='global open R2')
    analyzer.run_study(group_by='reps', metric='global open R2')

    # 2. Run Interaction Study (Shows dotted separators + Pivot Table)
    analyzer.run_interaction_study(factor_x='map', factor_hue='encoding', metric='global open R2')

    # 3. Analyze specific target
    analyzer.run_interaction_study(factor_x='encoding', factor_hue='reps', metric='global open R2')

    analyzer.run_interaction_study(factor_x='map', factor_hue='reps', metric='global open R2')
    analyzer.close()

