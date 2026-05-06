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
from openpyxl import load_workbook
from openpyxl.styles import PatternFill
from openpyxl.formatting.rule import CellIsRule

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
        
        self.df[metric] = pd.to_numeric(self.df[metric], errors='coerce')
        clean_df = self.df.dropna(subset=[metric]).copy()
        is_error_metric = any(x in metric.upper() for x in ["MSE", "LOSS", "MAE"])
        
        # 1. Estadísticas y Quality Score
        stats_table = clean_df.groupby(group_by)[metric].agg(['mean', 'std', 'count', 'max']).copy()
        stats_table['CV%'] = (stats_table['std'] / stats_table['mean']) * 100
        
        if is_error_metric:
            stats_table['Quality_Score'] = stats_table['mean'] * (1 + stats_table['CV%']/100)
            ascending_final = True
        else:
            stats_table['Quality_Score'] = stats_table['mean'] * (1 - stats_table['CV%']/100)
            ascending_final = False
        
        stats_table = stats_table.sort_values('Quality_Score', ascending=ascending_final)

        # 2. Normatividad
        norm_results = {idx: self.check_normality_full(clean_df[clean_df[group_by]==idx][metric], idx) for idx in stats_table.index}

        # 3. Análisis Pairwise (Crea el segundo Excel)
        if self.save_enabled:
            self.run_pairwise_analysis(clean_df, stats_table.index, metric, group_by, norm_results)

        # 4. Análisis Best vs Rest (Para el Excel 1)
        p_values, sig_status = [], []
        if len(stats_table) > 1:
            best_idx = stats_table.index[0]
            d_best = clean_df[clean_df[group_by] == best_idx][metric]
            
            p_values.append(1.0); sig_status.append("N/A (BEST)")
            
            for other_idx in stats_table.index[1:]:
                d_other = clean_df[clean_df[group_by] == other_idx][metric]
                _, p = stats.ttest_ind(d_best, d_other, equal_var=False) if (norm_results[best_idx] and norm_results[other_idx]) else stats.mannwhitneyu(d_best, d_other)
                p_values.append(round(p, 6))
                sig_status.append("YES" if p < 0.05 else "NO")

        if len(p_values) == len(stats_table):
            stats_table['p-value vs Best'] = p_values
            stats_table['Is_Significant'] = sig_status

        if self.writer:
            sheet_name = f"BestVsRest_{metric}"[:30].replace(" ", "_")
            stats_table.to_excel(self.writer, sheet_name=sheet_name)
        # 5. Visualization (Top/Mid/Bottom Subset)
        plot_indices = self.get_smart_subset(stats_table)
        viz_df = clean_df[clean_df[group_by].isin(plot_indices)].copy()
        viz_df[group_by] = pd.Categorical(viz_df[group_by], categories=plot_indices, ordered=True)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        sns.boxplot(x=group_by, y=metric, data=viz_df, palette=self.my_colors, ax=axes[0], showfliers=True)
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
    
    def run_pairwise_analysis(self, data_df, ordered_indices, metric, group_by, norm_results):
        """Genera una matriz de p-valores con colores (Verde: Sig, Rojo: No Sig)."""
        self.logger("\n>>> Generando matriz Pairwise con formato condicional...")
        
        limit = 100
        indices = ordered_indices[:limit]
        matrix = pd.DataFrame(index=indices, columns=indices)
        
        for i in indices:
            for j in indices:
                if i == j:
                    matrix.loc[i, j] = None  # Diagonal vacía
                else:
                    d1 = data_df[data_df[group_by] == i][metric]
                    d2 = data_df[data_df[group_by] == j][metric]
                    
                    if norm_results[i] and norm_results[j]:
                        _, p = stats.ttest_ind(d1, d2, equal_var=False)
                    else:
                        _, p = stats.mannwhitneyu(d1, d2)
                    matrix.loc[i, j] = round(p, 6)

        # Guardar archivo temporalmente
        pairwise_path = os.path.join(self.out_dir, f"pairwise_{metric.replace(' ','_')}.xlsx")
        matrix.to_excel(pairwise_path)

        # --- APLICAR COLORES CON OPENPYXL ---
        wb = load_workbook(pairwise_path)
        ws = wb.active

        # Colores (Estilo Excel: ARGB)
        green_fill = PatternFill(start_color='C6EFCE', end_color='C6EFCE', fill_type='solid') # Verde claro
        red_fill = PatternFill(start_color='FFC7CE', end_color='FFC7CE', fill_type='solid')   # Rojo claro

        # Determinar el rango de celdas con datos (ej. B2:Z26)
        max_col = ws.max_column
        max_row = ws.max_row
        # Convertimos coordenadas a formato Excel (A1)
        import openpyxl.utils as utils
        cell_range = f"B2:{utils.get_column_letter(max_col)}{max_row}"

        # Regla 1: Si es menor a 0.05 -> VERDE (Significativo)
        ws.conditional_formatting.add(cell_range,
            CellIsRule(operator='lessThan', formula=['0.05'], fill=green_fill))

        # Regla 2: Si es mayor o igual a 0.05 -> ROJO (No significativo)
        ws.conditional_formatting.add(cell_range,
            CellIsRule(operator='greaterThanOrEqual', formula=['0.05'], fill=red_fill))

        wb.save(pairwise_path)
        self.logger(f"Matriz coloreada guardada en: {pairwise_path}")

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
    parser.add_argument("--study_metrics", nargs='+')
    parser.add_argument("--interactions", nargs='+')
    parser.add_argument("--interaction_metrics", nargs='+')
    parser.add_argument("--save", action="store_true")
    
    args = parser.parse_args()
    analyzer = QNNAnalyzer(args.file, save_enabled=args.save)

    # LÓGICA INTELIGENTE:
    # Si el usuario no define estudios, comparamos las "Config_ID" (las filas agrupadas de Excel)
    if not args.studies and not args.interactions:
        config_column = analyzer._create_config_id()
        
        # Usamos tu métrica principal de Power Query por defecto
        default_metric = args.study_metrics if args.study_metrics else "global open R2"
        
        analyzer.logger(f"\n[MODO AUTOMÁTICO] Comparando configuraciones basadas en GroupKeys de Power Query.")
        for i in default_metric:
            analyzer.run_study(config_column, i)

    else:
        # Lógica manual si se pasan argumentos (se mantiene igual)
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