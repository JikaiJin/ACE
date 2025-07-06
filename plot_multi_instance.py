import sys
import os
import argparse
import glob
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Computer Modern Roman']
matplotlib.rcParams['axes.labelweight'] = 'bold'
matplotlib.rcParams['axes.titleweight'] = 'bold'
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats # Added for t-distribution confidence intervals

# Custom Log Tick Formatter (enhanced to show more ticks)
def custom_log_tick_formatter(val, pos):
    log10_val = np.log10(val)
    # Check for exact powers of 10
    if np.isclose(log10_val, np.round(log10_val)):
        exponent = int(np.round(log10_val))
        if exponent == 0:
            return "1"
        else:
            return r"$10^{{{}}}$".format(exponent)
    # Also show intermediate values like 2×10^n, 5×10^n
    elif np.isclose(val, 2 * 10**np.floor(log10_val)):
        exponent = int(np.floor(log10_val))
        if exponent == 0:
            return "2"
        else:
            return r"$2 \times 10^{{{}}}$".format(exponent)
    elif np.isclose(val, 5 * 10**np.floor(log10_val)):
        exponent = int(np.floor(log10_val))
        if exponent == 0:
            return "5"
        else:
            return r"$5 \times 10^{{{}}}$".format(exponent)
    else:
        return "" # No label for other ticks

def main(args):
    '''
    This function generates plots for comparing multiple instances.
    We will study the effect of varying sample size, support size and correlated treatment noise.
    '''
    os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
    parser = argparse.ArgumentParser(
        description="ACE estimators!")
    parser.add_argument("--input_dir", dest="input_dir", type=str, default=".")
    parser.add_argument("--output_dir", dest="output_dir", type=str, default=".")
    opts = parser.parse_args(args)

    multi_instance_dump_dir = opts.input_dir
    N_SAMPLES_DIR = os.path.join(opts.output_dir, "n_samples")
    N_SUPP_DIR = os.path.join(opts.output_dir, "n_support")
    N_COR_DIR = os.path.join(opts.output_dir, "cor")

    for dir in [N_SAMPLES_DIR, N_SUPP_DIR, N_COR_DIR]:
        if not os.path.exists(dir):
            os.makedirs(dir)
    

    scaled_n_samples = np.array([5 * i for i in range(1,17)])
    scale = 100
    n_samples = scale * scaled_n_samples
    methods = ['1st order', '2nd order', '3rd order'] + [str(i) + 'th order' for i in range(4,11)]
    method_inds = list(range(12))

    def collect_results(rec_tau_list):
        mse, bias, std = [], [], []
        for rec_tau in rec_tau_list:
            _bias, _std = np.mean(rec_tau, axis=0) - 3, np.std(rec_tau, axis=0)
            mse.append(_bias**2 + _std**2)
            bias.append(_bias)
            std.append(_std)
        return np.array(mse), np.array(bias), np.array(std)

    
    def plot_estimates_sample(rec_tau_list, methods, method_inds, save_dir, save_name_prefix='results_comparison'):
        """
        Plot estimates vs sample size comparison for multiple methods.
        Generates both CI and IQR versions of the plots.
        """
        if len(rec_tau_list) == 0:
            print("rec_tau_list is empty, skipping plot generation")
            return None, None, None, None, None, None, None, None

        rec_tau_array = np.array(rec_tau_list) # Shape: (n_sample_sizes, n_experiments, n_methods)
        true_effect = 3.0
        all_bias = rec_tau_array - true_effect  # Raw bias (no absolute value)

        sns.set_style("whitegrid")
        
        # Setup figure parameters
        figsize = (8, 5)
        cmap = plt.cm.Set1
        line_styles = ['-', '--', ':', '-.', '-']
        markers = ['*', 'o', 's', '^', 'D']
        
        plt.rcParams['axes.labelweight'] = 'bold'
        
        # Create figures for both CI and IQR versions
        # CI version figures
        fig_ci, ax_ci = plt.subplots(figsize=figsize)
        fig2_ci, ax2_ci = plt.subplots(figsize=figsize)
        fig3_ci, ax3_ci = plt.subplots(figsize=figsize)
        fig4_ci, ax4_ci = plt.subplots(figsize=figsize)
        
        # IQR version figures  
        fig_iqr, ax_iqr = plt.subplots(figsize=figsize)
        fig2_iqr, ax2_iqr = plt.subplots(figsize=figsize)
        fig3_iqr, ax3_iqr = plt.subplots(figsize=figsize)
        fig4_iqr, ax4_iqr = plt.subplots(figsize=figsize)
        
        all_axes = [ax_ci, ax2_ci, ax3_ci, ax4_ci, ax_iqr, ax2_iqr, ax3_iqr, ax4_iqr]
        
        for axis_item in all_axes:
            axis_item.spines['top'].set_visible(False)
            axis_item.spines['right'].set_visible(False)
            axis_item.spines['bottom'].set_linewidth(1.5)
            axis_item.spines['left'].set_linewidth(1.5)

        r = min(len(methods), len(method_inds))
        n_samples = np.array([500*i for i in range(1,17)])
        scaled_n_samples = n_samples / 1000.0  # Convert to thousands
        
        # Process data for each method
        for i, m in zip(method_inds[0:r], methods[0:r]):
            if i in range(5) and i != 3:  # Skip 4th order (index 3) 
                # ============ CI VERSION CALCULATIONS ============
                # --- RMSE with CI ---
                mean_rmses_ci = []
                ci_lower_rmses = []
                ci_upper_rmses = []

                for s_idx in range(rec_tau_array.shape[0]):
                    current_estimates = rec_tau_array[s_idx, :, i]
                    squared_errors = (current_estimates - true_effect)**2
                    rmse_values = np.sqrt(squared_errors)
                    
                    mean_rmse = np.mean(rmse_values)
                    n_experiments = len(rmse_values)
                    std_err = np.std(rmse_values, ddof=1) / np.sqrt(n_experiments)
                    
                    confidence_level = 0.95
                    alpha = 1 - confidence_level
                    t_critical = stats.t.ppf(1 - alpha/2, df=n_experiments-1)
                    
                    ci_lower = mean_rmse - t_critical * std_err
                    ci_upper = mean_rmse + t_critical * std_err
                    
                    mean_rmses_ci.append(mean_rmse)
                    ci_lower_rmses.append(max(ci_lower, 0))
                    ci_upper_rmses.append(ci_upper)
                
                mean_rmses_ci = np.array(mean_rmses_ci)
                ci_lower_rmses = np.array(ci_lower_rmses)
                ci_upper_rmses = np.array(ci_upper_rmses)

                # --- Bias with CI (RAW BIAS, not absolute) ---
                bias_method_i = all_bias[:, :, i]  # Shape: (n_sample_sizes, n_experiments)
                mean_bias_ci = []
                ci_lower_bias = []
                ci_upper_bias = []
                
                for s_idx in range(bias_method_i.shape[0]):
                    bias_values = bias_method_i[s_idx, :]  # Raw bias values (can be positive or negative)
                    
                    mean_bias = np.mean(bias_values)
                    n_experiments = len(bias_values)
                    std_err = np.std(bias_values, ddof=1) / np.sqrt(n_experiments)
                    
                    t_critical = stats.t.ppf(1 - alpha/2, df=n_experiments-1)
                    
                    ci_lower = mean_bias - t_critical * std_err
                    ci_upper = mean_bias + t_critical * std_err
                    
                    mean_bias_ci.append(mean_bias)
                    ci_lower_bias.append(ci_lower)
                    ci_upper_bias.append(ci_upper)
                
                mean_bias_ci = np.array(mean_bias_ci)
                ci_lower_bias = np.array(ci_lower_bias)
                ci_upper_bias = np.array(ci_upper_bias)

                # ============ IQR VERSION CALCULATIONS ============
                # --- RMSE with IQR ---
                rmses_iqr = []
                q25_sqrt_sq_errors = []
                q75_sqrt_sq_errors = []

                for s_idx in range(rec_tau_array.shape[0]):
                    current_estimates = rec_tau_array[s_idx, :, i]
                    squared_errors = (current_estimates - true_effect)**2
                    sqrt_squared_errors = np.sqrt(squared_errors)

                    rmses_iqr.append(np.sqrt(np.median(squared_errors)))
                    q25_sqrt_sq_errors.append(np.percentile(sqrt_squared_errors, 25))
                    q75_sqrt_sq_errors.append(np.percentile(sqrt_squared_errors, 75))
                
                rmses_iqr = np.array(rmses_iqr)
                q25_sqrt_sq_errors = np.array(q25_sqrt_sq_errors)
                q75_sqrt_sq_errors = np.array(q75_sqrt_sq_errors)

                # --- Bias with IQR (RAW BIAS, not absolute) ---
                q25_bias_iqr = np.percentile(bias_method_i, 25, axis=1)
                median_bias_iqr = np.percentile(bias_method_i, 50, axis=1)
                q75_bias_iqr = np.percentile(bias_method_i, 75, axis=1)

                # ============ PLOTTING CI VERSION ============
                # Plot ax_ci (RMSE with CI)
                ax_ci.plot(scaled_n_samples, mean_rmses_ci, marker=markers[i], markersize=12, 
                          markeredgecolor='black', markeredgewidth=0.8,
                          linestyle=line_styles[i], linewidth=2.0, color=cmap(i), label=m)
                ax_ci.fill_between(scaled_n_samples, ci_lower_rmses, ci_upper_rmses, 
                                  color=cmap(i), alpha=0.2, label='_nolegend_')
                ax_ci.plot(scaled_n_samples, ci_lower_rmses, marker=markers[i], markersize=8,
                          linestyle='--', linewidth=1.5, color=cmap(i), alpha=0.5, label='_nolegend_')
                ax_ci.plot(scaled_n_samples, ci_upper_rmses, marker=markers[i], markersize=8,
                          linestyle=':', linewidth=1.5, color=cmap(i), alpha=0.5, label='_nolegend_')
                
                # Plot ax3_ci (Bias with CI - RAW BIAS)
                ax3_ci.plot(scaled_n_samples, mean_bias_ci,
                           marker=markers[i], markersize=12, markeredgecolor='black', markeredgewidth=0.5,
                           linestyle=line_styles[i], linewidth=2.0, color=cmap(i), label=m)
                ax3_ci.fill_between(scaled_n_samples, ci_lower_bias, ci_upper_bias,
                                   color=cmap(i), alpha=0.2, label='_nolegend_')
                ax3_ci.plot(scaled_n_samples, ci_lower_bias, marker=markers[i], markersize=8,
                           linestyle='--', linewidth=1.5, color=cmap(i), alpha=0.5, label='_nolegend_')
                ax3_ci.plot(scaled_n_samples, ci_upper_bias, marker=markers[i], markersize=8,
                           linestyle=':', linewidth=1.5, color=cmap(i), alpha=0.5, label='_nolegend_')

                # ============ PLOTTING IQR VERSION ============
                # Plot ax_iqr (RMSE with IQR)
                ax_iqr.plot(scaled_n_samples, rmses_iqr, marker=markers[i], markersize=12, 
                           markeredgecolor='black', markeredgewidth=0.8,
                           linestyle=line_styles[i], linewidth=2.0, color=cmap(i), label=m)
                ax_iqr.fill_between(scaled_n_samples, q25_sqrt_sq_errors, q75_sqrt_sq_errors, 
                                   color=cmap(i), alpha=0.2, label='_nolegend_')
                ax_iqr.plot(scaled_n_samples, q25_sqrt_sq_errors, marker=markers[i], markersize=8,
                           linestyle='--', linewidth=1.5, color=cmap(i), alpha=0.5, label='_nolegend_')
                ax_iqr.plot(scaled_n_samples, q75_sqrt_sq_errors, marker=markers[i], markersize=8,
                           linestyle=':', linewidth=1.5, color=cmap(i), alpha=0.5, label='_nolegend_')
                
                # Plot ax3_iqr (Bias with IQR - RAW BIAS)
                ax3_iqr.plot(scaled_n_samples, median_bias_iqr,
                            marker=markers[i], markersize=12, markeredgecolor='black', markeredgewidth=0.5,
                            linestyle=line_styles[i], linewidth=2.0, color=cmap(i), label=m)
                ax3_iqr.fill_between(scaled_n_samples, q25_bias_iqr, q75_bias_iqr,
                                    color=cmap(i), alpha=0.2, label='_nolegend_')
                ax3_iqr.plot(scaled_n_samples, q25_bias_iqr, marker=markers[i], markersize=8,
                            linestyle='--', linewidth=1.5, color=cmap(i), alpha=0.5, label='_nolegend_')
                ax3_iqr.plot(scaled_n_samples, q75_bias_iqr, marker=markers[i], markersize=8,
                            linestyle=':', linewidth=1.5, color=cmap(i), alpha=0.5, label='_nolegend_')

                # ============ BOXPLOTS (SAME FOR BOTH VERSIONS) ============
                for ax2_version in [ax2_ci, ax2_iqr]:
                    if i == 0:  # 1st order method
                        positions_1st = np.arange(rec_tau_array.shape[0]) - 0.15
                        boxplot_data_1st = [rec_tau_array[s_idx, :, i] for s_idx in range(rec_tau_array.shape[0])]
                        bp_1st = ax2_version.boxplot(boxplot_data_1st, positions=positions_1st, widths=0.25, 
                                                   patch_artist=True, showfliers=True, 
                                                   boxprops=dict(facecolor=cmap(i), alpha=0.7),
                                                   medianprops=dict(color='black', linewidth=1.5))
                    elif i == 4:  # 5th order method  
                        positions_5th = np.arange(rec_tau_array.shape[0]) + 0.15
                        boxplot_data_5th = [rec_tau_array[s_idx, :, i] for s_idx in range(rec_tau_array.shape[0])]
                        bp_5th = ax2_version.boxplot(boxplot_data_5th, positions=positions_5th, widths=0.25, 
                                                   patch_artist=True, showfliers=True,
                                                   boxprops=dict(facecolor=cmap(i), alpha=0.7),
                                                   medianprops=dict(color='black', linewidth=1.5))
                        ax2_version.axhline(y=true_effect, color='red', linestyle='-', linewidth=2, alpha=0.8, label='True Effect')

                # ============ STANDARD DEVIATION PLOTS (SAME FOR BOTH VERSIONS) ============
                for ax4_version in [ax4_ci, ax4_iqr]:
                    empirical_stds_method_i = []
                    for s_idx in range(rec_tau_array.shape[0]):
                        current_estimates = rec_tau_array[s_idx, :, i]
                        empirical_stds_method_i.append(np.std(current_estimates))
                    
                    empirical_stds_method_i = np.array(empirical_stds_method_i)
                    ax4_version.plot(scaled_n_samples, empirical_stds_method_i,
                                   marker=markers[i], markersize=12, markeredgecolor='black', markeredgewidth=0.5,
                                   linestyle=line_styles[i], linewidth=2.0, color=cmap(i), label=m)

        # ============ AXIS FORMATTING ============
        tick_step = 3 
        
        # Calculate dynamic minimum y-axis limits based on actual data (keep original maximums)
        # Only use mean/median values for setting limits, not confidence interval bounds
        all_rmse_ci_means = []
        all_bias_ci_means = []
        all_std_data = []
        all_boxplot_data = []
        all_rmse_iqr_medians = []
        all_bias_iqr_medians = []
        
        for i, m in zip(method_inds[0:r], methods[0:r]):
            if i in range(5) and i != 3:  # Skip 4th order (index 3)
                # For CI version - collect only the mean values, not CI bounds
                for s_idx in range(rec_tau_array.shape[0]):
                    current_estimates = rec_tau_array[s_idx, :, i]
                    squared_errors = (current_estimates - true_effect)**2
                    rmse_values = np.sqrt(squared_errors)
                    sqrt_squared_errors = np.sqrt(squared_errors)
                    bias_values = current_estimates - true_effect  # Raw bias (not absolute)
                    
                    # Only collect mean values for CI version
                    mean_rmse = np.mean(rmse_values)
                    mean_bias = np.mean(bias_values)  # Can be positive or negative
                    
                    all_rmse_ci_means.append(mean_rmse)
                    all_bias_ci_means.append(mean_bias)
                    all_std_data.append(np.std(current_estimates))
                    
                    # Only collect median/percentile center values for IQR version
                    rmse_iqr = np.sqrt(np.median(squared_errors))
                    median_bias = np.percentile(bias_values, 50)  # Can be positive or negative
                    
                    all_rmse_iqr_medians.append(rmse_iqr)
                    all_bias_iqr_medians.append(median_bias)
                    
                # Boxplot data
                if i == 0 or i == 4:  # Only for methods that get boxplotted
                    for s_idx in range(rec_tau_array.shape[0]):
                        all_boxplot_data.extend(rec_tau_array[s_idx, :, i])
        
        # Calculate minimums with small margins (keep original maximums)
        min_rmse_ci = min(all_rmse_ci_means) * 0.8 if all_rmse_ci_means else 8e-3
        # For bias, use symmetric range around zero since bias can be positive or negative
        max_abs_bias_ci = max(abs(b) for b in all_bias_ci_means) if all_bias_ci_means else 0.1
        min_bias_ci = -max_abs_bias_ci * 1.2
        max_bias_ci = max_abs_bias_ci * 1.2
        
        min_std = min(all_std_data) * 0.8 if all_std_data else 2e-2
        min_boxplot = min(all_boxplot_data) * 0.95 if all_boxplot_data else 2.85
        min_rmse_iqr = min(all_rmse_iqr_medians) * 0.5 if all_rmse_iqr_medians else 8e-3
        
        # For IQR bias, use symmetric range around zero
        max_abs_bias_iqr = max(abs(b) for b in all_bias_iqr_medians) if all_bias_iqr_medians else 0.1
        min_bias_iqr = -max_abs_bias_iqr * 1.2
        max_bias_iqr = max_abs_bias_iqr * 1.2

        # Common formatting function
        def format_axes(ax_main, ax2_box, ax3_bias, ax4_std, version_suffix):
            # ax_main (RMSE)
            ax_main.set_xlabel(r'\textbf{Sample Size ($\times 10^3$)}', fontsize=24)
            if version_suffix == 'CI':
                ax_main.set_ylabel(r'\textbf{Root Mean Squared Error}', fontsize=24)
                ax_main.set_ylim(min_rmse_ci, 2e-1)  # Keep original max
            else:
                ax_main.set_ylabel(r'\textbf{Root Mean Sq. Error (IQR)}', fontsize=24)
                ax_main.set_ylim(min_rmse_iqr, 2e-1)  # Keep original max
            ax_main.set_yscale("log")
            # Force more tick locations with explicit tick positions
            ax_main.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=15, subs=(1.0, 2.0, 5.0)))
            ax_main.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_tick_formatter))
            ax_main.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2.0, 10.0)))
            # Alternative: Force specific tick locations if needed
            # ax_main.set_yticks([1e-3, 1e-2, 1e-1, 1e0])
            ax_main.set_xticks(scaled_n_samples[::tick_step]) 
            ax_main.set_xticklabels(scaled_n_samples[::tick_step], fontsize=16) 
            ax_main.tick_params(axis='y', labelsize=16)
          
            # ax2_box (Boxplot)
            ax2_box.set_xlabel(r'\textbf{Sample Size ($\times 10^3$)}', fontsize=24)
            ax2_box.set_xticks(np.arange(scaled_n_samples.shape[0])[::tick_step])
            ax2_box.set_xticklabels(scaled_n_samples[::tick_step], fontsize=16)
            ax2_box.set_ylim(min_boxplot, 3.15)  
            ax2_box.set_ylabel(r'\textbf{Estimates}', fontsize=24)
            ax2_box.tick_params(axis='y', labelsize=16)
          
            ax3_bias.set_xlabel(r'\textbf{Sample Size ($\times 10^3$)}', fontsize=24)
            if version_suffix == 'CI':
                ax3_bias.set_ylabel(r'\textbf{Bias}', fontsize=24)
                ax3_bias.set_ylim(min_bias_ci, max_bias_ci)
            else:
                ax3_bias.set_ylabel(r'\textbf{Median Bias (IQR)}', fontsize=24)
                ax3_bias.set_ylim(min_bias_iqr, max_bias_iqr)
            ax3_bias.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)  
            ax3_bias.set_xticks(scaled_n_samples[::tick_step]) 
            ax3_bias.set_xticklabels(scaled_n_samples[::tick_step], fontsize=16)
            ax3_bias.tick_params(axis='y', labelsize=16)

            ax4_std.set_xlabel(r'\textbf{Sample Size ($\times 10^3$)}', fontsize=24)
            ax4_std.set_ylabel(r'\textbf{Standard Deviation}', fontsize=24)
            ax4_std.set_yscale("log")
            ax4_std.set_ylim(min_std, 2e-1)  
            ax4_std.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=15, subs=(1.0, 2.0, 5.0)))
            ax4_std.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_tick_formatter))
            ax4_std.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2.0, 10.0)))
            ax4_std.set_xticks(scaled_n_samples[::tick_step])
            ax4_std.set_xticklabels(scaled_n_samples[::tick_step], fontsize=16)
            ax4_std.tick_params(axis='y', labelsize=16)

        format_axes(ax_ci, ax2_ci, ax3_ci, ax4_ci, 'CI')
        format_axes(ax_iqr, ax2_iqr, ax3_iqr, ax4_iqr, 'IQR')

        all_figures_axes = [
            (fig_ci, ax_ci), (fig2_ci, ax2_ci), (fig3_ci, ax3_ci), (fig4_ci, ax4_ci),
            (fig_iqr, ax_iqr), (fig2_iqr, ax2_iqr), (fig3_iqr, ax3_iqr), (fig4_iqr, ax4_iqr)
        ]
        
        for f, a in all_figures_axes:
            leg = a.legend(fontsize=14 if a in [ax3_ci, ax4_ci, ax_ci, ax3_iqr, ax4_iqr, ax_iqr] else 12,
                          frameon=True, fancybox=True, framealpha=0.8, 
                          edgecolor='black', loc='best', ncol=1)
            if leg.get_texts() or leg.get_lines():
                leg.set_title('Methods', prop={'size': 16 if a in [ax3_ci, ax4_ci, ax_ci, ax3_iqr, ax4_iqr, ax_iqr] else 18})
            f.tight_layout()

        save_path_ci = os.path.join(save_dir, f"{save_name_prefix}_rmse_ci.png")
        save_path2_ci = os.path.join(save_dir, f"{save_name_prefix}_boxplot_ci.png")
        save_path3_ci = os.path.join(save_dir, f"{save_name_prefix}_bias_ci.png")
        save_path4_ci = os.path.join(save_dir, f"{save_name_prefix}_std_ci.png")
        
        save_path_iqr = os.path.join(save_dir, f"{save_name_prefix}_rmse_iqr.png")
        save_path2_iqr = os.path.join(save_dir, f"{save_name_prefix}_boxplot_iqr.png")
        save_path3_iqr = os.path.join(save_dir, f"{save_name_prefix}_bias_iqr.png")
        save_path4_iqr = os.path.join(save_dir, f"{save_name_prefix}_std_iqr.png")

        fig_ci.savefig(save_path_ci, dpi=300, bbox_inches='tight')
        fig2_ci.savefig(save_path2_ci, dpi=300, bbox_inches='tight')
        fig3_ci.savefig(save_path3_ci, dpi=300, bbox_inches='tight')
        fig4_ci.savefig(save_path4_ci, dpi=300, bbox_inches='tight')
        
        fig_iqr.savefig(save_path_iqr, dpi=300, bbox_inches='tight')
        fig2_iqr.savefig(save_path2_iqr, dpi=300, bbox_inches='tight')
        fig3_iqr.savefig(save_path3_iqr, dpi=300, bbox_inches='tight')
        fig4_iqr.savefig(save_path4_iqr, dpi=300, bbox_inches='tight')

        print(f"CI version plots saved: {save_path_ci}, {save_path2_ci}, {save_path3_ci}, {save_path4_ci}")
        print(f"IQR version plots saved: {save_path_iqr}, {save_path2_iqr}, {save_path3_iqr}, {save_path4_iqr}")

        return fig_ci, fig2_ci, fig3_ci, fig4_ci, fig_iqr, fig2_iqr, fig3_iqr, fig4_iqr
    
    
    def plot_estimates_support(rec_tau_list, n_supports, method_inds=method_inds, methods=methods, r=6):
        # Get results data
        mse, bias, std = collect_results(rec_tau_list)
        
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 18
        
        fig, ax = plt.subplots(figsize=(8, 5))
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        
        ax.grid(True, linestyle='--', alpha=0.4)
        ax2.grid(True, linestyle='--', alpha=0.4)
        
        ax2.plot([-0.5, len(n_supports) - 0.5], [3., 3.], linestyle='--', color='#1A5276', 
                linewidth=2.0, label='True Effect', alpha=0.8)
        
        cmap = plt.get_cmap("tab10")
        for i, m in zip(method_inds[0:r], methods[0:r]):
            if i in [0, 1, 4] and i != 3:  # Skip 4th order
                _mse, _bias, _std = mse[:, i], np.abs(bias)[:, i], std[:, i]
                
                ax.plot(n_supports, _mse ** 0.5, marker='*', markersize=12, 
                        markeredgecolor='black', markeredgewidth=0.8,
                        linestyle='-', linewidth=2.0, color=cmap(i), label=m)
            
            # Compare the 2nd and 5th order ACE estimators
            if i == 1 or i == 4:
                flierprops = dict(marker='o', markerfacecolor=cmap(i), markersize=7,
                                markeredgecolor='black', markeredgewidth=0.8,
                                linestyle='none')
                
                box = ax2.boxplot(rec_tau_list[:,:,i].T, whis=[1, 99], patch_artist=True, 
                                positions=list(range(len(n_supports))),
                                medianprops=dict(color=cmap(i), linewidth=2.0), 
                                boxprops=dict(facecolor='white', color=cmap(i), alpha=0.5),
                                whiskerprops=dict(color=cmap(i), linewidth=1.5),
                                capprops=dict(color=cmap(i), linewidth=1.5),
                                showfliers=True, flierprops=flierprops)
                
                ax2.plot([], [], color=cmap(i), linewidth=2.0, label=m)
        
        all_mse_data = []
        all_estimates_data = []
        for i, m in zip(method_inds[0:r], methods[0:r]):
            if i in [0, 1, 4] and i != 3:  
                _mse = mse[:, i]
                all_mse_data.extend(_mse ** 0.5)
            if i == 1 or i == 4:  
                all_estimates_data.extend(rec_tau_list[:,:,i].flatten())
        
        min_mse_support = min(all_mse_data) * 0.8 if all_mse_data else 1e-3
        min_estimates_support = min(all_estimates_data) * 0.95 if all_estimates_data else 2.85
        
        ax.set_ylim(min_mse_support, 3e-1)  
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=15, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_tick_formatter))
        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2.0, 10.0)))
        ax.set_xlabel(r'\textbf{Support Size}', fontsize=24, fontweight='bold')
        ax.set_ylabel(r"\textbf{RMSE}", fontsize=24, fontweight='bold')
        
        ax.set_xticks(n_supports)
        ax.set_xticklabels(n_supports, fontsize=18)
        ax.tick_params(axis='y', labelsize=18)
        
        ax2.set_xlabel(r'\textbf{Support Size}', fontsize=24, fontweight='bold')
        ax2.set_xticks(list(range(len(n_supports))))
        ax2.set_xticklabels(n_supports, fontsize=18)
        ax2.set_ylim(min_estimates_support, 3.15)  # Keep original max
        ax2.set_ylabel(r'\textbf{Estimates}', fontsize=24, fontweight='bold')
        ax2.tick_params(axis='y', labelsize=18)
        
        min_mse_idx = np.argmin(mse[:, method_inds[0]])
        
        
        for f, a in zip([fig, fig2], [ax, ax2]):
            leg = a.legend(fontsize=16, frameon=True, fancybox=True, framealpha=0.9, 
                        edgecolor='black', loc='best')
            leg.set_title('Methods', prop={'size': 18, 'weight': 'bold'})
            
            f.tight_layout()
        
        return fig, fig2
    
    def plot_estimates_correlated(rec_tau_list, cor_list, method_inds=method_inds, methods=methods, r=6):
        
        mse, bias, std = collect_results(rec_tau_list)
        
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 18
        
        fig, ax = plt.subplots(figsize=(9, 5))
        fig2, ax2 = plt.subplots(figsize=(9, 5))
        
        ax.grid(True, linestyle='--', alpha=0.4)
        ax2.grid(True, linestyle='--', alpha=0.4)
        
        ax2.plot([-0.5, 9.5], [3., 3.], linestyle='--', color='#1A5276', 
                linewidth=2.0, label='True Effect', alpha=0.8)
        
        cmap = plt.get_cmap("tab10")
        for i, m in zip(method_inds[0:r], methods[0:r]):
            if (i <= 1 or i == 4) and i != 3:  # Skip 4th order
                _mse, _bias, _std = mse[:, i], np.abs(bias)[:, i], std[:, i]
                
                ax.plot(cor_list, _mse ** 0.5, marker='*', markersize=12, 
                        markeredgecolor='black', markeredgewidth=0.8,
                        linestyle='-', linewidth=2.0, color=cmap(i), label=m)
            
            if i == 1 or i == 4:
                flierprops = dict(marker='o', markerfacecolor=cmap(i), markersize=7,
                                markeredgecolor='black', markeredgewidth=0.8,
                                linestyle='none')
                
                box = ax2.boxplot(rec_tau_list[:,:,i].T, whis=[1, 99], patch_artist=True, 
                                positions=list(range(len(cor_list))),
                                medianprops=dict(color=cmap(i), linewidth=2.0), 
                                boxprops=dict(facecolor='white', color=cmap(i), alpha=0.5),
                                whiskerprops=dict(color=cmap(i), linewidth=1.5),
                                capprops=dict(color=cmap(i), linewidth=1.5),
                                showfliers=True, flierprops=flierprops)
                
                ax2.plot([], [], color=cmap(i), linewidth=2.0, label=m)
        
        all_mse_corr_data = []
        all_estimates_corr_data = []
        for i, m in zip(method_inds[0:r], methods[0:r]):
            if (i <= 1 or i == 4) and i != 3:  
                _mse = mse[:, i]
                all_mse_corr_data.extend(_mse ** 0.5)
            if i == 1 or i == 4:  
                all_estimates_corr_data.extend(rec_tau_list[:,:,i].flatten())
        
        min_mse_corr = min(all_mse_corr_data) * 0.8 if all_mse_corr_data else 6e-4
        min_estimates_corr = min(all_estimates_corr_data) * 0.95 if all_estimates_corr_data else 2.9
        
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=15, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_log_tick_formatter))
        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2.0, 10.0)))
        ax.set_xlabel(r'\textbf{Correlation Coefficient}', fontsize=24, fontweight='bold')
        ax.set_ylabel(r"\textbf{RMSE}", fontsize=24, fontweight='bold')
        ax.set_ylim(min_mse_corr, 5e-2) 
        
        ax.set_xticks(cor_list)
        ax.set_xticklabels([f"{c:.2f}" for c in cor_list], fontsize=18)
        ax.tick_params(axis='y', labelsize=18)
        
        ax2.set_xticks(list(range(len(cor_list))))
        ax2.set_xticklabels([f"{c:.1f}" for c in cor_list], fontsize=18)
        ax2.set_xlabel(r'\textbf{Correlation Coefficient}', fontsize=24, fontweight='bold')
        ax2.set_ylim(min_estimates_corr, 3.1) 
        ax2.set_ylabel(r'\textbf{Estimates}', fontsize=24, fontweight='bold')
        ax2.tick_params(axis='y', labelsize=18)
        
        if len(cor_list) > 2:
            for i, m in zip(method_inds[0:r], methods[0:r]):
                if i <= 1 or i == 4:
                    min_mse_idx = np.argmin(mse[:, i])
                    if min_mse_idx in [0, len(cor_list)-1]:  
                        ax.annotate(f'Min MSE',
                                    xy=(cor_list[min_mse_idx], mse[min_mse_idx, i]),
                                    xytext=(5, 5), textcoords='offset points',
                                    fontsize=16, fontweight='bold',
                                    arrowprops=dict(arrowstyle='->', color='black'))
        
        for f, a in zip([fig, fig2], [ax, ax2]):
            leg = a.legend(fontsize=16, frameon=True, fancybox=True, framealpha=0.9, 
                        edgecolor='black', loc='best')
            leg.set_title('Methods', prop={'size': 18, 'weight': 'bold'})
            
            f.tight_layout()
        
        return fig, fig2

    n_experiments, supports, n_dim, cor = 20000, 40, 100, 0.0
    rec_tau_list = []
    for s, sample_size in enumerate(n_samples):
        file = glob.glob(os.path.join(multi_instance_dump_dir, 'recovered_coefficients', \
        'recovered_coefficients_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_cor_{}_seed_*'.format(sample_size,n_dim,n_experiments,supports,cor)))[0]
        rec_tau_list.append(joblib.load(file))
        
        

    fig_ci, fig2_ci, fig3_ci, fig4_ci, fig_iqr, fig2_iqr, fig3_iqr, fig4_iqr = plot_estimates_sample(np.array(rec_tau_list), methods, method_inds, N_SAMPLES_DIR)
    fig_ci.savefig(os.path.join(N_SAMPLES_DIR,'multi_instance_comparison_support_{}_n_dim_{}_n_exp_{}_ci.png'.format(supports,n_dim,n_experiments)), dpi=300)
    fig_ci.savefig(os.path.join(N_SAMPLES_DIR,'multi_instance_comparison_support_{}_n_dim_{}_n_exp_{}_ci.pdf'.format(supports,n_dim,n_experiments)), dpi=300)
    fig2_ci.savefig(os.path.join(N_SAMPLES_DIR,'boxplot_multi_instance_comparison_support_{}_n_dim_{}_n_exp_{}_ci.png'.format(supports,n_dim,n_experiments)), dpi=300)
    fig2_ci.savefig(os.path.join(N_SAMPLES_DIR,'boxplot_multi_instance_comparison_support_{}_n_dim_{}_n_exp_{}_ci.pdf'.format(supports,n_dim,n_experiments)), dpi=300)
    fig3_ci.savefig(os.path.join(N_SAMPLES_DIR,'multi_instance_bias_std_comparison_support_{}_n_dim_{}_n_exp_{}_ci.png'.format(supports,n_dim,n_experiments)), dpi=300)
    fig3_ci.savefig(os.path.join(N_SAMPLES_DIR,'multi_instance_bias_std_comparison_support_{}_n_dim_{}_n_exp_{}_ci.pdf'.format(supports,n_dim,n_experiments)), dpi=300)
    fig4_ci.savefig(os.path.join(N_SAMPLES_DIR,'multi_instance_raw_estimates_comparison_support_{}_n_dim_{}_n_exp_{}_ci.png'.format(supports,n_dim,n_experiments)), dpi=300)
    fig4_ci.savefig(os.path.join(N_SAMPLES_DIR,'multi_instance_raw_estimates_comparison_support_{}_n_dim_{}_n_exp_{}_ci.pdf'.format(supports,n_dim,n_experiments)), dpi=300)
    fig_iqr.savefig(os.path.join(N_SAMPLES_DIR,'multi_instance_comparison_support_{}_n_dim_{}_n_exp_{}_iqr.png'.format(supports,n_dim,n_experiments)), dpi=300)
    fig_iqr.savefig(os.path.join(N_SAMPLES_DIR,'multi_instance_comparison_support_{}_n_dim_{}_n_exp_{}_iqr.pdf'.format(supports,n_dim,n_experiments)), dpi=300)
    fig2_iqr.savefig(os.path.join(N_SAMPLES_DIR,'boxplot_multi_instance_comparison_support_{}_n_dim_{}_n_exp_{}_iqr.png'.format(supports,n_dim,n_experiments)), dpi=300)
    fig2_iqr.savefig(os.path.join(N_SAMPLES_DIR,'boxplot_multi_instance_comparison_support_{}_n_dim_{}_n_exp_{}_iqr.pdf'.format(supports,n_dim,n_experiments)), dpi=300)
    fig3_iqr.savefig(os.path.join(N_SAMPLES_DIR,'multi_instance_bias_std_comparison_support_{}_n_dim_{}_n_exp_{}_iqr.png'.format(supports,n_dim,n_experiments)), dpi=300)
    fig3_iqr.savefig(os.path.join(N_SAMPLES_DIR,'multi_instance_bias_std_comparison_support_{}_n_dim_{}_n_exp_{}_iqr.pdf'.format(supports,n_dim,n_experiments)), dpi=300)
    fig4_iqr.savefig(os.path.join(N_SAMPLES_DIR,'multi_instance_raw_estimates_comparison_support_{}_n_dim_{}_n_exp_{}_iqr.png'.format(supports,n_dim,n_experiments)), dpi=300)
    fig4_iqr.savefig(os.path.join(N_SAMPLES_DIR,'multi_instance_raw_estimates_comparison_support_{}_n_dim_{}_n_exp_{}_iqr.pdf'.format(supports,n_dim,n_experiments)), dpi=300)
        

    '''
    Setup for the experiments on varying support size with other hyperparameters fixed.
    '''

    
    n_experiments, samples, n_dim, cor = 1000, 10000, 1000, 0.
    n_supports = [20*i for i in range(1, 11)]
    rec_tau_list = []
    for s, support_size in enumerate(n_supports):
        file = glob.glob(os.path.join(multi_instance_dump_dir, 'recovered_coefficients', \
        'recovered_coefficients_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_cor_{}_seed_*'.format(samples,n_dim,n_experiments,support_size,cor)))[0]
        rec_tau_list.append(joblib.load(file))

    fig, fig2 = plot_estimates_support(np.array(rec_tau_list), n_supports)
    fig.savefig(os.path.join(N_SUPP_DIR, 'multi_instance_comparison_samples_{}_n_dim_{}_n_exp_{}.pdf'.format(samples,n_dim,n_experiments)), dpi=300)
    fig2.savefig(os.path.join(N_SUPP_DIR, 'boxplot_multi_instance_comparison_samples_{}_n_dim_{}_n_exp_{}.pdf'.format(samples,n_dim,n_experiments)), dpi=300)
    
    '''
    Setup for the experiments on varying correlation of eta and X with other hyperparameters fixed.
    '''

    n_experiments, samples, n_dim, n_supports = 2000, 20000, 100, 40
    cor_list = [0.02 * i for i in range(10)]
    rec_tau_list = []
    for s, cor in enumerate(cor_list):
        file = glob.glob(os.path.join(multi_instance_dump_dir, 'recovered_coefficients', \
        'recovered_coefficients_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_cor_{}_seed_*'.format(samples,n_dim,n_experiments,n_supports,cor)))[0]
        rec_tau_list.append(joblib.load(file))

    fig, fig2 = plot_estimates_correlated(np.array(rec_tau_list), cor_list)
    fig.savefig(os.path.join(N_COR_DIR, 'cor_multi_instance_comparison_samples_{}_n_dim_{}_n_exp_{}_support_{}.pdf'.format(samples,n_dim,n_experiments,n_supports)), dpi=300)
    fig2.savefig(os.path.join(N_COR_DIR, 'boxplot_cor_multi_instance_comparison_samples_{}_n_dim_{}_n_exp_{}_support_{}.pdf'.format(samples,n_dim,n_experiments,n_supports)), dpi=300)
   
    
    
if __name__=="__main__":
    main(sys.argv[1:])
