import sys
import os
import argparse
import glob
import numpy as np
import joblib
import matplotlib
import seaborn as sns
matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['text.usetex'] = True

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
    

    scaled_n_samples = np.array([2 * i for i in range(1,11)])
    scale = 1000
    n_samples = scale * scaled_n_samples
    methods = ['1st order', '2nd order', '3rd order'] + [str(i) + 'th order' for i in range(4,11)]
    method_inds = list(range(10))

    def collect_results(rec_tau_list):
        mse, bias, std = [], [], []
        for rec_tau in rec_tau_list:
            _bias, _std = np.mean(rec_tau, axis=0) - 3, np.std(rec_tau, axis=0)
            mse.append(_bias**2 + _std**2)
            bias.append(_bias)
            std.append(_std)
        return np.array(mse), np.array(bias), np.array(std)

    
    def plot_estimates_sample(rec_tau_list, scaled_n_samples, method_inds=method_inds, methods=methods, scale=1000, r=6):
        # Get results data
        mse, bias, std = collect_results(rec_tau_list)
        
        # Set a clean, professional style with larger default font size
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 14
        
        # Define a colorblind-friendly color palette (will use with tab10 indices)
        colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860', '#DA8BC3']
        
        # Create figures with better dimensions
        fig, ax = plt.subplots(figsize=(8, 5))
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        
        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.4)
        ax2.grid(True, linestyle='--', alpha=0.4)
        ax3.grid(True, linestyle='--', alpha=0.4)
        
        # Add reference line with improved visibility
        ax2.plot([-0.5, 9.5], [3., 3.], linestyle='--', color='#1A5276', 
                linewidth=2.0, label='True Effect', alpha=0.8)
        
        # Plot data with improved styling
        cmap = plt.get_cmap("tab10")
        for i, m in zip(method_inds[0:r], methods[0:r]):
            if i in [0, 1, 4]:
                _mse, _bias, _std = mse[:, i], np.abs(bias)[:, i], std[:, i]
                print("bias: ", _bias, " std: ", _std)
                
                # MSE plot with enhanced markers
                ax.plot(scaled_n_samples, _mse, marker='*', markersize=12, 
                        markeredgecolor='black', markeredgewidth=0.8,
                        linestyle='-', linewidth=2.0, color=cmap(i), label=m)
                
                # Bias and std plot with enhanced markers
                ax3.plot(scaled_n_samples, _bias, marker='*', markersize=12, 
                        markeredgecolor='black', markeredgewidth=0.8,
                        linestyle='-', linewidth=2.0, color=cmap(i), label=m+' bias')
                ax3.plot(scaled_n_samples, _std, marker='o', markersize=10, 
                        markeredgecolor='black', markeredgewidth=0.8,
                        linestyle='--', linewidth=2.0, color=cmap(i), label=m+' std')
            
            # We compare the results for the 1st and 5th order ACE estimators
            if i == 0 or i == 4:
                # Enhanced boxplot styling
                flierprops = dict(marker='o', markerfacecolor=cmap(i), markersize=7,
                                markeredgecolor='black', markeredgewidth=0.8,
                                linestyle='none')
                
                box = ax2.boxplot(rec_tau_list[:,:,i].T, whis=[1, 99], patch_artist=True, 
                                positions=list(range(scaled_n_samples.shape[0])),
                                medianprops=dict(color=cmap(i), linewidth=2.0), 
                                boxprops=dict(facecolor='white', color=cmap(i), alpha=0.5),
                                whiskerprops=dict(color=cmap(i), linewidth=1.5),
                                capprops=dict(color=cmap(i), linewidth=1.5),
                                showfliers=True, flierprops=flierprops)
                
                ax2.plot([], [], color=cmap(i), linewidth=2.0, label=m)
        
        ax.set_xlabel(r'Sample Size ($\times 10^3$)', fontsize=16, fontweight='bold')
        ax.set_ylabel("Mean-Squared Error", fontsize=16, fontweight='bold')
        ax.set_yscale("log")
        ax.set_ylim(6e-4, 2e-2)
        
        ax.set_xticks(scaled_n_samples)
        ax.set_xticklabels(scaled_n_samples, fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        
        ax2.set_xlabel(r'Sample Size ($\times 10^3$)', fontsize=16, fontweight='bold')
        ax2.set_xticks(list(range(scaled_n_samples.shape[0])))
        ax2.set_xticklabels(scaled_n_samples, fontsize=14)
        ax2.set_ylim(2.85, 3.15)
        ax2.set_ylabel("Estimates", fontsize=16, fontweight='bold')
        ax2.tick_params(axis='y', labelsize=14)
        
        ax3.set_xlabel(r'Sample Size ($\times 10^3$)', fontsize=16, fontweight='bold')
        ax3.set_ylabel("Error Metrics", fontsize=16, fontweight='bold')
        ax3.set_yscale("log")
        ax3.set_ylim(5e-6, 2e-2)
        ax3.set_xticks(scaled_n_samples)
        ax3.set_xticklabels(scaled_n_samples, fontsize=14)
        ax3.tick_params(axis='y', labelsize=14)
        
        for f, a in zip([fig, fig2, fig3], [ax, ax2, ax3]):
            leg = a.legend(fontsize=12, frameon=True, fancybox=True, framealpha=0.9, 
                        edgecolor='black', loc='best')
            leg.set_title('Methods', prop={'size': 14, 'weight': 'bold'})
            
            f.tight_layout()
        
        return fig, fig2, fig3
    
    
    def plot_estimates_support(rec_tau_list, n_supports, method_inds=method_inds, methods=methods, r=6):
        # Get results data
        mse, bias, std = collect_results(rec_tau_list)
        
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 14
        
        fig, ax = plt.subplots(figsize=(8, 5))
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        
        ax.grid(True, linestyle='--', alpha=0.4)
        ax2.grid(True, linestyle='--', alpha=0.4)
        
        ax2.plot([-0.5, len(n_supports) - 0.5], [3., 3.], linestyle='--', color='#1A5276', 
                linewidth=2.0, label='True Effect', alpha=0.8)
        
        cmap = plt.get_cmap("tab10")
        for i, m in zip(method_inds[0:r], methods[0:r]):
            if i in [0, 1, 4]:
                _mse, _bias, _std = mse[:, i], np.abs(bias)[:, i], std[:, i]
                
                ax.plot(n_supports, _mse, marker='*', markersize=12, 
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
        
        ax.set_ylim(1e-3, 2e-2)
        ax.set_yscale("log")
        ax.set_xlabel('Support Size', fontsize=16, fontweight='bold')
        ax.set_ylabel("Mean-Squared Error", fontsize=16, fontweight='bold')
        
        ax.set_xticks(n_supports)
        ax.set_xticklabels(n_supports, fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        
        ax2.set_xlabel('Support Size', fontsize=16, fontweight='bold')
        ax2.set_xticks(list(range(len(n_supports))))
        ax2.set_xticklabels(n_supports, fontsize=14)
        ax2.set_ylim(2.85, 3.15)
        ax2.set_ylabel("Estimates", fontsize=16, fontweight='bold')
        ax2.tick_params(axis='y', labelsize=14)
        
        min_mse_idx = np.argmin(mse[:, method_inds[0]])
        ax.annotate(f'Min MSE: {mse[min_mse_idx, method_inds[0]]:.4f}',
                    xy=(n_supports[min_mse_idx], mse[min_mse_idx, method_inds[0]]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=12, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='black'))
        
        for f, a in zip([fig, fig2], [ax, ax2]):
            leg = a.legend(fontsize=12, frameon=True, fancybox=True, framealpha=0.9, 
                        edgecolor='black', loc='best')
            leg.set_title('Methods', prop={'size': 14, 'weight': 'bold'})
            
            f.tight_layout()
        
        return fig, fig2
    
    # Compare the performance of ACE estimators in the case when eta and X are correlated.
    
    def plot_estimates_correlated(rec_tau_list, cor_list, method_inds=method_inds, methods=methods, r=6):
        
        mse, bias, std = collect_results(rec_tau_list)
        
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 14
        
        fig, ax = plt.subplots(figsize=(8, 5))
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        
        ax.grid(True, linestyle='--', alpha=0.4)
        ax2.grid(True, linestyle='--', alpha=0.4)
        
        ax2.plot([-0.5, 9.5], [3., 3.], linestyle='--', color='#1A5276', 
                linewidth=2.0, label='True Effect', alpha=0.8)
        
        cmap = plt.get_cmap("tab10")
        for i, m in zip(method_inds[0:r], methods[0:r]):
            if i <= 1 or i == 4:
                _mse, _bias, _std = mse[:, i], np.abs(bias)[:, i], std[:, i]
                
                ax.plot(cor_list, _mse, marker='*', markersize=12, 
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
        
        ax.set_yscale("log")
        ax.set_xlabel('Correlation Coefficient', fontsize=16, fontweight='bold')
        ax.set_ylabel("Mean-Squared Error", fontsize=16, fontweight='bold')
        ax.set_ylim(6e-4, 2e-3)
        
        ax.set_xticks(cor_list)
        ax.set_xticklabels([f"{c:.1f}" for c in cor_list], fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        
        ax2.set_xticks(list(range(len(cor_list))))
        ax2.set_xticklabels([f"{c:.1f}" for c in cor_list], fontsize=14)
        ax2.set_xlabel('Correlation Coefficient', fontsize=16, fontweight='bold')
        ax2.set_ylim(2.9, 3.1)
        ax2.set_ylabel("Estimates", fontsize=16, fontweight='bold')
        ax2.tick_params(axis='y', labelsize=14)
        
        # Highlight the trend
        if len(cor_list) > 2:
            # Find min and max MSE for each method to show trend
            for i, m in zip(method_inds[0:r], methods[0:r]):
                if i <= 1 or i == 4:
                    min_mse_idx = np.argmin(mse[:, i])
                    if min_mse_idx in [0, len(cor_list)-1]:  # Only annotate if at an endpoint
                        ax.annotate(f'Min MSE',
                                    xy=(cor_list[min_mse_idx], mse[min_mse_idx, i]),
                                    xytext=(5, 5), textcoords='offset points',
                                    fontsize=11, fontweight='bold',
                                    arrowprops=dict(arrowstyle='->', color='black'))
        
        for f, a in zip([fig, fig2], [ax, ax2]):
            leg = a.legend(fontsize=12, frameon=True, fancybox=True, framealpha=0.9, 
                        edgecolor='black', loc='best')
            leg.set_title('Methods', prop={'size': 14, 'weight': 'bold'})
            
            f.tight_layout()
        
        return fig, fig2

    '''
    Setup for the experiments on varying sample size with other hyperparameters fixed.
    '''
    
    n_experiments, supports, n_dim, cor = 2000, 40, 100, 0.0
    rec_tau_list = []
    for s, sample_size in enumerate(n_samples):
        file = glob.glob(os.path.join(multi_instance_dump_dir, 'recovered_coefficients', \
        'recovered_coefficients_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_cor_{}_seed_*'.format(sample_size,n_dim,n_experiments,supports,cor)))[0]
        rec_tau_list.append(joblib.load(file))
        
        

    fig, fig2, fig3 = plot_estimates_sample(np.array(rec_tau_list), scaled_n_samples, method_inds, methods)
    fig.savefig(os.path.join(N_SAMPLES_DIR,'multi_instance_comparison_support_{}_n_dim_{}_n_exp_{}.png'.format(supports,n_dim,n_experiments)), dpi=300)
    fig.savefig(os.path.join(N_SAMPLES_DIR,'multi_instance_comparison_support_{}_n_dim_{}_n_exp_{}.pdf'.format(supports,n_dim,n_experiments)), dpi=300)
    fig2.savefig(os.path.join(N_SAMPLES_DIR,'boxplot_multi_instance_comparison_support_{}_n_dim_{}_n_exp_{}.png'.format(supports,n_dim,n_experiments)), dpi=300)
    fig2.savefig(os.path.join(N_SAMPLES_DIR,'boxplot_multi_instance_comparison_support_{}_n_dim_{}_n_exp_{}.pdf'.format(supports,n_dim,n_experiments)), dpi=300)
    fig3.savefig(os.path.join(N_SAMPLES_DIR,'multi_instance_bias_std_comparison_support_{}_n_dim_{}_n_exp_{}.png'.format(supports,n_dim,n_experiments)), dpi=300)
    fig3.savefig(os.path.join(N_SAMPLES_DIR,'multi_instance_bias_std_comparison_support_{}_n_dim_{}_n_exp_{}.pdf'.format(supports,n_dim,n_experiments)), dpi=300)
        

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