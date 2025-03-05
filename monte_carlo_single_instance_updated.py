import sys
import os
import matplotlib
matplotlib.use('Agg')
import numpy as np
from sklearn.linear_model import LassoCV, Lasso
from main_estimation import all_together_cross_fitting
from construct_moment import kstat_pp, kstat
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
import seaborn as sns
from scipy.stats import norm
import joblib
import argparse
from joblib import delayed, Parallel


def experiment(x, eta, epsilon, tau, s_p, a_p, s_q, a_q, lambda_reg = 0, r = 10, cor = 0.08):
    # Generate price as a function of co-variates
    p = np.dot(x[:, s_p], a_p) + eta + cor * np.multiply(x[:, s_p[0]], eta)
    # Generate demand as a function of price and co-variates
    q = tau * p + np.dot(x[:, s_q], a_q) + epsilon
    model_p = Lasso(alpha=lambda_reg) 
    model_q = Lasso(alpha=lambda_reg)
    return all_together_cross_fitting(x, p, q, r, model_p = model_p, model_q = model_q)

def main(args):
    os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
    parser = argparse.ArgumentParser(
        description="Second order orthogonal ML!")
    parser.add_argument("--n_samples", dest="n_samples",
                        type=int, help='n_samples', default=5000)
    parser.add_argument("--n_dim", dest="n_dim",
                        type=int, help='n_dim', default=1000)
    parser.add_argument("--n_experiments", dest="n_experiments",
                        type=int, help='n_experiments', default=2000)
    parser.add_argument("--support_size", dest="support_size",
                        type=int, help='support_size', default=400)
    parser.add_argument("--seed", dest="seed",
                        type=int, help='seed', default=2025)
    parser.add_argument("--sigma_q", dest="sigma_q",
                        type=int, help='sigma_q', default=12143)
    parser.add_argument("--cor", dest="cor",
                        type=float, help='cor', default=0.1)
    parser.add_argument("--output_dir", dest="output_dir", type=str, default=".")
    opts = parser.parse_args(args)
    sigma_q = opts.sigma_q

    np.random.seed(42)

    ERR_DIR = os.path.join(opts.output_dir, 'model_errors')
    RESULT_DIR = os.path.join(opts.output_dir, 'recovered_coefficients')
    VAR_EST_DIR = os.path.join(opts.output_dir, 'var_estimates')

    if not os.path.exists(ERR_DIR):
        os.makedirs(ERR_DIR)
        os.makedirs(RESULT_DIR)
        os.makedirs(VAR_EST_DIR)
        
    # We will work with a sparse linear model with high dimensional co-variates
    
    # Number of (price, demand, co-variate) samples
    n_samples = opts.n_samples
    # Dimension of co-variates
    n_dim = opts.n_dim
    # How many experiments to run to see the distribution of the recovered coefficient between price and demand
    n_experiments = opts.n_experiments
    support_size = opts.support_size 
    cor = opts.cor
    print("Sample size: {}".format(n_samples))
    print("Dimension of covariate: {}".format(n_dim))
    print("Support size of sparse functions: {}".format(support_size))
    print("Correlation coefficient of noise: {}".format(cor))

    # ------------ True parameters -------------

    # Support and coefficients for treatment as function of co-variates
    s_p = np.random.choice(range(n_dim), size=support_size, replace=False)
    a_p = np.random.uniform(0, 5, size=support_size)
    

    # Support and coefficients for outcome as function of co-variates
    s_q = s_p 
    a_q = np.random.uniform(0, 5, size=support_size)
    
    methods = ['1st order', '2nd order', '3rd order'] + [str(i) + 'th order' for i in range(4,11)]

    np.random.seed(opts.seed)

    # Distribution of residuals of treatment
    discounts = np.array([0, -.5, -2., -4.])
    probs = np.array([.65, .2, .1, .05])
    
    mean_discount = np.dot(discounts, probs)
    eta_sample = lambda x: np.array([discounts[i] - mean_discount for i in np.argmax(np.random.multinomial(1, probs, x), axis=1)])
    # Calculate moments of the residual distribution
    moments_p = [np.dot(probs, (discounts - mean_discount)**k) for k in range(8)]
    cums_p = np.array(kstat_pp(moments_p, 7))

    print("Cumulants of eta: ", cums_p)
    
    

    # Distribution of outcome residuals
    epsilon_sample = lambda x: np.random.uniform(-sigma_q, sigma_q, size=x)

    # Treatment effect
    tau = 3.0

    true_coef_p = np.zeros(n_dim)
    true_coef_p[s_p] = a_p
    true_coef_q = np.zeros(n_dim)
    true_coef_q[s_q] = a_q
    true_coef_q[s_p] += tau * a_p
    
    # ------------- Run the experiments ----------------
    
    # Coefficients recovered by orthogonal ML
    lambda_reg = np.sqrt(np.log(n_dim)/(n_samples))
    results = Parallel(n_jobs=-1, verbose=1)(delayed(experiment)(
                                            np.random.normal(size=(n_samples, n_dim)),
                                            eta_sample(n_samples),
                                            epsilon_sample(n_samples),
                                            tau, s_p, a_p, s_q, a_q, lambda_reg = lambda_reg,
                                            cor = cor
                                            ) for t in range(n_experiments))
    rec_tau = np.array([tau for tau, _, _, _, _ in results])
    var = np.array([v for _, v, _, _, _ in results])
    first_stage_mse = np.array([[np.linalg.norm(true_coef_p - coef_p), \
                                 np.linalg.norm(true_coef_q - coef_q)] for _, _, coef_p, coef_q, _ in results])
    cums_est = np.array([c for _, _, _, _, c in results])
    print("Estimates of cumulants: ", cums_est[0, 0:6])



    joblib.dump(rec_tau, os.path.join(RESULT_DIR,'recovered_coefficients_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_cor_{}_seed_{}.jbl'.format(n_samples,n_dim,n_experiments,support_size,cor, opts.seed)))
    joblib.dump(first_stage_mse, os.path.join(RESULT_DIR,'model_errors_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_cor_{}_seed_{}.jbl'.format(n_samples,n_dim,n_experiments,support_size,cor, opts.seed)))
    print("Done with experiments!")


    # ------------- Plot the figures --------------

    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 14  # Increased base font size

    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']

    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.title("Model_p Error Distribution", fontsize=18, fontweight='bold', pad=15)
    n, bins, patches = plt.hist(np.array(first_stage_mse)[:,0].flatten(), 10, density=True,
                            color=colors[0], edgecolor='black', linewidth=1.2, alpha=0.7)
    mean_p = np.mean(np.array(first_stage_mse)[:,0].flatten())
    plt.axvline(mean_p, color=colors[4], linestyle='--', linewidth=2, 
                label=f'Mean: {mean_p:.4f}')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.xlabel("Error Value", fontsize=16, fontweight='bold')
    plt.ylabel("Density", fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, frameon=True, framealpha=0.9, edgecolor='black')
    plt.tick_params(axis='both', which='major', labelsize=14)

    plt.subplot(1, 2, 2)
    plt.title("Model_q Error Distribution", fontsize=18, fontweight='bold', pad=15)
    n, bins, patches = plt.hist(np.array(first_stage_mse)[:,1].flatten(), 10, density=True,
                            color=colors[1], edgecolor='black', linewidth=1.2, alpha=0.7)
    mean_q = np.mean(np.array(first_stage_mse)[:,1].flatten())
    plt.axvline(mean_q, color=colors[4], linestyle='--', linewidth=2, 
                label=f'Mean: {mean_q:.4f}')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.xlabel("Error Value", fontsize=16, fontweight='bold')
    plt.ylabel("Density", fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, frameon=True, framealpha=0.9, edgecolor='black')
    plt.tick_params(axis='both', which='major', labelsize=14)

    plt.suptitle(f"Model Error Distributions (n={n_samples}, dim={n_dim}, support={support_size})", 
                fontsize=20, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for suptitle

    plt.savefig(os.path.join(ERR_DIR, 'model_errors_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_sigma_q_{}_cor_{}.png'.format(
        n_samples, n_dim, n_experiments, support_size, sigma_q, cor)), dpi=300, bbox_inches='tight')    
    plt.savefig(os.path.join(ERR_DIR, 'model_errors_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_sigma_q_{}_cor_{}.pdf'.format(
        n_samples, n_dim, n_experiments, support_size, sigma_q, cor)), dpi=300, bbox_inches='tight')


    pred_err = np.zeros((6, n_experiments))
    for s in range(6):
        # predicted variance for order s+1
        pred_err[s] = np.sqrt(var[:,s]/n_samples) / np.abs(cums_est[:,s+2]) * 2


    # --------------- boxplot for the estimate distribution for varying sample size -------------

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 14

    cmap = plt.get_cmap("tab10")
    flierprops = dict(marker='o', markerfacecolor=cmap(0), markersize=7,
                    markeredgecolor='black', markeredgewidth=0.8,
                    linestyle='none')

    box = ax.boxplot(pred_err.T, whis=[1, 99], patch_artist=True, positions=list(range(6)),
                    medianprops=dict(color=cmap(0), linewidth=2.0),
                    boxprops=dict(facecolor='white', color=cmap(0), alpha=0.5),
                    whiskerprops=dict(color=cmap(0), linewidth=1.5),
                    capprops=dict(color=cmap(0), linewidth=1.5),
                    showfliers=True, flierprops=flierprops)

    # Add grid and styling
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.grid(True, which='minor', linestyle=':', alpha=0.2)

    # Enhanced labels and title
    ax.set_xlabel("Order of Orthogonality", fontsize=16, fontweight='bold')
    ax.set_ylabel("Prediction Error (Log Scale)", fontsize=16, fontweight='bold')
    ax.set_title("Prediction Error by Orthogonality Order", fontsize=18, fontweight='bold', pad=15)

    # Improved axis ticks
    ax.set_xticks(list(range(6)))
    ax.set_xticklabels(list(range(1, 7)), fontsize=14)
    ax.tick_params(axis='y', labelsize=14)

    # Set scale and limits as in original
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 1.)

    # Add summary statistics text box
    median_values = np.median(pred_err, axis=1)
    stats_text = f"Summary: Orders 1-3 median = {np.mean(median_values[:3]):.3f}, Orders 4-6 median = {np.mean(median_values[3:]):.3f}"
    ax.text(0.5, 0.02, stats_text, transform=ax.transAxes, ha='center',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8, edgecolor='lightgray'),
        fontsize=12)

    # Adjust layout
    fig.tight_layout()

    # Save the figure with high resolution
    fig.savefig(os.path.join(VAR_EST_DIR, 'var_estimate_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_sigma_q_{}_cor_{}.pdf'.format(
        n_samples, n_dim, n_experiments, support_size, sigma_q, cor)), dpi=300, bbox_inches='tight')



    def plot_estimates(ax, estimate_list, true_tau, plot=True, title="Histogram of estimates"):
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 14
        
        sigma = np.std(estimate_list)
        mu = np.mean(estimate_list)
        bias = np.abs(true_tau - mu)
        mse = bias ** 2 + sigma ** 2
        
        n, bins, patches = ax.hist(estimate_list, 40, density=True, 
                                color='#55A868', edgecolor='black', 
                                linewidth=0.8, alpha=0.7)
        
        y = norm.pdf(bins, mu, sigma)
        l = ax.plot(bins, y, color='#DD8452', linestyle='--', 
                    linewidth=2.5, label='Normal Distribution')
        
        ax.plot([true_tau, true_tau], [0, np.max(y) * 1.1], 
                color='#4C72B0', linestyle='--', linewidth=2.5, 
                label='True Effect')
        
        ax.plot([mu, mu], [0, np.max(y) * 0.9], 
                color='#C44E52', linestyle='-', linewidth=2.5, 
                label='Mean Estimate')
        
        ax.grid(True, linestyle='--', alpha=0.4)
        
        ax.set_title(f"{title}\nBias: {bias:.4f}, Std: {sigma:.4f}, MSE: {mse:.4f}", 
                    fontsize=14, fontweight='bold', pad=15)
        
        ax.set_xlabel("Estimate Value", fontsize=16, fontweight='bold')
        ax.set_ylabel("Probability Density", fontsize=16, fontweight='bold')
        
        ax.legend(fontsize=12, frameon=True, fancybox=True, framealpha=0.9, 
                edgecolor='black', loc='best')
        
        ax.tick_params(axis='both', labelsize=12)
        
        stats_text = f"μ = {mu:.4f}, σ = {sigma:.4f}"
        ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, 
            ha='right', va='top', fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", 
                        alpha=0.8, edgecolor='lightgray'))
        
        return bias, sigma
        
    def plot_confidence_intervals(intervals, true_parameter, max_display=2000):
        
        """
        Visualize confidence intervals and their coverage of the true parameter.
        
        Parameters:
        intervals: list of tuples (lower_bound, upper_bound)
        true_parameter: float, the true parameter value
        max_display: int, maximum number of intervals to display
        """
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 14
        
        intervals = np.array(intervals)
        
        contains_true = (intervals[:, 0] <= true_parameter) & (intervals[:, 1] >= true_parameter)
        
        coverage = np.mean(contains_true) * 100
        
        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.grid(True, linestyle='--', alpha=0.3)
        
        n_intervals = min(len(intervals), max_display)
        y_positions = np.arange(n_intervals)
        
        segments = np.array([[[lower, y], [upper, y]] 
                            for (lower, upper), y in zip(intervals[:n_intervals], y_positions)])
        
        colors = ['#55A868' if contains else '#C44E52' for contains in contains_true[:n_intervals]]
        
        lc = LineCollection(segments, colors=colors, linewidths=2.5, capstyle='round')
        ax.add_collection(lc)
        
        ax.axvline(x=true_parameter, color='#4C72B0', linestyle='--', 
                linewidth=3, label='True Parameter')
        
        interval_widths = intervals[:, 1] - intervals[:, 0]
        avg_width = np.mean(interval_widths)
        
        margin = avg_width * 0.5  # Add some margin around intervals
        x_min = max(min(intervals[:, 0]) - margin, true_parameter - avg_width * 2)
        x_max = min(max(intervals[:, 1]) + margin, true_parameter + avg_width * 2)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-1, n_intervals)
        
        ax.set_xlabel('Parameter Value', fontsize=16, fontweight='bold')
        ax.set_ylabel('Experiment Number', fontsize=16, fontweight='bold')
        
        coverage_diff = coverage - 95.0
        coverage_text = f"Coverage: {coverage:.1f}% (Expected: 95.0%)"
        if abs(coverage_diff) > 2:
            coverage_text += f", Difference: {coverage_diff:+.1f}%"
        
        plt.title(coverage_text, fontsize=18, fontweight='bold', pad=15)
        
        # Add legend with improved styling

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#55A868', edgecolor='black', label='Contains True Parameter'),
            Patch(facecolor='#C44E52', edgecolor='black', label='Does Not Contain True Parameter'),
            Patch(facecolor='#4C72B0', edgecolor='black', label='True Parameter')
        ]
        ax.legend(handles=legend_elements, fontsize=12, frameon=True, 
                fancybox=True, framealpha=0.9, edgecolor='black')
        
        # Summary information text box
        summary_text = (
            f"Total Intervals: {len(intervals)}\n"
            f"Showing: {n_intervals}\n"
            f"Avg. Width: {avg_width:.4f}\n"
            f"Actual Coverage: {coverage:.1f}%"
        )
        
        plt.figtext(0.02, 0.02, summary_text, fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", 
                            alpha=0.8, edgecolor='lightgray'))
        
        ax.tick_params(axis='both', labelsize=12)
        
        if n_intervals > 20:
            step = max(1, n_intervals // 10)
            ax.set_yticks(np.arange(0, n_intervals, step))
        
        plt.tight_layout()
        
        return fig


    for i in range(10):
        fig, ax = plt.subplots(figsize=(8, 5))
        bias_ortho, sigma_ortho = plot_estimates(ax, rec_tau[:,i].flatten(), tau, title= methods[i] + " ACE")
        print(f"Ortho ML MSE: with order {i}: {bias_ortho**2 + sigma_ortho**2}")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULT_DIR, 'recovered_coefficients_from_order_{}_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_sigma_q_{}_cor_{}.png'.format(i+1,n_samples,n_dim,n_experiments,support_size,sigma_q,cor)), dpi=300, bbox_inches='tight')    
        fig.savefig(os.path.join(RESULT_DIR, 'recovered_coefficients_from_order_{}_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_sigma_q_{}_cor_{}.pdf'.format(i+1,n_samples,n_dim,n_experiments,support_size,sigma_q,cor)), dpi=300, bbox_inches='tight')

        if i < 6:
            margins = 1.96 * pred_err[i]
            intervals = np.column_stack([rec_tau[:,i] - margins, rec_tau[:,i] + margins])
            fig = plot_confidence_intervals(intervals, tau)
            fig.savefig(os.path.join(RESULT_DIR, 'CIs_from_order_{}_n_samples_{}_n_dim_{}_n_exp_{}_support_{}_sigma_q_{}_cor_{}.pdf'.format(i+1,n_samples,n_dim,n_experiments,support_size,sigma_q,cor)), dpi=300, bbox_inches='tight')
        
from multiprocessing import Pool

if __name__=="__main__":
    main(sys.argv[1:])