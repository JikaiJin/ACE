# Structure-agnostic Cumulant-based estimation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Implements the agnostic cumulant-based estimator (ACE) as a generalized and more efficient version of double/debiased and orthogonal machine learning.

## Features

- A synthetic experiment setting to simulatent a price experiment, motivated by prior work https://arxiv.org/pdf/1711.00342.
- Experiments with varying sample sizes, support sizes, and noise correlations. These experients test the validity of our approach as well as its sensitivity to violation of the assumptions.
- Visualization for both single and multiple experiment instances.

## Installation

```bash
# Clone the repository
git clone https://github.com/JikaiJin/its_hard_to_be_normal.git
cd its_hard_to_be_normal
```

## Usage

### Single Instance Experiments

```bash
# Run experiments with different sample sizes
./single_instance_sample_size.sh single_instance_dir

# Run experiments with different support sizes
./single_instance_support_size.sh single_instance_dir

# Run experiments with different noise correlations
./single_instance_noise_correlation.sh single_instance_dir
```

### Multi-Instance Experiments

```bash
# Run multiple instances
./multi_instance.sh multi_instance_dir
```


## File Descriptions

### Core Modules

- **construct_moment.py**: Contain functions that will be used in our definition of the moment function, which involves cumulants.
- **main_estimation.py**: Obtain the desired parameter estimate from first-stage nuisance estimate.

### Single Instance Experiments

- **monte_carlo_single_instance_updated.py**: Set up the synthetic experiment that we will use to test our method, and plot the results for a single instance. Also store the relevant results that will be used to generate the multi-instance plots.
- **single_instance_sample_size.sh**: Test our methods on single instance with varying sample sizes.
- **single_instance_support_size.sh**: Test our methods on single instance with varying support sizes.
- **single_instance_noise_correlation.sh**: Test our methods on single instance with varying levels of correlation between the treatment noise $\eta$ and covariate $X$.

### Multi-Instance Experiments

- **plot_multi_instance.py**: Compare performance of different methods across different sample sizes, support sizes and noise correlation levels.
- **multi_instance.sh**: Execute the **plot_multi_instance.py** file and output plots for multi-instances.



## License

This project is licensed under the MIT License - see the [LICENSE](https://opensource.org/licenses/MIT) file for details.