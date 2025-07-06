#!/bin/bash
dir=$1
mkdir -p $dir
echo $dir

# Ensure proper environment for matplotlib
export MPLBACKEND="Agg"
# Ensure LaTeX is available for text rendering
command -v latex >/dev/null 2>&1 || { echo "LaTeX is required but not installed. Please install TeX Live or similar."; exit 1; }

mkdir -p multi_instance
python3 plot_multi_instance.py --input_dir single_instance_dir --output_dir multi_instance

echo "Saved figures in multi_instance!" 