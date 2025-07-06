RANDOM=42
dir=$1
mkdir -p $dir
echo $dir

for n_samples in $(seq 500 500 8000);
do
    python3 monte_carlo_single_instance_updated.py --n_samples $n_samples --n_experiments 20000 --n_dim 100 --support_size 40 \
            --output_dir $dir --seed $((RANDOM % 9999 + 1)) --sigma_q 3 --cor 0.0
done


mkdir -p figures
mkdir -p figures/single_instance_updated
python3 plot_dumps_single_instance.py --input_dir $dir --output_dir figures/single_instance_updated

echo "Saved figures in figures/single_instance_updated!" 