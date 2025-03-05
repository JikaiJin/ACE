RANDOM=42
dir=$1
mkdir -p $dir
echo $dir

for support_size in $(seq 20 20 200);
do
    python3 monte_carlo_single_instance_updated.py --n_samples 10000 --n_experiments 1000 --n_dim 1000 --support_size $support_size \
            --output_dir $dir --seed $((RANDOM % 9999 + 1)) --sigma_q 3 --cor 0.0
done
 


mkdir -p figures
mkdir -p figures/single_instance_updated
python3 plot_dumps_single_instance.py --input_dir $dir --output_dir figures/single_instance_updated

echo "Saved figures in figures/single_instance_updated!" 