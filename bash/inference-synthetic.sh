CUDA=0
data_file="data/data_csv/inference_synthetic.csv"  # TODO: download data
checkpoint_path="checkpoints/synthetic/auprc=0.905_migrated.ckpt"
results_prefix="output/synthetic/results"
sample_size=1000

python src/inference.py \
    --config_file config/inference.yaml \
    --gpu $CUDA \
    --data_file $data_file \
    --sample_size $sample_size \
    --batch_size 1 \
    --results_prefix $results_prefix \
    --checkpoint_path $checkpoint_path \
    # --break_cycles  # optional: post-process predictions into a DAG
