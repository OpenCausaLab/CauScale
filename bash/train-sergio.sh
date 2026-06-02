CUDA=0
NUM_GPU=8
data_file="data/data_csv/train-sergio.csv"  # TODO: generate data
inference_data_file="data/data_csv/inference_sergio.csv"  # TODO: download data
batch_size=1
sample_size=5000

exp_name="sergio_samp${sample_size}_bs${batch_size}_gpun${NUM_GPU}"
results_prefix="output/sergio/${exp_name}"

python src/train.py \
    --config_file config/train.yaml \
    --gpu $CUDA \
    --num_gpu $NUM_GPU \
    --data_file "$data_file" \
    --inference_data_file "$inference_data_file" \
    --results_prefix "$results_prefix" \
    --output_name "$exp_name" \
    --batch_size $batch_size \
    --sample_size $sample_size \
    --limit_train_batches 1000
