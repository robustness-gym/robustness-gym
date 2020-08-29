export TRAIN_FILE=/export/home/Data/WikiText-2/wikitext-2-raw/wiki.train.raw
export TEST_FILE=/export/home/Data/WikiText-2/wikitext-2-raw/wiki.valid.raw
export MODEL_PATH=/export/home/Experiments/20200615/output-gpt2-tiny/

python ./language_modeling.py \
    --output_dir=output-gpt2-tiny \
    --model_type=gpt2 \
    --model_name_or_path=${MODEL_PATH} \
    --do_train \
    --train_data_file=${TRAIN_FILE} \
    --do_eval \
    --eval_data_file=${TEST_FILE} \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --num_train_epochs 30 \
    --save_total_limit 1
