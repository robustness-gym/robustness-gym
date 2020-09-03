ARTICLE_START_INDEX=$1
ARTICLE_END_INDEX=$2
MODEL_PATH=/export/home/Experiments/20200615/output-gpt2-tiny/
DATA_BASE_DIR=/export/home/Data/WikiText-2/articles/
EXPERIMENT_BASE_DIR=/export/home/Experiments/20200623-sanity-check-wikitext2

echo "Training models from index ${ARTICLE_START_INDEX} to ${ARTICLE_END_INDEX}"

for (( i=$ARTICLE_START_INDEX; i<=$ARTICLE_END_INDEX; i++ ))
do
    TRAIN_FILE=${DATA_BASE_DIR}/article-no-${i}.txt
    echo "Training model on ${TRAIN_FILE}"

    python ./language_modeling.py \
        --output_dir=${EXPERIMENT_BASE_DIR}/article-no-${i} \
        --model_type=gpt2 \
        --model_name_or_path=${MODEL_PATH} \
        --do_train \
        --train_data_file=${TRAIN_FILE} \
        --per_device_train_batch_size 128 \
        --per_device_eval_batch_size 128 \
        --num_train_epochs 30 \
        --save_total_limit 1
done
