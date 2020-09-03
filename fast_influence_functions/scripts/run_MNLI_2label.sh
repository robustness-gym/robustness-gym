export GLUE_DIR=/export/home/Data/Glue

python -m torch.distributed.launch \
    --nproc_per_node 4 run_glue.py \
    --model_name_or_path bert-base-cased \
    --task_name mnli-2 \
    --do_train \
    --do_eval \
    --data_dir $GLUE_DIR/MNLI/ \
    --max_seq_length 128 \
    --per_device_train_batch_size 128 \
    --learning_rate 2e-5 \
    --num_train_epochs 10.0 \
    --output_dir output_dir \
    --weight_decay 0.005 \
    --save_steps 5000 \
    --logging_steps 100 \
    --save_total_limit 1
