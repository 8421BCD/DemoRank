# ------------------------- for main experiment -------------------------
MODEL_NAME_OR_PATH=../../../../llm/deberta-v3-base # lr = 1e-5
DIR=dir
# OUTPUT_DIR=${DIR}/checkpoint/cross_$(date +%F-%H%M.%S)
OUTPUT_DIR=${DIR}/checkpoint/cross_msmarco
DATA_DIR=${DIR}/data/
deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 12366 train_reranker.py --deepspeed ds_config.json \
    --train_data_shards_file "/home/u20238046/workspace_lwh/project/demorank/data/for_distill/train_data_reranker_greedy/msmarco_0_100000_flan-t5-xl(all4shot_top50).jsonl" \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 32 \
    --gradient_checkpointing True \
    --kd_cont_loss_weight 0.2 \
    --seed 123 \
    --do_train \
    --fp16 \
    --train_file ../../data/for_distill/train.jsonl \
    --max_len 512 \
    --train_n_passages 50 \
    --dataloader_num_workers 1 \
    --learning_rate 1e-5 \
    --warmup_steps 400 \
    --logging_steps 50 \
    --output_dir "${OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --save_total_limit 10 \
    --save_strategy epoch \
    --num_train_epochs 2 \
    --remove_unused_columns False \
    --overwrite_output_dir \
    --disable_tqdm True \
    --report_to none "$@" \
    --use_rankloss \




############################# for comparison with supervised models #############################
# querynum=20000
# MODEL_NAME_OR_PATH=../../../../llm/deberta-v3-base # lr = 1e-5
# DIR=dir
# # OUTPUT_DIR=${DIR}/checkpoint/cross_$(date +%F-%H%M.%S)
# OUTPUT_DIR=${DIR}/checkpoint/cross_msmarco_querynum=${querynum}
# DATA_DIR=${DIR}/data/
# # MODEL_NAME_OR_PATH=/data/wenhan_liu/workspace/llm/deberta-v3-base # lr = 5e-5
# # For electra-large, learning rate > 1e-5 will lead to instability empirically
# deepspeed --include localhost:4,5,6,7 --master_port 12366 train_reranker.py --deepspeed ds_config.json \
#     --train_data_shards_file "/home/u20238046/workspace_lwh/project/demorank/data/for_distill/train_data_reranker_greedy/msmarco_flan-t5-xl_querynum=20000_checkpoint-1875(all4shot_top50).jsonl" \
#     --model_name_or_path "${MODEL_NAME_OR_PATH}" \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 32 \
#     --gradient_checkpointing True \
#     --kd_cont_loss_weight 0.2 \
#     --seed 123 \
#     --do_train \
#     --fp16 \
#     --train_file /home/u20238046/workspace_lwh/project/demorank/data/for_distill/train.jsonl \
#     --max_len 512 \
#     --train_n_passages 50 \
#     --dataloader_num_workers 1 \
#     --learning_rate 1e-5 \
#     --warmup_steps 100 \
#     --logging_steps 50 \
#     --output_dir "${OUTPUT_DIR}" \
#     --data_dir "${DATA_DIR}" \
#     --save_total_limit 10 \
#     --save_strategy epoch \
#     --num_train_epochs 3 \
#     --remove_unused_columns False \
#     --overwrite_output_dir \
#     --disable_tqdm True \
#     --report_to none "$@" \
#     --use_rankloss \

