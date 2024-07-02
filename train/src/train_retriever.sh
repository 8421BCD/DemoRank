# ------------------------- for main experiment -------------------------
MODEL_NAME_OR_PATH="../../../../llm/e5-base-v2"
DIR=dir
# OUTPUT_DIR=${DIR}/checkpoint/kd_$(date +%F-%H%M.%S)
OUTPUT_DIR=${DIR}/checkpoint/msmarco_qd
DATA_DIR=${DIR}/data/

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 12325 train_retriever.py --deepspeed ds_config.json \
    --train_data_shards_file /home/u20238046/workspace_lwh/project/demorank/data/for_distill/train_data_retriever/msmarco_0_200000_flan-t5-xl_1shot.jsonl \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 32 \
    --gradient_checkpointing True \
    --kd_cont_loss_weight 0.2 \
    --l2_normalize True --t 0.01 \
    --pool_type avg \
    --seed 123 \
    --do_train \
    --fp16 \
    --train_file ../../data/for_distill/train.jsonl \
    --max_len 512 \
    --train_n_passages 50 \
    --dataloader_num_workers 1 \
    --learning_rate 3e-5 \
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

cd ../../build_index/
bash build_dense_index_after_train.sh



# ------------------------- for comparison with supervised models -------------------------
# querynum=20000
# MODEL_NAME_OR_PATH="../../../../llm/e5-base-v2"
# DIR=dir
# # OUTPUT_DIR=${DIR}/checkpoint/kd_$(date +%F-%H%M.%S)
# # OUTPUT_DIR="${DIR}/checkpoint/msmarco_qd"
# OUTPUT_DIR="${DIR}/checkpoint/msmarco_qd_querynum=${querynum}"
# DATA_DIR=${DIR}/data/

# deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 12325 train_retriever.py --deepspeed ds_config.json \
#     --model_name_or_path "${MODEL_NAME_OR_PATH}" \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 32 \
#     --gradient_checkpointing True \
#     --kd_cont_loss_weight 0.2 \
#     --l2_normalize True --t 0.01 \
#     --pool_type avg \
#     --seed 123 \
#     --do_train \
#     --fp16 \
#     --train_file ../../data/for_distill/train.jsonl \
#     --max_len 512 \
#     --train_n_passages 50 \
#     --dataloader_num_workers 1 \
#     --learning_rate 3e-5 \
#     --warmup_steps 100 \
#     --logging_steps 50 \
#     --output_dir "${OUTPUT_DIR}" \
#     --data_dir "${DATA_DIR}" \
#     --save_total_limit 10 \
#     --save_strategy epoch \
#     --num_train_epochs 5 \
#     --remove_unused_columns False \
#     --overwrite_output_dir \
#     --disable_tqdm True \
#     --report_to none "$@" \
#     --use_rankloss \
#     --train_data_shards_file "/home/u20238046/workspace_lwh/project/demorank/data/for_distill/train_data_retriever/msmarco_flan-t5-xl_1shot_querynum=${querynum}.jsonl" \
#     # --uni_dataset True \
#     # --uni_dataset_path /home/u20238046/workspace_lwh/project/demorank/data/for_distill/train_data_retriever_kshot \

# cd ../../build_index/
# bash build_dense_index_after_train.sh


