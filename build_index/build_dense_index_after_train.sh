export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ------------------------------------------ for main experiment ------------------------------------------
checkpoint_dir="/home/u20238046/workspace_lwh/project/demorank/train/src/dir/checkpoint"
NEWEST_FOLDER=$(ls -lt $checkpoint_dir | grep '^d' | head -n 1 | awk '{print $9}')
checkpoints=$(find "$checkpoint_dir/$NEWEST_FOLDER" -mindepth 1 -maxdepth 1 -type d -printf '%T+ %f\n' | sort | awk '{print $2}')
for model in $checkpoints
do
    datasets=("msmarco")
    # datasets=("msmarco" "fiqa" "nq" "scifact" "nfcorpus" "quora" "hotpotqa" "fever")
    for dataset in "${datasets[@]}"
    do
        torchrun --master_port 3943 --nproc_per_node 8 build_dense_index_after_train.py \
                --eval_batch_size 32 \
                --model_path /home/u20238046/workspace_lwh/project/demorank/train/src/dir/checkpoint/$NEWEST_FOLDER/$model \
                --output_dir /home/u20238046/workspace_lwh/project/demorank/data/for_index/query_doc/$dataset/${NEWEST_FOLDER}_${model} \
                --demonstration_pool_path /home/u20238046/workspace_lwh/project/demorank/data/for_index/query_doc/$dataset/collection/collection.jsonl \
                --dataset $dataset \
                --output_embedding_size 768 \

    done
done


# ------------------------------------------ for comparison with supervised models ------------------------------------------
# querynum=20000
# checkpoint_dir="/home/u20238046/workspace_lwh/project/demorank/train/src/dir/checkpoint"
# NEWEST_FOLDER=$(ls -lt $checkpoint_dir | grep '^d' | head -n 1 | awk '{print $9}')
# checkpoints=$(find "$checkpoint_dir/$NEWEST_FOLDER" -mindepth 1 -maxdepth 1 -type d -printf '%T+ %f\n' | sort | awk '{print $2}')
# for model in $checkpoints
# do
#     datasets=("msmarco")
#     # datasets=("msmarco" "fiqa" "nq" "scifact" "nfcorpus" "quora" "hotpotqa" "fever")
#     for dataset in "${datasets[@]}"
#     do
#         demonstration_pool_path=../data/for_index/query_doc/${dataset}/collection_qnum=${querynum}/collection.jsonl
#         index_dir=../data/for_index/query_doc/${dataset}/${NEWEST_FOLDER}_${model}

#         torchrun --master_port 3943 --nproc_per_node 8 build_dense_index_after_train.py \
#                 --eval_batch_size 32 \
#                 --model_path /home/u20238046/workspace_lwh/project/demorank/train/src/dir/checkpoint/$NEWEST_FOLDER/$model \
#                 --output_dir $index_dir \
#                 --demonstration_pool_path $demonstration_pool_path \
#                 --dataset $dataset \
#                 --output_embedding_size 768 \

#     done
# done
