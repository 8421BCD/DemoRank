export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# -------------------------- construct retriever data through llm scoring --------------------------
start_idx=0
end_idx=200000 # the range of train inputs to score
shot_num=1
MODEL=flan-t5-xl
# datasets=("msmarco" "fever" "nq" "hotpotqa")
datasets=("msmarco")
for DATASET in "${datasets[@]}"
do
    torchrun --master_port 1113 --nproc_per_node 8 llm_score.py \
        --start_idx $start_idx \
        --end_idx $end_idx \
        --random_seed 11 \
        --demonstration_totalnum 50 \
        --demonstration_num_each_type 25 \
        --shot_num $shot_num \
        --model_name_or_path ../../llm/$MODEL \
        --dataset $DATASET \
        --llm_reward_path data/for_distill/train_data_retriever/ \
        --output ${MODEL}_${DATASET}_${shot_num}shot.txt \
        --output_dir runs/$DATASET \
        --llm_dtype bf16 \
        --per_device_eval_batch_size 20 \
        --id_demon_path data/for_index/query_doc/${DATASET}/id_demon.json \
        --id_demonquery_path data/for_index/query_doc/${DATASET}/id_demonquery.json \
        --id_demondoc_path data/for_index/query_doc/${DATASET}/id_demondoc.json \
        --demon_sparse_index_path data/for_index/query_doc/${DATASET}/index_bm25 \
        --iteration_number 1 \
        --traindata_type retriever \

done



# -------------------------- reranker data (without dependency-aware) --------------------------
# start_idx=0
# end_idx=100000
# shot_num=1
# MODEL=flan-t5-xl
# dir=nq_qd
# model=checkpoint-4688
# datasets=("msmarco")
# for DATASET in "${datasets[@]}"
# do
#     torchrun --master_port 1123 --nproc_per_node 8 llm_score.py \
#         --start_idx $start_idx \
#         --end_idx $end_idx \
#         --random_seed 11 \
#         --demonstration_totalnum 50 \
#         --demonstration_num_each_type 50 \
#         --shot_num 1 \
#         --model_name_or_path ../../llm/$MODEL \
#         --dataset $DATASET \
#         --llm_reward_path data/for_distill/train_data_reranker \
#         --output ${MODEL}_${DATASET}_${start_idx}_${end_idx}.txt \
#         --output_dir runs/$DATASET \
#         --llm_dtype bf16 \
#         --per_device_eval_batch_size 8 \
#         --id_demon_path data/for_index/query_doc/${DATASET}/id_demon.json \
#         --id_demonquery_path data/for_index/query_doc/${DATASET}/id_demonquery.json \
#         --id_demondoc_path data/for_index/query_doc/${DATASET}/id_demondoc.json \
#         --retriever_path llm_retriever/src/dir/checkpoint/$dir/$model \
#         --demon_dense_index_path data/for_index/query_doc/$DATASET/${dir}_${model} \
#         --iteration_number 1 \
#         --traindata_type reranker \

# done



################################ for subset ##############################
# querynum=20000
# # -------------------------------- 1. retriever --------------------------------
# shot_num=1
# MODEL=flan-t5-xl
# # datasets=("nfcorpus")
# datasets=("msmarco")
# for DATASET in "${datasets[@]}"
# do
#     torchrun --master_port 1113 --nproc_per_node 8 llm_score.py \
#         --random_seed 11 \
#         --demonstration_totalnum 100 \
#         --demonstration_num_each_type 50 \
#         --shot_num $shot_num \
#         --model_name_or_path ../../llm/$MODEL \
#         --dataset $DATASET \
#         --llm_reward_path data/for_distill/train_data_retriever/ \
#         --output ${MODEL}_${DATASET}_${shot_num}shot.txt \
#         --output_dir runs/$DATASET \
#         --llm_dtype bf16 \
#         --per_device_eval_batch_size 20 \
#         --id_demon_path data/for_index/query_doc/${DATASET}/id_demon_qnum=${querynum}.json \
#         --id_demonquery_path data/for_index/query_doc/${DATASET}/id_demonquery_qnum=${querynum}.json \
#         --id_demondoc_path data/for_index/query_doc/${DATASET}/id_demondoc_qnum=${querynum}.json \
#         --demon_sparse_index_path data/for_index/query_doc/${DATASET}/index_bm25_qnum=${querynum} \
#         --iteration_number 1 \
#         --traindata_type retriever \
#         --is_subset True \
#         --subset_querynum $querynum
# done
# -------------------------------- 2. reranker --------------------------------
# shot_num=1
# MODEL=flan-t5-xl
# dir=msmarco_qd_querynum=$querynum
# model=checkpoint-3125
# # datasets=("fiqa" "nq" "scifact" "nfcorpus" "quora" "hotpotqa")
# datasets=("msmarco")
# for DATASET in "${datasets[@]}"
# do
#     torchrun --master_port 1123 --nproc_per_node 8 llm_score.py \
#         --random_seed 11 \
#         --demonstration_totalnum 50 \
#         --demonstration_num_each_type 50 \
#         --shot_num 1 \
#         --model_name_or_path ../../llm/$MODEL \
#         --dataset $DATASET \
#         --llm_reward_path data/for_distill/train_data_reranker \
#         --output ${MODEL}_${DATASET}_${start_idx}_${end_idx}.txt \
#         --output_dir runs/$DATASET \
#         --llm_dtype bf16 \
#         --per_device_eval_batch_size 8 \
#         --id_demon_path data/for_index/query_doc/${DATASET}/id_demon.json \
#         --id_demonquery_path data/for_index/query_doc/${DATASET}/id_demonquery.json \
#         --id_demondoc_path data/for_index/query_doc/${DATASET}/id_demondoc.json \
#         --retriever_path llm_retriever/src/dir/checkpoint/$dir/$model \
#         --demon_dense_index_path data/for_index/query_doc/$DATASET/${dir}_${model} \
#         --iteration_number 1 \
#         --traindata_type reranker \
#         --is_subset True \
#         --subset_querynum $querynum

# done




