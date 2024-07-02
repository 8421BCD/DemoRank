export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

################## for msmarco ##################
start_idx=0
end_idx=100000
MODEL=flan-t5-xl
dir=msmarco_qd
model=epoch2
datasets=("msmarco")
for DATASET in "${datasets[@]}"
do
    torchrun --master_port 1123 --nproc_per_node 8 llm_score_dependency_aware_rerank.py \
        --start_idx $start_idx \
        --end_idx $end_idx \
        --random_seed 11 \
        --demonstration_totalnum 50 \
        --demonstration_num_each_type 25 \
        --model_name_or_path ../../llm/$MODEL \
        --dataset $DATASET \
        --llm_reward_path data/for_distill/train_data_reranker_greedy \
        --output ${MODEL}_${DATASET}_${start_idx}_${end_idx}.txt \
        --output_dir runs/$DATASET \
        --llm_dtype bf16 \
        --per_device_eval_batch_size 8 \
        --id_demon_path data/for_index/query_doc/${DATASET}/id_demon.json \
        --id_demonquery_path data/for_index/query_doc/${DATASET}/id_demonquery.json \
        --id_demondoc_path data/for_index/query_doc/${DATASET}/id_demondoc.json \
        --retriever_path train/src/dir/checkpoint/$dir/$model \
        --demon_dense_index_path data/for_index/query_doc/$DATASET/${dir}_${model} \
        --iteration_number 1 \
        --traindata_type reranker \

done

################## for hotpotqa ##################
# start_idx=0
# end_idx=100000
# MODEL=flan-t5-xl
# dir=hotpotqa_qd
# model=epoch2
# datasets=("hotpotqa")
# for DATASET in "${datasets[@]}"
# do
#     torchrun --master_port 1123 --nproc_per_node 8 llm_score_dependency_aware_rerank.py \
#         --start_idx $start_idx \
#         --end_idx $end_idx \
#         --random_seed 11 \
#         --demonstration_totalnum 50 \
#         --demonstration_num_each_type 25 \
#         --model_name_or_path ../../llm/$MODEL \
#         --dataset $DATASET \
#         --llm_reward_path data/for_distill/train_data_reranker_greedy \
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

################## for nq ##################
# start_idx=0
# end_idx=100000
# MODEL=flan-t5-xl
# dir=nq_qd
# model=epoch2
# datasets=("nq")
# for DATASET in "${datasets[@]}"
# do
#     torchrun --master_port 1123 --nproc_per_node 8 llm_score_dependency_aware_rerank.py \
#         --start_idx $start_idx \
#         --end_idx $end_idx \
#         --random_seed 11 \
#         --demonstration_totalnum 50 \
#         --demonstration_num_each_type 25 \
#         --model_name_or_path ../../llm/$MODEL \
#         --dataset $DATASET \
#         --llm_reward_path data/for_distill/train_data_reranker_greedy \
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

################## for fever ##################
# start_idx=0
# end_idx=100000
# MODEL=flan-t5-xl
# dir=fever_qd
# model=epoch2
# datasets=("fever")
# for DATASET in "${datasets[@]}"
# do
#     torchrun --master_port 1123 --nproc_per_node 8 llm_score_dependency_aware_rerank.py \
#         --start_idx $start_idx \
#         --end_idx $end_idx \
#         --random_seed 11 \
#         --demonstration_totalnum 50 \
#         --demonstration_num_each_type 25 \
#         --model_name_or_path ../../llm/$MODEL \
#         --dataset $DATASET \
#         --llm_reward_path data/for_distill/train_data_reranker_greedy \
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


################################ for comparison with supervised models ##############################
# querynum=20000

# MODEL=flan-t5-xl
# dir=msmarco_qd_querynum=$querynum
# model=checkpoint-1875
# # datasets=("fiqa" "nq" "scifact" "nfcorpus" "quora" "hotpotqa")
# datasets=("msmarco")
# for DATASET in "${datasets[@]}"
# do
#     torchrun --master_port 1123 --nproc_per_node 5 llm_score_dependency_aware_rerank.py \
#         --random_seed 11 \
#         --demonstration_totalnum 50 \
#         --demonstration_num_each_type 50 \
#         --model_name_or_path ../../llm/$MODEL \
#         --dataset $DATASET \
#         --llm_reward_path data/for_distill/train_data_reranker_greedy \
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

################################ for e5(retriever) ##############################
# start_idx=0
# end_idx=100000
# MODEL=flan-t5-xl
# datasets=("nq")
# for DATASET in "${datasets[@]}"
# do
#     torchrun --master_port 1123 --nproc_per_node 8 llm_score_dependency_aware_rerank.py \
#         --start_idx $start_idx \
#         --end_idx $end_idx \
#         --random_seed 11 \
#         --demonstration_totalnum 50 \
#         --demonstration_num_each_type 25 \
#         --model_name_or_path ../../llm/$MODEL \
#         --dataset $DATASET \
#         --llm_reward_path data/for_distill/train_data_reranker_greedy \
#         --output ${MODEL}_${DATASET}_${start_idx}_${end_idx}.txt \
#         --output_dir runs/$DATASET \
#         --llm_dtype bf16 \
#         --per_device_eval_batch_size 8 \
#         --id_demon_path data/for_index/query_doc/${DATASET}/id_demon.json \
#         --id_demonquery_path data/for_index/query_doc/${DATASET}/id_demonquery.json \
#         --id_demondoc_path data/for_index/query_doc/${DATASET}/id_demondoc.json \
#         --retriever_path ../../llm/e5-base-v2 \
#         --demon_dense_index_path data/for_index/query_doc/$DATASET/e5-base-v2 \
#         --iteration_number 1 \
#         --traindata_type reranker \

# done
