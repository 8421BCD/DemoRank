export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
for MODEL in "flan-t5-xl"
do
    for shot_num in 3
    do
        # Main experiment datasets: "hotpotqa" "nq" "fever" "dl19" "dl20" "msmarco"
        # BEIR datasets: "robust04" "scidocs" "dbpedia" "news"  "fiqa" "quora" "nfcorpus" 
        for DATASET in "dl19"
        do
            if [ $DATASET = "dl19" ] || [ $DATASET = "dl20" ] || [ $DATASET = "msmarco" ] || [ $DATASET = "robust04" ] || [ $DATASET = "scidocs" ] || [ $DATASET = "dbpedia" ] || [ $DATASET = "news" ] || [ $DATASET = "fiqa" ] || [ $DATASET = "quora" ] || [ $DATASET = "nfcorpus" ]; then
                dataset_class="msmarco"
            else
                dataset_class=$DATASET
            fi

            # ---------------------- main experiments - baselines ----------------------

            # for demonstration_type in "none" "random" "bm25" "kmeans" "dbs"
            for demonstration_type in "none"
            do
                torchrun --master_port 3471 --nproc_per_node 8 run_kshot.py \
                    --model_name_or_path /fs/archive/share/u2022000170/$MODEL \
                    --dataset $DATASET \
                    --dataset_class $dataset_class \
                    --demonstration_type $demonstration_type \
                    --shot_num $shot_num \
                    --output ${MODEL}_${demonstration_type}.txt \
                    --output_dir runs/$DATASET \
                    --llm_dtype bf16 \
                    --per_device_eval_batch_size 16 \
                    --demon_sparse_index_path data/for_index/query_doc/$dataset_class/index_bm25 \
                    --demon_dense_index_path data/for_index/query_doc/$dataset_class/e5-base-v2 \
                    --retriever_path ../../llm/e5-base-v2 \
                    --id_demon_path data/for_index/query_doc/$dataset_class/id_demon.json \
                    --id_demonquery_path data/for_index/query_doc/$dataset_class/id_demonquery.json \
                    --id_demondoc_path data/for_index/query_doc/$dataset_class/id_demondoc.json 
            done

            # use sbert retriever
            # demonstration_type="sbert" 
            # torchrun --master_port 3471 --nproc_per_node 8 run_kshot.py \
            #     --model_name_or_path /fs/archive/share/u2022000170/$MODEL \
            #     --dataset $DATASET \
            #     --dataset_class $dataset_class \
            #     --demonstration_type $demonstration_type \
            #     --shot_num $shot_num \
            #     --output ${MODEL}_${demonstration_type}.txt \
            #     --output_dir runs/$DATASET \
            #     --llm_dtype bf16 \
            #     --per_device_eval_batch_size 16 \
            #     --demon_sparse_index_path data/for_index/query_doc/$dataset_class/index_bm25 \
            #     --demon_dense_index_path data/for_index/query_doc/$dataset_class/paraphrase-mpnet-base-v2 \
            #     --retriever_path ../../llm/paraphrase-mpnet-base-v2 \
            #     --id_demon_path data/for_index/query_doc/$dataset_class/id_demon.json \
            #     --id_demonquery_path data/for_index/query_doc/$dataset_class/id_demonquery.json \
            #     --id_demondoc_path data/for_index/query_doc/$dataset_class/id_demondoc.json \

            # use e5 retriever
            # demonstration_type="e5" 
            # torchrun --master_port 3471 --nproc_per_node 8 run_kshot.py \
            #     --model_name_or_path /fs/archive/share/u2022000170/$MODEL \
            #     --dataset $DATASET \
            #     --dataset_class $dataset_class \
            #     --demonstration_type $demonstration_type \
            #     --shot_num $shot_num \
            #     --output ${MODEL}_${demonstration_type}.txt \
            #     --output_dir runs/$DATASET \
            #     --llm_dtype bf16 \
            #     --per_device_eval_batch_size 16 \
            #     --demon_sparse_index_path data/for_index/query_doc/$dataset_class/index_bm25 \
            #     --demon_dense_index_path data/for_index/query_doc/$dataset_class/e5-base-v2 \
            #     --retriever_path ../../llm/e5-base-v2 \
            #     --id_demon_path data/for_index/query_doc/$dataset_class/id_demon.json \
            #     --id_demonquery_path data/for_index/query_doc/$dataset_class/id_demonquery.json \
            #     --id_demondoc_path data/for_index/query_doc/$dataset_class/id_demondoc.json \
            
            # ------------------------------ main experiments: Ours ------------------------------
            ########### DRetriever only ###########
            # demonstration_type=demor
            # retriever_dir=${dataset_class}_qd
            # retriever_model=epoch2
            # torchrun --master_port 3649 --nproc_per_node 4 run_kshot.py \
            #     --model_name_or_path /fs/archive/share/u2022000170/$MODEL \
            #     --dataset $DATASET \
            #     --dataset_class $dataset_class \
            #     --demonstration_type $demonstration_type \
            #     --shot_num $shot_num \
            #     --output ${MODEL}_${demonstration_type}.txt \
            #     --output_dir runs/$DATASET \
            #     --llm_dtype bf16 \
            #     --per_device_eval_batch_size 8 \
            #     --demon_dense_index_path data/for_index/query_doc/$dataset_class/${retriever_dir}_${retriever_model} \
            #     --retriever_path train/src/dir/checkpoint/$retriever_dir/$retriever_model \
            #     --id_demon_path data/for_index/query_doc/$dataset_class/id_demon.json \
            #     --id_demonquery_path data/for_index/query_doc/$dataset_class/id_demonquery.json \
            #     --id_demondoc_path data/for_index/query_doc/$dataset_class/id_demondoc.json \

            ########### DemoRank(DRetriever+DReranker) ###########
            demonstration_type=demor
            retriever_dir=${dataset_class}_qd
            retriever_model=epoch2
            reranker_dir=cross_${dataset_class}
            reranker_model=epoch2
            torchrun --master_port 3229 --nproc_per_node 8 run_kshot.py \
                --model_name_or_path /fs/archive/share/u2022000170/$MODEL \
                --dataset $DATASET \
                --dataset_class $dataset_class \
                --demonstration_type $demonstration_type \
                --shot_num $shot_num \
                --output ${MODEL}_${demonstration_type}.txt \
                --output_dir runs/$DATASET \
                --llm_dtype bf16 \
                --per_device_eval_batch_size 8 \
                --demon_dense_index_path data/for_index/query_doc/$dataset_class/${retriever_dir}_${retriever_model} \
                --retriever_path train/src/dir/checkpoint/$retriever_dir/$retriever_model \
                --id_demon_path data/for_index/query_doc/$dataset_class/id_demon.json \
                --id_demonquery_path data/for_index/query_doc/$dataset_class/id_demonquery.json \
                --id_demondoc_path data/for_index/query_doc/$dataset_class/id_demondoc.json \
                --demonstration_rerank True \
                --demonstration_rerank_num 50 \
                --demonstration_reranker_path train/src/dir/checkpoint/$reranker_dir/$reranker_model \


        done
    done
done





# ------------------------------ auxiliary experiments: e5+reranker ------------------------------

# demonstration_type=dense

# reranker_dir=cross_nq_e5
# reranker_model="checkpoint-3126"
# for MODEL in "flan-t5-xl"
# do
#     for DATASET in "nq"
#     # for DATASET in "fiqa" "nq" "scifact" "nfcorpus" "quora"
#     do
#         if [ $DATASET = "dl19" ] || [ $DATASET = "dl20" ] || [ $DATASET = "msmarco" ]; then
#             dataset_class="msmarco"
#         elif [ $DATASET = "covid" ]; then
#             dataset_class="nfcorpus"
#         elif [ $DATASET = "fiqa" ]; then
#             dataset_class="hotpotqa"
#         else
#             dataset_class=$DATASET
#         fi
#         for shot_num in 3
#         do
#             torchrun --master_port 3649 --nproc_per_node 7 run_kshot.py \
#                 --model_name_or_path /fs/archive/share/u2022000170/$MODEL \
#                 --dataset $DATASET \
#                 --dataset_class $dataset_class \
#                 --demonstration_type $demonstration_type \
#                 --shot_num $shot_num \
#                 --output ${MODEL}_${demonstration_type}.txt \
#                 --output_dir runs/$DATASET \
#                 --llm_dtype bf16 \
#                 --per_device_eval_batch_size 8 \
#                 --demon_dense_index_path data/for_index/query_doc/$dataset_class/e5-base-v2 \
#                 --retriever_path ../../llm/e5-base-v2 \
#                 --id_demon_path data/for_index/query_doc/$dataset_class/id_demon.json \
#                 --id_demonquery_path data/for_index/query_doc/$dataset_class/id_demonquery.json \
#                 --id_demondoc_path data/for_index/query_doc/$dataset_class/id_demondoc.json \
#                 --demonstration_rerank True \
#                 --demonstration_rerank_num 50 \
#                 --demonstration_reranker_path train/src/dir/checkpoint/$reranker_dir/$reranker_model \
        
#         done
#     done
# done

