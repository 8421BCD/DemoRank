import argparse
import datetime
import time
from pyserini.search.lucene import LuceneSearcher
from pyserini.search import FaissSearcher
from pyserini.search._base import get_topics, get_qrels
from pyserini.output_writer import OutputFormat, get_output_writer
from dataclasses import dataclass, field
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    PreTrainedTokenizer,
    AutoTokenizer,
    Trainer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    logging,
    set_seed
)
import transformers
from torch.utils.data import Dataset, DataLoader
import torch
import json
from typing import Dict, Optional, Sequence
import copy
import logging
import os
import random
from prompts import PROMPT_DICT, PROMPT_DICT_YES_NO, DOC_FORMAT_DIC, DEFAULT_PROMPT, GBQ_PROMPT
from trec_eval import EvalFunction
from index_and_topics import THE_INDEX, THE_TOPICS, TRAIN_NUM
from tqdm import tqdm
from utils import get_yes_no_prompt, get_yes_no_prompt_for_qd
import csv
import pytrec_eval
from accelerate import Accelerator
from retrieve_utils import MyFaissSearcher
import torch.nn.functional as F
import numpy as np
from itertools import permutations

os.environ["PYSERINI_CACHE"] = "./cache" # set cache path
logger = logging.getLogger(__name__)

@dataclass
class PyseriniArguments:
    dataset: str = field(metadata={'help': 'test dataset.'})
    output: str = field(metadata={'help': 'Path to output file.'})
    output_dir: str = field(metadata={'help': 'Dir to output file.'})
    output_format: Optional[str] = field(default='trec', metadata={'help': 'Output format.'})
    hits: int = field(default=100, metadata={'help': 'Number of hits to retrieve per query.'})
    threads: int = field(default=16, metadata={'help': 'Number of threads.'})
    remove_query: Optional[bool] = field(default=True, metadata={'help': 'Remove query from output.'})
    save_first_stage_run: Optional[bool] = field(default=True, metadata={'help': 'Save first-stage run.'})
    remove_duplicates: Optional[bool] = field(default=False, metadata={'help': 'Remove duplicates from output.'})
    notes: str = field(default='', metadata={'help': 'notes for code running'})

@dataclass
class LLMArguments:
    model_name_or_path: str = field(metadata={'help': 'HF LLM name or path.'})
    in_context: Optional[bool] = field(default=False, metadata={'help': 'Whether to use in-context LLM.'})
    scoring_func: Optional[str] = field(default='yes_no', metadata={'help': 'Scoring function.'})
    cache_dir: Optional[str] = field(default='./cache', metadata={'help': 'Path to cache directory.'})
    data_path: Optional[str] = field(default=None, metadata={'help': 'Path to train data directory.'})
    llm_dtype: str = field(default='bf16', metadata={'help': 'Data type of llm.'})
    per_device_eval_batch_size: int = field(default=6, metadata={'help': 'inference batchsize'})
    llm_reward_path: str = field(default='data/for_distill/train_data_shards', metadata={'help': 'the path of scores of demonstrations.'})

@dataclass
class DemonArguments:
    demonstration_type: Optional[str] = field(default='none', metadata={'help': 'none, random, bm25, dense'})
    demon_sparse_index_path: str = field(default='data/for_index/query_doc/msmarco/index_bm25', metadata={'help': 'the path of demonstration index.'})
    demon_dense_index_path: str = field(default='data/for_index/query_doc/msmarco/e5-base-v2/', metadata={'help': 'the path of demonstration index.'})    
    id_demon_path: str = field(default='data/for_index/query_doc/msmarco/id_demon.json', metadata={'help': 'the path of id_demon dict.'})
    id_demonquery_path: str = field(default='data/for_index/query_doc/msmarco/id_demonquery.json', metadata={'help': 'the path of id_demonquery dict for training demonstration corpus.'})
    id_demondoc_path: str = field(default='data/for_index/query_doc/msmarco/id_demondoc.json', metadata={'help': 'the path of id_demondoc dict for training demonstration corpus'})
    start_idx: int = field(default=0, metadata={'help': 'start query idx'})
    end_idx: int = field(default=3000, metadata={'help': 'end query idx'})
    demonstration_totalnum: int = field(default=100, metadata={'help': 'the total number of demonstration candidates'})
    retriever_path: str = field(default='', metadata={'help': 'the path of demonstration retriever'})
    demonstration_num_each_type: int = field(default=10, metadata={'help': 'the demonstration number for each kind of sampling method'})
    iteration_number: int = field(default=1, metadata={'help': 'the number of iteration'})
    random_seed: int = field(default=11, metadata={'help': 'the random seed for different iteration'})
    shot_num: int = field(default=3, metadata={'help': 'how many times to expand the input'})
    traindata_type: str = field(default='retriever', metadata={'help': 'construct the train data for demonstration retriever or reranker'})
    is_subset: str = field(default='False', metadata={'help': 'whether to score the subset of train_input'})
    subset_querynum: int = field(default=10000, metadata={'help': 'the query number of the subset'})

@dataclass
class SearchResult:
    docid: str
    score: float
    raw: str


class LLMDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    def __init__(self, input_list, labels_list, args, tokenizer, all_demonstrations_list, id_demon):
        super(LLMDataset, self).__init__()
        self.sources = []
        self.targets = []
        for i in range(len(input_list)):
            current_input = input_list[i]
            all_demonstrations = all_demonstrations_list[i]
            label = labels_list[i]
            qd_text = current_input['input_text']
            if args.traindata_type == 'retriever':
                for demonid in all_demonstrations:
                    self.sources.append(get_yes_no_prompt_for_qd(args.index, qd_text, [id_demon[demonid]]))  # 1-shot
                    self.targets.append(label)
            elif args.traindata_type == 'reranker': # the reranker here is not our dependency-aware reranker, it treats each demonstration independently
                for demonid in all_demonstrations:
                    self.sources.append(get_yes_no_prompt_for_qd(args.index, qd_text, [id_demon[demonid]]))  # 1-shot
                    self.targets.append(label)
            self.sources.append(get_yes_no_prompt_for_qd(args.index, qd_text, None))
            self.targets.append(label)

        save_path = f'data/for_distill/sources_targets/{args.start_idx}_{args.end_idx}.jsonl'
        with open(save_path, 'w') as f: 
            for i in range(len(self.sources)):
                f.write(json.dumps({'source': self.sources[i], 'target': self.targets[i]}) + '\n')
            # print('sources and targets saved!')

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i):
        return dict(sources=self.sources[i], targets=self.targets[i])

def write_run(output_writer, results, pyserini_args):
    with output_writer:
        for topic, hits in results:
            if pyserini_args.remove_duplicates:
                seen_docids = set()
                dedup_hits = []
                for hit in hits:
                    if hit.docid.strip() in seen_docids:
                        continue
                    seen_docids.add(hit.docid.strip())
                    dedup_hits.append(hit)
                hits = dedup_hits

            # For some test collections, a query is doc from the corpus (e.g., arguana in BEIR).
            # We want to remove the query from the results.
            if pyserini_args.remove_query:
                hits = [hit for hit in hits if hit.docid != topic]

            # write results
            output_writer.write(topic, hits)

def get_random_prompts(qd_info, id_demon, demonstration_num, args):
    """
        return: dict(qid, demonstrations)
    """
    all_demonstrations = list(id_demon.keys())
    demon_idxs = random.sample(range(len(all_demonstrations)), demonstration_num + 20) 

    demonid_list = []
    train_input_id = qd_info['input_id']
    for demon_idx in demon_idxs:
        demon_id = all_demonstrations[demon_idx]
        if demon_id.split('#')[0] != train_input_id.split('#')[0]:  # filter the demonstrations which contain the same query with input
            demonid_list.append(demon_id)
    demonid_list = demonid_list[:demonstration_num]
    return demonid_list


def get_few_shot_prompts(qd_info, id_demonquery, id_demondoc, id_demon, searcher, demonstration_num, args, is_e5_retrieval = False):
    """
        return: dict(qid, demonstrations)
    """
    if args.traindata_type == 'retriever':
        # Note that, iterative training is a technique used for demonstration retriever training. However, it is not the focus of our paper and we do not apply iteratively training. Thus, we set the iteration_number as 1 in this paper.
        if args.iteration_number == 1: # During first iteration, we use bm25 to retrieve
            query = qd_info['query_text']
            query = 'query: ' + query if is_e5_retrieval else query
            hits = searcher.search(query, k=demonstration_num + 10) # following previous studies, we use the query in the train_input for retrieval
        else:  # using trained demonstration retriever to retrieve
            hits = searcher.search(qd_info['input_text'], k=demonstration_num + 10)
    elif args.traindata_type == 'reranker':
        hits = searcher.search(qd_info['input_text'], k=demonstration_num + 10)
    
    train_input_id = qd_info['input_id']
    demonid_list = []
    for hit in hits:
        if hit.docid.split('#')[0] != train_input_id.split('#')[0]:  # filter the demonstration containing the same query with input
            demonid_list.append(hit.docid)
    demonid_list = demonid_list[:demonstration_num]
    return demonid_list

def select_demon_by_rank(demon_ids, scores): # select demons based on rank
    combined = list(zip(demon_ids, scores))
    sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)  # Sort by scores in descending order
    
    # Calculate exp(-rank) for each demon_id
    weights = [np.exp(-rank) for rank in range(1, len(sorted_combined) + 1)]
    # weights = [1/rank for rank in range(1, len(sorted_combined) + 1)]
    
    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [weight / total_weight for weight in weights]
    
    # Sample a demon_id based on the normalized weights
    chosen_id = np.random.choice([item[0] for item in sorted_combined], p=normalized_weights)
    
    return chosen_id

# merge and remove duplicates, random_demonstrations are used to pad to demonstration_totalnum
def merge_demonstrations(bm25_demonstrations, random_demonstrations, demonstration_totalnum):
    merged_demonstrations = bm25_demonstrations
    for demonid in random_demonstrations:
        if demonid not in merged_demonstrations:
            merged_demonstrations.append(demonid)
    merged_demonstrations = merged_demonstrations[:demonstration_totalnum] # keep top-demonstration_totalnum

    return merged_demonstrations

def eval_ndcg(topic_id, run_trec_file, qrels):
    with open(run_trec_file, 'r')as f:
        run_data = f.readlines()
    runs = {}
    for line in run_data:
        line = line.split(" ")
        sample_id = line[0]
        doc_id = line[2]
        score = float(line[4])
        if sample_id not in runs:
            runs[sample_id] = {}
        runs[sample_id][doc_id] = score
    # for efficiency
    new_qrels = {topic_id: qrels[topic_id]}
    evaluator = pytrec_eval.RelevanceEvaluator(new_qrels, {'ndcg_cut.10,50'})
    res = evaluator.evaluate(runs)
    ndcg10 = res[topic_id]['ndcg_cut_10']
    ndcg50 = res[topic_id]['ndcg_cut_50']
    return ndcg10, ndcg50

def main():
    parser = HfArgumentParser((PyseriniArguments, LLMArguments, DemonArguments))
    pyserini_args, model_args, demon_args = parser.parse_args_into_dataclasses()
    pyserini_args: PyseriniArguments
    model_args: LLMArguments
    demon_args: DemonArguments

    random.seed(demon_args.random_seed) 
    set_seed(demon_args.random_seed) 

    accelerator = Accelerator()
    device = accelerator.device

    pyserini_args.index = THE_INDEX[pyserini_args.dataset]
        
    args = argparse.Namespace(
        **vars(pyserini_args), **vars(model_args), **vars(demon_args)
    )
    if args.traindata_type == 'retriever':
        demon_args.end_idx = TRAIN_NUM[pyserini_args.dataset]
    # iteration_number is set as 1 in this project
    print(f'############ iteration_number={args.iteration_number}, dataset={args.dataset} ############')
    if not os.path.exists(pyserini_args.output_dir):
        os.makedirs(pyserini_args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        model_max_length=2048,
    )
    tokenizer.pad_token = tokenizer.eos_token
    if 't5' in model_args.model_name_or_path:
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"
    else:
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"

    model_name = args.model_name_or_path.split('/')[-1]
    if args.is_subset == 'True': # for our auxiliary experiments (for comparison with supervised models in Section 5.3.2 of our paper)
        with open(f'data/for_distill/train_input/{args.dataset}_{args.subset_querynum}.json', 'r') as f:
            train_input = json.load(f)
        result_path = os.path.join(model_args.llm_reward_path, f'{args.dataset}_{model_name}_{demon_args.shot_num}shot_querynum={args.subset_querynum}.jsonl')  
    else:
        with open(f'data/for_distill/train_input/{args.dataset}.json', 'r') as f:
            train_input = json.load(f)
        train_input = train_input[demon_args.start_idx: demon_args.end_idx]
        result_path = os.path.join(model_args.llm_reward_path, f'{args.dataset}_{demon_args.start_idx}_{demon_args.end_idx}_{model_name}_{demon_args.shot_num}shot.jsonl')  
    done_demonids = set()
    if os.path.exists(result_path):
        with open(result_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                done_demonids.add(data['train_input_id'])
    # get id_demonquery, id_demondoc, id_demon
    with open(args.id_demonquery_path, 'r') as f: 
        id_demonquery = json.load(f)
    with open(args.id_demondoc_path, 'r') as f: 
        id_demondoc = json.load(f)
    with open(args.id_demon_path, 'r') as f:  # id_demon will be used for random sample
        id_demon = json.load(f)

    # load model
    if args.llm_dtype == "bf16":
        dtype = torch.bfloat16
    elif args.llm_dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    if 't5' in model_args.model_name_or_path:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            torch_dtype=dtype,
            device_map={'': device} 
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            torch_dtype=dtype,
            device_map={'': device}
        )
    model.to(device)
    model.eval()
    option_ids = [
        tokenizer.encode("Yes", add_special_tokens=False)[0],
        tokenizer.encode("No", add_special_tokens=False)[0]
    ]
    option_tensors = torch.tensor(option_ids, dtype=torch.long).to(device)
    #####################################################
    # retrieve demonstration
    #####################################################
    print('loading auxiliary data...')
    if args.traindata_type == 'retriever':
        if demon_args.iteration_number == 1:
            demon_searcher_bm25 = LuceneSearcher(demon_args.demon_sparse_index_path)
        else:
            demon_searcher_dense = MyFaissSearcher(index_dir=demon_args.demon_dense_index_path, query_encoder=demon_args.retriever_path, gpu_idx=device.index, device=device)
            retriever_name = args.retriever_path.split('/')[-1]
    elif args.traindata_type == 'reranker':
        demon_searcher_dense = MyFaissSearcher(index_dir=demon_args.demon_dense_index_path, query_encoder=demon_args.retriever_path, gpu_idx=device.index, device=device)
        retriever_name = args.retriever_path.split('/')[-1]
    else: 
        raise ValueError('traindata_type must be retriever or reranker')
    #####################################################
    # llm scoring
    #####################################################
    with torch.no_grad():
        done_num = 0
        while done_num < len(train_input) and train_input[done_num]['input_id'] in done_demonids:
            done_num += 1
        print(f'{done_num} train inputs passed')
        train_input_batchsize = 8 # process a batch of input at a time
        train_input_shards = [train_input[i: i + train_input_batchsize] for i in range(done_num, len(train_input), train_input_batchsize)]
        for input_list in tqdm(train_input_shards, desc=f'process train inputs of [{args.dataset}], total:{len(train_input) - done_num}', ncols=100): # process a batch of train inputs 
            labels_list = ['Yes' if current_input['input_id'].split('#')[-1] == '1' else 'No' for current_input in input_list]
            all_demonstrations_list = []
            for current_input in input_list:
                if args.traindata_type == 'retriever':
                    if args.iteration_number == 1:
                        bm25_demonstrations = get_few_shot_prompts(current_input, id_demonquery, id_demondoc, id_demon, demon_searcher_bm25, demon_args.demonstration_num_each_type, args)
                        random_demonstrations = get_random_prompts(current_input, id_demon, demon_args.demonstration_totalnum, args) # sample enough random demonstration for following merge
                        all_demonstrations = merge_demonstrations(bm25_demonstrations, random_demonstrations, demon_args.demonstration_totalnum)
                    else:
                        dense_demonstrations = get_few_shot_prompts(current_input, id_demonquery, id_demondoc, id_demon, demon_searcher_dense, demon_args.demonstration_num_each_type, args)
                        random_demonstrations = get_random_prompts(current_input, id_demon, demon_args.demonstration_totalnum, args) # sample enough random demonstration for following merge
                        all_demonstrations = merge_demonstrations(dense_demonstrations, random_demonstrations, demon_args.demonstration_totalnum) # [ id, id, id,]
                elif args.traindata_type == 'reranker':
                    dense_demonstrations = get_few_shot_prompts(current_input, id_demonquery, id_demondoc, id_demon, demon_searcher_dense, demon_args.demonstration_totalnum, args)
                    all_demonstrations = dense_demonstrations
                else: 
                    raise ValueError('traindata_type must be retriever or reranker')
                all_demonstrations_list.append(all_demonstrations)

            option_logits = []
            results = []
            dataset = LLMDataset(input_list, labels_list, args, tokenizer, all_demonstrations_list, id_demon)
            dataloader = DataLoader(dataset, batch_size=args.per_device_eval_batch_size)
            dataloader = accelerator.prepare(dataloader)
            for batch in dataloader:
                inputs = tokenizer(batch['sources'], return_tensors="pt", padding="longest", max_length=tokenizer.model_max_length, truncation=True,)
                input_ids=inputs['input_ids'].to(device)
                attention_mask=inputs['attention_mask'].to(device)
                labels = tokenizer(batch['targets'], return_tensors="pt", padding="longest", max_length=tokenizer.model_max_length, truncation=True,).input_ids
                labels[labels == tokenizer.pad_token_id] = -100
                labels = labels.to(device)
                if 't5' in model_args.model_name_or_path:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    logits = outputs.logits # bsz, seq_len, vocab_size
                    logits = logits[:, 0, :]
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits # bsz, seq_len, vocab_size
                    logits = logits[:, -1, :]   
                batch_size = logits.shape[0]
                option_tensor_index = option_tensors.unsqueeze(0).repeat(batch_size, 1)
                option_logits_batch = torch.gather(logits, dim=1, index=option_tensor_index)
                option_logits_batch = F.softmax(option_logits_batch, dim=-1)
                option_logits_batch_gathered = accelerator.gather_for_metrics(option_logits_batch)
                option_logits.extend(option_logits_batch_gathered)
            option_logits = torch.stack(option_logits, dim=0)
            begin_idx = 0
            for i in range(len(all_demonstrations_list)):
                chunk_size = len(all_demonstrations_list[i]) + 1
                label = labels_list[i]
                if label == 'Yes': # Yes
                    scores = option_logits[begin_idx:begin_idx + chunk_size, 0].tolist()
                else:              # No
                    scores = option_logits[begin_idx:begin_idx + chunk_size, 1].tolist()
                begin_idx += chunk_size
                result = {
                        'train_input_id': input_list[i]['input_id'],
                        'selected_demonids': [],
                        'label': label,
                        'none_score': scores[-1],
                        'train_input': input_list[i]['input_text'],
                        'demon_ids': all_demonstrations_list[i],
                        'scores': scores[:-1],
                        }
                results.append(result)
            assert(begin_idx == len(dataset))
            if accelerator.process_index == 0:
                with open(result_path, 'a') as f:
                    for result in results:
                        f.write(json.dumps(result) + '\n')

if __name__ == '__main__':
    main()