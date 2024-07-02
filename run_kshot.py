import argparse
import datetime
from pyserini.search.lucene import LuceneSearcher
from pyserini.search import FaissSearcher
from pyserini.search._base import get_topics, get_qrels
from pyserini.output_writer import OutputFormat, get_output_writer
from dataclasses import dataclass, field
from transformers import (
    HfArgumentParser,
    # LlamaTokenizer,
    TrainingArguments,
    PreTrainedTokenizer,
    AutoTokenizer,
    Trainer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    logging,
    set_seed
)
from torch.utils.data import Dataset, DataLoader
import torch
import json
from typing import Dict, Optional, Sequence
import copy
import logging
import os
import random
from prompts import PROMPT_DICT_YES_NO, DOC_FORMAT_DIC
from trec_eval import Eval
from index_and_topics import THE_INDEX, THE_TOPICS, THE_QRELS, DOC_MAXLEN, QUERY_MAXLEN
from tqdm import tqdm
from utils import get_yes_no_prompt, get_yes_no_prompt_system_user, clean_and_trunc_text
import csv
from accelerate import Accelerator
from retrieve_utils import MyFaissSearcher
import torch.nn.functional as F
import traceback
import pytrec_eval
import numpy as np
from itertools import permutations

# transformers.logging.set_verbosity_info()
os.environ["PYSERINI_CACHE"] = "./cache"
IGNORE_INDEX = -100
random.seed(11)  # 11
set_seed(11)  # 11
logger = logging.getLogger(__name__)


@dataclass
class PyseriniArguments:
    dataset: str = field(metadata={'help': 'test dataset.'})
    dataset_class: str = field(metadata={'help': ''})
    output: str = field(metadata={'help': 'Path to output file.'})
    output_dir: str = field(metadata={'help': 'Dir to output file.'})
    output_format: Optional[str] = field(default='trec', metadata={'help': 'Output format.'})
    hits: int = field(default=100, metadata={'help': 'Number of hits to retrieve per query.'})
    threads: int = field(default=30, metadata={'help': 'Number of threads.'})
    batchsize: int = field(default=32, metadata={'help': 'batchsize for dense retrieval'})
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

@dataclass
class DemonArguments:
    demonstration_type: Optional[str] = field(default='none', metadata={'help': 'none, random, bm25, dense'})
    shot_num: Optional[int] = field(default=0, metadata={'help': 'number of shot for inference'})
    # topn: Optional[int] = field(default=1, metadata={'help': 'n-th retrieved demonstration to use'})
    demon_sparse_index_path: str = field(default='data/for_index/query_doc/msmarco/index_bm25', metadata={'help': 'the path of demonstration index.'})
    demon_dense_index_path: str = field(default='data/for_index/query_doc/msmarco/e5-base-v2/', metadata={'help': 'the path of demonstration index.'})    
    id_demon_path: str = field(default='data/for_index/query_doc/msmarco/id_demon.json', metadata={'help': 'the path of id_demon dict'})
    id_demonquery_path: str = field(default='data/for_index/query_doc/msmarco/id_demonquery.json', metadata={'help': 'the path of id_demonquery dict'})
    id_demondoc_path: str = field(default='data/for_index/query_doc/msmarco/id_demondoc.json', metadata={'help': 'the path of id_demondoc dict'})
    retriever_path: str = field(default='', metadata={'help': 'the path of demonstration retriever'})
    demonstration_rerank: Optional[bool] = field(default=False, metadata={'help': 'Whether to rerank the demonstration candidate set'})
    demonstration_rerank_num: Optional[int] = field(default=100, metadata={'help': 'the number of demonstrations to rerank'})
    demonstration_reranker_path: Optional[str] = field(default='', metadata={'help': 'the path of demonstration reranker'})
    demonstration_reranker_batchsize: Optional[int] = field(default=128, metadata={'help': 'the batchsize for reranker inference'})

@dataclass
class SearchResult:
    docid: str
    score: float
    raw: str


class LLMDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, results, id_query, id_doc, topics, pyserini_args, model_args: LLMArguments, tokenizer: PreTrainedTokenizer, id_demon,
                 qdid_few_shot_prompts=None):
        super(LLMDataset, self).__init__()
        logging.warning("processing first stage results...")
        self.sources = []
        self.targets = []
        # for llama3
        self.systems = []
        self.users = []

        model_name = model_args.model_name_or_path.split('/')[-1]
        for qid, ranking in tqdm(results, ncols=100):
            # query = topics[qid]
            # query = clean_and_trunc_text(tokenizer, query, QUERY_MAXLEN[pyserini_args.dataset])
            query = id_query[qid]
            for doc in ranking:
                qdid = qid + '#' + doc.docid
                json_doc = json.loads(doc.raw)
                # doc_text = DOC_FORMAT_DIC[pyserini_args.index].format_map(json_doc)
                # doc_text = clean_and_trunc_text(tokenizer, doc_text, 130)
                doc_text = id_doc[doc.docid]
                self.sources.append(get_yes_no_prompt(pyserini_args.index, query, doc_text, [id_demon[demon_id] for demon_id in qdid_few_shot_prompts[qdid]] if qdid_few_shot_prompts else None, model_name))
                self.targets.append("Yes")
                system, user = get_yes_no_prompt_system_user(pyserini_args.index, query, doc_text, [id_demon[demon_id] for demon_id in qdid_few_shot_prompts[qdid]] if qdid_few_shot_prompts else None, model_name)
                self.systems.append(system)
                self.users.append(user)
        with open(f'data/input_output/{pyserini_args.dataset}.jsonl', 'w') as f:
            for i in range(len(self.sources)):
                f.write(json.dumps({'source': self.sources[i], 'target': self.targets[i]}) + '\n')
    def __len__(self):
        return len(self.sources)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(sources=self.sources[i], targets=self.targets[i], systems=self.systems[i], users=self.users[i])


class DemonRerankDataset(Dataset):
    def __init__(self, id_query, id_doc, id_demonquery, id_demondoc, id_demon, topics, results, qdid_few_shot_prompts, tokenizer, args):
        super(DemonRerankDataset, self).__init__()
        self.topics = topics
        self.id_query = id_query
        self.id_doc = id_doc
        self.id_demonquery = id_demonquery
        self.id_demondoc = id_demondoc
        self.id_demon = id_demon
        self.qdid_few_shot_prompts = qdid_few_shot_prompts
        self.tokenizer = tokenizer
        self.args = args

        self.qids = []
        self.docids = []
        # self.demonqids = []
        # self.demondocids = []
        # self.demonlabelids = []
        # self.demon_candidates = [] # each element is a demonid
        self.demonids = [] # each element is a demonid
        for result in results:
            qid = result[0]
            for doc in result[1]:
                docid = doc.docid
                qdid = qid + '#' + docid
                for candidate in qdid_few_shot_prompts[qdid]:
                    self.qids.append(qid)
                    self.docids.append(docid)
                    self.demonids.append(candidate)
    def __len__(self):
        return len(self.qids)

    def __getitem__(self, i):
        return dict(qid=self.qids[i], docid=self.docids[i], demonid=self.demonids[i])
    
    def collate_fn(self, batch):
        inputs, demons = [], []
        for data in batch:
            qid = data['qid']
            # querytext = self.topics[qid]
            # querytext = clean_and_trunc_text(self.tokenizer, querytext, QUERY_MAXLEN[self.args.dataset])
            querytext = self.id_query[qid]
            docid = data['docid']
            # doctext = DOC_FORMAT_DIC[self.args.index].format_map(self.id_doc[docid])
            # doctext = clean_and_trunc_text(self.tokenizer, doctext, DOC_MAXLEN[self.args.dataset])
            doctext = self.id_doc[docid]
            input = PROMPT_DICT_YES_NO[self.args.index][self.args.model_name].format_map({'qry': querytext, 'doc': doctext})
            # demon_text = '\n\n'.join([self.id_demon[demonid] for demonid in data['demon_candidates']])
            demon_text = self.id_demon[data['demonid']]
            inputs.append(input)
            demons.append(demon_text)
        tokenized_inputs = self.tokenizer(inputs,
                                        text_pair=demons,
                                        max_length=512,
                                        padding='longest',
                                        truncation=True,
                                        return_token_type_ids=False,
                                        return_tensors="pt")
        return tokenized_inputs


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

def get_dbs_prompts(results, args):
    difficulty_score_dir = '/home/u20238046/workspace_lwh/project/demorank/data/llm_difficulty_score'
    with open(os.path.join(difficulty_score_dir, f'{args.dataset_class}_{args.model_name}.json'), 'r') as f: 
        all_demonstrations = json.load(f)
    qdid_demons = {}
    for result in results:
        qid = result[0]
        for doc in result[1]:
            docid = doc.docid
            qdid = qid + '#' + docid
            qdid_demons[qdid] = [demonstration['demon_id'] for demonstration in all_demonstrations[:args.shot_num]]
    return qdid_demons

def get_kmeans_prompts(results, args):
    qdid_demons = {}
    demonids = []
    kmeans_kshot_path = f'/home/u20238046/workspace_lwh/project/demorank/data/for_index/query_doc/{args.dataset_class}/kmeans_results/{args.shot_num}shot.txt'
    with open(kmeans_kshot_path, 'r') as f: 
        for line in f: 
            demonids.append(line.strip())
    for result in results:
        qid = result[0]
        for doc in result[1]:
            docid = doc.docid
            qdid = qid + '#' + docid
            qdid_demons[qdid] = demonids
    return qdid_demons

def get_random_prompts(results, id_demon, args):
    """
        return: Dict(qid, List)
    """
    qdid_demons = {}
    all_demonstrations = list(id_demon.keys())

    search_num = args.shot_num
    for result in results:
        qid = result[0]
        for doc in result[1]:
            docid = doc.docid
            qdid = qid + '#' + docid
            demonstration_list = []
            demon_idxs = random.sample(range(len(all_demonstrations)), search_num)
            demon_ids = [all_demonstrations[idx] for idx in demon_idxs]

            for demon_id in demon_ids:
                demonstration_list.append(demon_id)
            qdid_demons[qdid] = demonstration_list

    return qdid_demons

def get_few_shot_prompts(results, id_query, id_doc, topics, searcher, tokenizer, args):
    """
        return: dict(qid, demonstrations)
    """
    qdid_demons = {}
    for result in tqdm(results, desc='retrieving demonstration...', ncols=100):
        qid = result[0]
        # querytext = topics[qid]
        # querytext = clean_and_trunc_text(tokenizer, querytext, QUERY_MAXLEN[args.dataset])
        querytext = id_query[qid]
        batch_queries = []
        batch_docs = []
        batch_inputs = []
        batch_inputids = []
        for doc in result[1]:
            docid = doc.docid
            qdid = qid + '#' + docid
            # doctext = DOC_FORMAT_DIC[args.index].format_map(json.loads(doc.raw))
            # doctext = clean_and_trunc_text(tokenizer, doctext, DOC_MAXLEN[args.dataset])
            doctext = id_doc[docid]
            current_input = PROMPT_DICT_YES_NO[args.index][args.model_name].format_map({'qry': querytext, 'doc': doctext})
            
            if args.demonstration_type == 'e5':
                batch_queries.append('query: ' + querytext)
                batch_inputs.append('query: ' + current_input)
            else:
                batch_queries.append(querytext)
                batch_inputs.append(current_input)
            batch_docs.append(doctext)
            batch_inputids.append(qdid)
        search_num = args.shot_num if args.demonstration_rerank == False else args.demonstration_rerank_num
        if args.demonstration_type == 'bm25':
            qdid_hits = searcher.batch_search(batch_queries, batch_inputids, k=search_num, threads=args.threads) # use query to search
        elif args.demonstration_type == 'e5':
            qdid_hits = searcher.batch_search(batch_queries, batch_inputids, k=search_num, batchsize=args.batchsize) # use query to search
        elif args.demonstration_type in ['sbert', 'demor']:
            qdid_hits = searcher.batch_search(batch_inputs, batch_inputids, k=search_num, batchsize=args.batchsize) # input is demonstration string 
        else:
            raise ValueError('demonstration_type not exists')
        for qdid in batch_inputids:
            qdid_demons[qdid] = [hit.docid for hit in qdid_hits[qdid]]
            
    return qdid_demons

def evaluate_results(args, out_path):
    all_metrics = Eval(out_path, THE_QRELS[args.dataset])
    print(all_metrics)
    result = {'model': args.model_name_or_path.split(f'/')[-1],
              'scoring_func': args.scoring_func,
              'demonstration_type': args.demonstration_type,
              'shot_num': args.shot_num if args.demonstration_type != 'none' else 0,
              'demonstration_rerank': args.demonstration_rerank,
              'datetime': str(datetime.datetime.now()),
              'retriever_model': '_'.join(args.retriever_path.split('/')[-2:]) if args.demonstration_type in ['e5', 'sbert', 'demor'] else '',
              'reranker_model': '_'.join(args.demonstration_reranker_path.split('/')[-2:]) if args.demonstration_rerank else '',
              'notes': args.notes,
              **all_metrics}
    result_path = 'results/metrics_{}.json'.format(args.dataset)
    if os.path.exists(result_path) == False:
        with open(result_path, 'w') as f: 
            json.dump([], f, indent=4)
    with open(result_path, 'r') as f:
        json_data = json.load(f)
        json_data.append(result)
    with open(result_path, 'w') as f: 
        json.dump(json_data, f, indent=4)

def main():
    parser = HfArgumentParser((PyseriniArguments, LLMArguments, DemonArguments))
    pyserini_args, model_args, demon_args = parser.parse_args_into_dataclasses()
    pyserini_args: PyseriniArguments
    model_args: LLMArguments
    demon_args: DemonArguments

    accelerator = Accelerator()
    device = accelerator.device
    model_args.model_name = model_args.model_name_or_path.split('/')[-1]
    pyserini_args.index = THE_INDEX[pyserini_args.dataset]
    pyserini_args.topics = THE_TOPICS[pyserini_args.dataset] if pyserini_args.dataset != 'dl20' else 'dl20'
    args = argparse.Namespace(
        **vars(pyserini_args), **vars(model_args), **vars(demon_args)
    )
    if not os.path.exists(pyserini_args.output_dir):
        os.makedirs(pyserini_args.output_dir)
    if 't5' in model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            model_max_length=2048,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            model_max_length=2048,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
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

    #####################################################
    # The first stage run
    #####################################################

    print('First stage run...')
    searcher = LuceneSearcher.from_prebuilt_index(pyserini_args.index)

    topics = {str(qid): content['title'] for qid, content in get_topics(pyserini_args.topics).items()}
    qrels = {str(qid): {str(docid): int(score) for docid, score in docs.items()} 
             for qid, docs in get_qrels(THE_TOPICS[pyserini_args.dataset]).items()}
    batch_topic_ids = []
    batch_topics = []
    for topic_id in list(topics.keys()):
        if topic_id in qrels:
            batch_topic_ids.append(topic_id)
            batch_topics.append(topics[topic_id])

    first_state_run_path = os.path.join(pyserini_args.output_dir, 'bm25.txt')
    if os.path.exists(first_state_run_path):
        print(f'Loading first stage run from {first_state_run_path}.')
        results = []
        with open(first_state_run_path, 'r') as f:
            current_qid = None
            current_ranking = []
            for line in f:
                qid, _, docid, _, score, _ = line.strip().split()
                if qid != current_qid:
                    if current_qid is not None:
                        results.append((current_qid, current_ranking[:pyserini_args.hits]))
                    current_ranking = []
                    current_qid = qid
                current_ranking.append(SearchResult(docid=docid, score=float(score), raw=searcher.doc(docid).raw()))
            results.append((current_qid, current_ranking[:pyserini_args.hits]))
    else:
        results = searcher.batch_search(
            batch_topics, batch_topic_ids, k=pyserini_args.hits, threads=pyserini_args.threads
        )
        results = [(id_, results[id_]) for id_ in batch_topic_ids]

        if pyserini_args.save_first_stage_run:
            out_path = os.path.join(pyserini_args.output_dir, 'bm25.txt')
            output_writer = get_output_writer(out_path, OutputFormat(pyserini_args.output_format), 'w',
                                              max_hits=pyserini_args.hits, tag='bm25', topics=topics, )
            write_run(output_writer, results, pyserini_args)
    # results = random.sample(results, 400)
    #####################################################
    # retrieve demonstration
    #####################################################
    qdid_few_shot_prompts = None
    # prepare auxiliary data source
    print('loading auxiliary data')
    with open(demon_args.id_demon_path, 'r') as f: 
        id_demon = json.load(f)
    id_query_path = f'../../data/ranking_trunc_queries_docs/id_query_{args.dataset}.json'
    with open(id_query_path, 'r') as f: 
        id_query = json.load(f)
    if args.dataset in ['dl19', 'dl20']:
        id_doc_path = f'../../data/ranking_trunc_queries_docs/id_doc_msmarco.json'
    else:
        id_doc_path = f'../../data/ranking_trunc_queries_docs/id_doc_{args.dataset}.json'
    with open(id_doc_path, 'r') as f:
        id_doc = json.load(f)
    if demon_args.demonstration_type != 'none':
        # msmarco_path = '../../data/ms_marco/passage_ranking/'
        # with open(os.path.join(msmarco_path, 'collection.json'), 'r') as f:
        #     id_doc = json.load(f)
        # id_doc = None
        # qrels_train = {}
        # with open(os.path.join(msmarco_path, 'qrels/qrels.train.tsv'), 'r') as f: #
        #     reader = csv.reader(f, delimiter='\t')
        #     for line in reader:
        #         qid, docid = line[0], line[2]
        #         if qid not in qrels_train:
        #             qrels_train[qid] = []
        #         qrels_train[qid].append(docid)
        # qrels_train = None
        with open(demon_args.id_demonquery_path, 'r') as f: 
            id_demonquery = json.load(f)
        with open(demon_args.id_demondoc_path, 'r') as f:
            id_demondoc = json.load(f)
        current_process_idx = accelerator.local_process_index
        total_process_num = accelerator.num_processes
        results_chunk = len(results) // total_process_num
        if current_process_idx + 1 != total_process_num:
            now_results = results[current_process_idx * results_chunk: (current_process_idx + 1) * results_chunk]
        else:
            now_results = results[current_process_idx * results_chunk:]

        if demon_args.demonstration_type == 'random':
            current_qdid_few_shot_prompts = get_random_prompts(now_results, id_demon, args) #lwh
        elif demon_args.demonstration_type == 'bm25':
            demon_searcher = LuceneSearcher(demon_args.demon_sparse_index_path)
            current_qdid_few_shot_prompts = get_few_shot_prompts(now_results, id_query, id_doc, topics, demon_searcher, tokenizer, args)
        elif demon_args.demonstration_type == 'dbs':
            current_qdid_few_shot_prompts = get_dbs_prompts(now_results, args)
        elif demon_args.demonstration_type == 'kmeans':
            current_qdid_few_shot_prompts = get_kmeans_prompts(now_results, args)
        elif demon_args.demonstration_type in ['sbert', 'e5', 'demor']:
            demon_searcher = MyFaissSearcher(index_dir=demon_args.demon_dense_index_path, query_encoder=demon_args.retriever_path, gpu_idx=device.index, device=device)
            current_qdid_few_shot_prompts = get_few_shot_prompts(now_results, id_query, id_doc, topics, demon_searcher, tokenizer, args)
        # save qdid_few_shot_prompts for each process
        with open(f'runs/qdid_few_shot_prompts_shard{current_process_idx}.json', 'w') as f: 
            json.dump(current_qdid_few_shot_prompts, f, indent=4)
        accelerator.wait_for_everyone()
        # merge
        qdid_few_shot_prompts = {}
        for i in range(accelerator.num_processes):
            with open(f'runs/qdid_few_shot_prompts_shard{i}.json', 'r') as f: 
                qdid_few_shot_prompts_shard = json.load(f)
                qdid_few_shot_prompts.update(qdid_few_shot_prompts_shard)
    ################################## rerank demonstrations ##################################
    # if args.shot_num != 1:  # for different shot exp
    #     with open(f'runs/qdid_few_shot_prompts_reranked_{args.dataset}.json', 'r') as f: 
    #         qdid_few_shot_prompts = json.load(f)
    #         for qdid in qdid_few_shot_prompts.keys():
    #             qdid_few_shot_prompts[qdid] = qdid_few_shot_prompts[qdid][:args.shot_num]
    # else:  # for different shot exp
    if args.demonstration_rerank:
        demon_reranker_tokenizer = AutoTokenizer.from_pretrained(args.demonstration_reranker_path, use_fast=True)
        demon_reranker = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.demonstration_reranker_path, num_labels=1, torch_dtype=torch.float16) # accelerate the bert inference process
        print(f'reranker loaded from {args.demonstration_reranker_path}')
        demon_reranker = demon_reranker.to(device).eval()
        rerank_dataset = DemonRerankDataset(id_query, id_doc, id_demonquery, id_demondoc, id_demon, topics, results, qdid_few_shot_prompts, demon_reranker_tokenizer, args)
        rerank_dataloader = DataLoader(rerank_dataset, batch_size=args.demonstration_reranker_batchsize, collate_fn=rerank_dataset.collate_fn)
        rerank_dataloader = accelerator.prepare(rerank_dataloader)
        rerank_scores = []

        with torch.no_grad():
            for batch_dict in tqdm(rerank_dataloader, desc=f'reranking demonstration', ncols=100):
                input_ids = batch_dict['input_ids'].to(device)
                attention_mask = batch_dict['attention_mask'].to(device)
                scores = demon_reranker(input_ids=input_ids, attention_mask=attention_mask).logits
                scores = scores.squeeze(-1)
                scores = accelerator.gather_for_metrics(scores)
                rerank_scores.extend(scores.tolist())
        start_idx = 0
        # qdid_few_shot_prompts_for_save = {} # for different shot exp
        for result in results:
            qid = result[0]
            for doc in result[1]:
                docid = doc.docid
                qdid = qid + '#' + docid
                demon_candidate_list = qdid_few_shot_prompts[qdid]
                current_scores = rerank_scores[start_idx: start_idx + len(demon_candidate_list)]
                reranked_candidates, reranked_scores = zip(*sorted(zip(demon_candidate_list, current_scores), key=lambda x: x[1], reverse=True))
                qdid_few_shot_prompts[qdid] = reranked_candidates[:args.shot_num]
                # qdid_few_shot_prompts_for_save[qdid] = reranked_candidates # for different shot exp
                start_idx += len(demon_candidate_list)
        assert(start_idx == len(rerank_scores))
    # if accelerator.local_process_index == 0: # for different shot exp
    #     with open(f'runs/qdid_few_shot_prompts_reranked_{args.dataset}.json', 'w') as f: 
    #         json.dump(qdid_few_shot_prompts_for_save, f, indent=4)


    ################################## LLM inference ##################################
    option_ids = [
        tokenizer.encode("Yes", add_special_tokens=False)[0],
        tokenizer.encode("No", add_special_tokens=False)[0]
    ]
    option_tensors = torch.tensor(option_ids, dtype=torch.long).to(device)
    # print('option_ids:', option_ids) # []
    # print(tokenizer.convert_ids_to_tokens(option_ids)) # 
    dataset = LLMDataset(results, id_query, id_doc, topics, pyserini_args, model_args, tokenizer, id_demon, qdid_few_shot_prompts)
    dataloader = DataLoader(dataset, batch_size=args.per_device_eval_batch_size)
    dataloader = accelerator.prepare(dataloader)
    scores = []
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), desc=f'Inferencing...', ncols=100):
            if 't5' in model_args.model_name_or_path:
                inputs = tokenizer(batch['sources'], return_tensors="pt", padding="longest", max_length=tokenizer.model_max_length, truncation=True,)
                input_ids=inputs['input_ids'].to(device)
                attention_mask=inputs['attention_mask'].to(device)
                labels = tokenizer(batch['targets'], return_tensors="pt", padding="longest", max_length=tokenizer.model_max_length, truncation=True,).input_ids
                labels[labels == tokenizer.pad_token_id] = -100
                labels = labels.to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits # bsz, seq_len, vocab_size
                logits = logits[:, 0, :]
            else: # for llama-based model
                prompts = []
                for system_content, user_content, source in zip(batch['systems'], batch['users'], batch['sources']):
                    messages = [
                        {"role": "system", "content": 'Answer the following question. Only output Yes or No without any other words'},
                        {"role": "user", "content": source},
                    ]
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    prompts.append(prompt)
                inputs = tokenizer(prompts, return_tensors="pt", padding="longest", max_length=tokenizer.model_max_length, truncation=True,)
                input_ids=inputs['input_ids'].to(device)
                attention_mask=inputs['attention_mask'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits # bsz, seq_len, vocab_size
                # print(logits.shape)
                logits = logits[:, -1, :]
            # print(logits.argmax(-1)) # llama: [8241, 3782] flan: [2163, 465]
            # print(tokenizer.convert_ids_to_tokens(logits.argmax(-1).tolist())) # llama: [Yes, No]  flan: [_Yes, _No]

            batch_size = logits.shape[0]
            option_tensor_index = option_tensors.unsqueeze(0).repeat(batch_size, 1)
            option_logits = torch.gather(logits, dim=1, index=option_tensor_index)
            option_logits = F.softmax(option_logits, dim=-1)
            scores_batch = option_logits[:, 0]
            scores_batch = accelerator.gather_for_metrics(scores_batch)
            scores.extend(scores_batch.tolist())
    if accelerator.process_index == 0:
        line_counter = 0

        for (topic_id, ranking) in results:
            for hit in ranking:
                hit.score = scores[line_counter]
                line_counter += 1
            # sort ranking by score
            ranking.sort(key=lambda x: x.score, reverse=True)

        out_path = os.path.join(pyserini_args.output_dir, pyserini_args.output)
        output_writer = get_output_writer(out_path, OutputFormat(pyserini_args.output_format), 'w',
                                        max_hits=pyserini_args.hits, tag='second_stage', topics=topics, )
        write_run(output_writer, results, pyserini_args)

        # -------------------- evaluate --------------------
        evaluate_results(args, out_path)

if __name__ == '__main__':
    main()