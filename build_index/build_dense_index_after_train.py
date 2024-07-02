import argparse
import csv
import json
import os
import random
import tempfile
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append("..")
from transformers import (
    AutoTokenizer,
    AutoModel,
)
import torch
import subprocess
from retrieve_utils import (
    construct_flatindex_from_embeddings, 
    index_retrieve, convert_index_to_gpu
)
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
import torch.nn.functional as F
from functools import partial
import faiss
from utils import clean_and_trunc_text
from index_and_topics import DOC_MAXLEN, QUERY_MAXLEN

os.environ["PYSERINI_CACHE"] = "../cache"


def add_prefix(example):
    return {'contents': 'query: ' + example['contents']}

def pool(last_hidden_states, attention_mask, pool_type: str):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

    if pool_type == "avg":
        emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pool_type == "cls":
        emb = last_hidden[:, 0]
    else:
        raise ValueError(f"pool_type {pool_type} not supported")
    return emb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--output_embedding_size", type=int, default=768)
    parser.add_argument("--model_path", type=str, default='/data/wenhan_liu/workspace/llm/e5-base-v2')
    parser.add_argument("--dataset", type=str, default='msmarco')
    parser.add_argument("--demonstration_pool_path", type=str, default='/data/wenhan_liu/workspace/project/demorank/data/for_index/query_1pos_1neg/collection/collection.jsonl')
    parser.add_argument("--output_dir", type=str, default='/data/wenhan_liu/workspace/project/demorank/data/for_index/query_1pos_1neg/e5-base-v2')
    parser.add_argument("--pool_type", type=str, default='avg')
    parser.add_argument("--add_prefix", type=bool, default=False, help='add query: for e5 model')

    args = parser.parse_args()
    args.doc_memmap_path = os.path.join(args.output_dir, "passages.memmap")
    args.docid_memmap_path = os.path.join(args.output_dir, "passages-id.memmap")
    args.my_docid_txt_path = os.path.join(args.output_dir, "my_passages-id.txt")
    args.index_path = os.path.join(args.output_dir, "index.idx")
    os.makedirs(args.output_dir, exist_ok=True)
    accelerator = Accelerator()
    device = accelerator.device
    dataset_id_demonquery_path = f'/home/u20238046/workspace_lwh/project/demorank/data/for_index/query_doc/{args.dataset}/id_demonquery.json'
    dataset_id_demondoc_path = f'/home/u20238046/workspace_lwh/project/demorank/data/for_index/query_doc/{args.dataset}/id_demondoc.json'
    id_demon_path = f'/home/u20238046/workspace_lwh/project/demorank/data/for_index/query_doc/{args.dataset}/id_demon.json'

    # with open(dataset_id_demonquery_path, 'r') as f:
    #     id_demonquery = json.load(f)
    # with open(dataset_id_demondoc_path, 'r') as f:
    #     id_demondoc = json.load(f)
    with open(id_demon_path, 'r') as f:
        id_demon = json.load(f)
    print(f'loading model from {args.model_path}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path, torch_dtype=torch.float16)
    model.to(device)
    model.eval()
    with accelerator.main_process_first():
        dataset = load_dataset('json', data_files=args.demonstration_pool_path, split='train')
        if args.add_prefix:
            print('add_prefix')
            dataset = dataset.map(add_prefix, num_proc=32)

    dataloader = DataLoader(dataset, batch_size=args.eval_batch_size)
    dataloader = accelerator.prepare(dataloader)

    # ------------------------- encode -------------------------
    if accelerator.local_process_index == 0:
        doc_memmap = np.memmap(args.doc_memmap_path, 
            dtype=np.float16, mode="w+", shape=(len(dataset), args.output_embedding_size))
        docid_memmap = np.memmap(args.docid_memmap_path,
            dtype=np.int32, mode="w+", shape=(len(dataset), ))
        my_docid = np.array(dataset['id'])
        np.savetxt(args.my_docid_txt_path, my_docid, fmt='%s')
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        write_index = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, total=len(dataloader), desc=f'Inferencing...', ncols=100):
                # ----------------- input is query,document pair -----------------
                # batch_queries, batch_docs, batch_labels, batch_inputs = [], [], [], []
                # for id in batch['id']:
                #     qid, docid, label = id.split('#')
                #     label = 'Yes' if label == '1' else 'No'
                #     batch_queries.append(id_demonquery[qid])
                #     batch_docs.append(id_demondoc[docid])
                #     batch_labels.append(label)
                #     batch_inputs.append(id_demonquery[qid] + ' [SEP] ' + id_demondoc[docid] + ' [SEP] ' + label)
                # inputs = tokenizer(batch_inputs,
                #                     max_length=args.max_input_length,
                #                     truncation=True,
                #                     padding=True,
                #                     return_token_type_ids=False,
                #                     pad_to_multiple_of=8,
                #                     return_tensors="pt")
                # input_ids=inputs['input_ids'].to(device)
                # attention_mask=inputs['attention_mask'].to(device)
                # outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                # ----------------- input is demonstration string -----------------
                batch_demons = [id_demon[demon_id] for demon_id in batch['id']]
                inputs = tokenizer(batch_demons,
                                    max_length=args.max_input_length,
                                    truncation=True,
                                    padding=True,
                                    return_token_type_ids=False,
                                    pad_to_multiple_of=8,
                                    return_tensors="pt")
                input_ids=inputs['input_ids'].to(device)
                attention_mask=inputs['attention_mask'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)


                embeds = pool(last_hidden_states=outputs.last_hidden_state,
                            attention_mask=attention_mask,
                            pool_type=args.pool_type) # avg pool
                embeds = F.normalize(embeds, dim=-1).contiguous()
                embeds = accelerator.gather_for_metrics(embeds)
                embeds = embeds.detach().cpu().numpy()   
                # batch_id = batch['id']    
                # batch_id = accelerator.gather_for_metrics(batch_id)
                if accelerator.local_process_index == 0:
                    write_size = len(embeds)
                    doc_memmap[write_index:write_index+write_size] = embeds
                    docid_memmap[write_index:write_index+write_size] = list(range(write_index, write_index+write_size))
                    write_index += write_size
            if accelerator.local_process_index == 0:
                assert write_index == len(doc_memmap) == len(docid_memmap)
    except:
        subprocess.check_call(["rm", args.doc_memmap_path])
        subprocess.check_call(["rm", args.docid_memmap_path])
        raise

    # ------------------------- build and save index -------------------------
    if accelerator.local_process_index == 0:
        doc_embeddings = np.memmap(args.doc_memmap_path,
            dtype=np.float16, mode="r")
        doc_ids = np.memmap(args.docid_memmap_path, 
            dtype=np.int32, mode="r")
        doc_embeddings = doc_embeddings.reshape(-1, args.output_embedding_size)
        # time1 = time.time()
        index = construct_flatindex_from_embeddings(doc_embeddings, doc_ids)
        faiss.write_index(index, args.index_path)
        # time2 = time.time()
        # print(time2 - time1)

