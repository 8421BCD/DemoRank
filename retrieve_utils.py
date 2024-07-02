import argparse
import csv
import json
import sys

sys.path += ['./']
import os
import faiss
import numpy as np
from tqdm import tqdm
from timeit import default_timer as timer
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModel,
)
from typing import Dict, List, Union, Optional, Tuple
import torch.nn.functional as F
from dataclasses import dataclass
import torch.nn as nn

import torch

def index_retrieve(index, query_embeddings, topk, batch=None):
    print("Query Num", len(query_embeddings))
    start = timer()
    if batch is None:
        _, nearest_neighbors = index.search(query_embeddings, topk)
    else:
        query_offset_base = 0
        pbar = tqdm(total=len(query_embeddings))
        nearest_neighbors = []
        while query_offset_base < len(query_embeddings):
            batch_query_embeddings = query_embeddings[query_offset_base:query_offset_base+ batch]
            batch_nn = index.search(batch_query_embeddings, topk)[1]
            nearest_neighbors.extend(batch_nn.tolist())
            query_offset_base += len(batch_query_embeddings)
            pbar.update(len(batch_query_embeddings))
        pbar.close()

    elapsed_time = timer() - start
    elapsed_time_per_query = 1000 * elapsed_time / len(query_embeddings)
    print(f"Elapsed Time: {elapsed_time:.1f}s, Elapsed Time per query: {elapsed_time_per_query:.1f}ms")
    return nearest_neighbors



def construct_flatindex_from_embeddings(embeddings, ids=None):
    dim = embeddings.shape[1]
    # print('embedding shape: ' + str(embeddings.shape))
    index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    if ids is not None:
        ids = ids.astype(np.int64)
        # print(ids.shape, ids.dtype)
        index = faiss.IndexIDMap2(index)
        index.add_with_ids(embeddings, ids)
    else:
        index.add(embeddings)
    return index


gpu_resources = []

def convert_index_to_gpu(index, faiss_gpu_index, useFloat16=False):
    if type(faiss_gpu_index) == list and len(faiss_gpu_index) == 1:
        faiss_gpu_index = faiss_gpu_index[0]
    if isinstance(faiss_gpu_index, int):
        res = faiss.StandardGpuResources()
        res.setTempMemory(512*1024*1024)
        co = faiss.GpuClonerOptions()
        co.useFloat16 = useFloat16
        index = faiss.index_cpu_to_gpu(res, faiss_gpu_index, index, co)
    else:
        global gpu_resources
        if len(gpu_resources) == 0:
            import torch
            for i in range(torch.cuda.device_count()):
                res = faiss.StandardGpuResources()
                res.setTempMemory(256*1024*1024)
                gpu_resources.append(res)

        assert isinstance(faiss_gpu_index, list)
        vres = faiss.GpuResourcesVector()
        vdev = faiss.IntVector()
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = useFloat16
        for i in faiss_gpu_index:
            vdev.push_back(i)
            vres.push_back(gpu_resources[i])
        index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)

    return index

def pool(last_hidden_states, attention_mask, pool_type: str):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

    if pool_type == "avg":
        emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pool_type == "cls":
        emb = last_hidden[:, 0]
    else:
        raise ValueError(f"pool_type {pool_type} not supported")
    return emb

@dataclass
class DenseSearchResult:
    docid: str
    score: float

class MyFaissSearcher:
    """using current input to retrieve demonstration"""

    def __init__(self, index_dir, query_encoder, gpu_idx=None, device='cuda:0'):
        self.device = device
        self.query_encoder = AutoModel.from_pretrained(query_encoder, torch_dtype=torch.float16)
        print(f'retriever loaded from {query_encoder}')
        self.query_encoder.to(self.device)
        self.query_encoder.eval()
        self.query_tokenizer = AutoTokenizer.from_pretrained(query_encoder)
        self.index, self.docids = self.load_index(index_dir, gpu_idx)


    def load_index(self, index_dir, gpu_idx=None):
        index_path = os.path.join(index_dir, 'index.idx')
        docid_path = os.path.join(index_dir, 'my_passages-id.txt')
        index = faiss.read_index(index_path)
        if gpu_idx is not None:
            index = convert_index_to_gpu(index, gpu_idx, False)
        else:
            faiss.omp_set_num_threads(32)
        docids = self.load_docids(docid_path)
        return index, docids
    
    @staticmethod
    def load_docids(docid_path: str) -> List[str]:
        id_f = open(docid_path, 'r')
        docids = [line.rstrip() for line in id_f.readlines()]
        id_f.close()
        return docids
    
    def encode_query(self, inputs):
        with torch.no_grad():
            inputs_tokenized = self.query_tokenizer(inputs,
                                          max_length=512, 
                                          truncation=True, 
                                          padding=True, 
                                          return_token_type_ids=False,
                                          pad_to_multiple_of=8,
                                          return_tensors="pt")
            input_ids=inputs_tokenized['input_ids'].to(self.device)
            attention_mask=inputs_tokenized['attention_mask'].to(self.device)
            outputs = self.query_encoder(input_ids=input_ids, attention_mask=attention_mask)
            embeds = pool(last_hidden_states=outputs.last_hidden_state,
                        attention_mask=attention_mask,
                        pool_type='avg') # avg pool
            embeds = F.normalize(embeds, dim=-1).contiguous()
        return embeds

    def search(self, inputs, k: int = 10) -> List[DenseSearchResult]:
        """Search the collection.

        Parameters
        ----------
        inputs : Union[str, np.ndarray]
            input text or input embeddings
        k : int
            Number of hits to return.
        Returns
        -------
            List[DenseSearchResult]
        """
        if isinstance(inputs, str):
            inputs = self.encode_query([inputs])
        distances, indexes = self.index.search(inputs.cpu().numpy(), k)
        distances = distances.flat
        indexes = indexes.flat
        return [DenseSearchResult(self.docids[idx], score)
                for score, idx in zip(distances, indexes) if idx != -1]

    def batch_search(self, inputs, inputids: List[str], k: int = 10, batchsize: int = 32) \
            -> Union[Dict[str, List[DenseSearchResult]], Tuple[np.ndarray, Dict[str, List[DenseSearchResult]]]]:
        """

        Parameters
        ----------
        queries : Union[List[str], np.ndarray]
            List of query texts or list of query embeddings
        inputids : List[str]
            List of corresponding query ids.
        k : int
            Number of hits to return.
        batchsize : int
            the size of batch to process.

        Returns
        -------
        Union[Dict[str, List[DenseSearchResult]], Tuple[np.ndarray, Dict[str, List[PRFDenseSearchResult]]]]
            Either returns a dictionary holding the search results, with the query ids as keys and the
            corresponding lists of search results as the values.
            Or returns a tuple with ndarray of query vectors and a dictionary of PRF Dense Search Results with vectors
        """
        qdid_hits = {}
        for start in range(0, len(inputs), batchsize):
            batch_inputs = inputs[start: start + batchsize]
            batch_inputids = inputids[start: start + batchsize]
            emb_q = self.encode_query(batch_inputs)
            batch_distances, batch_indexes = self.index.search(emb_q.cpu().numpy(), k)
            for inputid, distances, indexes in zip(batch_inputids, batch_distances, batch_indexes):
                qdid_hits[inputid] = [DenseSearchResult(self.docids[idx], score)
                                 for score, idx in zip(distances, indexes) if idx != -1]
        return qdid_hits
