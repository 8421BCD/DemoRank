import json
import os.path
import random
import torch

from copy import deepcopy
from functools import partial
from typing import Dict, List, Optional
from datasets import load_dataset, Dataset
from transformers.file_utils import PaddingStrategy
from transformers import PreTrainedTokenizerFast, Trainer

from config import Arguments
from logger_config import logger
from util import get_input_files
from .loader_utils import group_doc_ids, filter_invalid_examples
from data_utils import sample_train_data, to_positive_negative_format, to_positive_negative_format_new


class CrossEncoderDataset(torch.utils.data.Dataset):

    def __init__(self, input_files: List[str], args: Arguments,
                 tokenizer: PreTrainedTokenizerFast):
        self.args = args
        self.input_files = input_files
        self.tokenizer = tokenizer
        self.id_demonquery = {}
        self.id_demondoc = {}
        self.id_demon = {}
        self.datasets = []
        
        if args.uni_dataset: # all msmarco+beir to train
            for file in os.listdir(args.uni_dataset_path):
                self.datasets.append(file.split('_')[0])
        else:
            dataset = os.path.basename(args.train_data_shards_file).split('_')[0]
            self.datasets.append(dataset)
        for dataset in self.datasets:
            demonquery_path = f'/home/u20238046/workspace_lwh/project/demorank/data/for_index/query_doc/{dataset}/id_demonquery.json'
            demondoc_path = f'/home/u20238046/workspace_lwh/project/demorank/data/for_index/query_doc/{dataset}/id_demondoc.json'
            demon_path = f'/home/u20238046/workspace_lwh/project/demorank/data/for_index/query_doc/{dataset}/id_demon.json'
            with open(demonquery_path, 'r') as f:
                self.id_demonquery[dataset] = json.load(f)
            with open(demondoc_path, 'r') as f:
                self.id_demondoc[dataset] = json.load(f)
            with open(demon_path, 'r') as f: 
                self.id_demon[dataset] = json.load(f)

        with self.args.main_process_first(desc="pre-processing"):
            self.dataset: Dataset = load_dataset('json', data_files=self.input_files, split='train')
            print('dataset loaded')
            # self.dataset = filter_invalid_examples(args, self.dataset)
            # -------------------- map code --------------------
            # self.dataset = self.dataset.map(
            #     partial(sample_train_data,
            #             train_num = args.train_n_passages),
            #     load_from_cache_file=args.world_size > 1,
            #     desc='sampling data',
            #     remove_columns=['demon_ids', 'scores', 'none_score'],
            # )
        self.dataset.set_transform(self._transform_func_qd)
        # use its state to decide which positives/negatives to sample
        self.trainer: Optional[Trainer] = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def _transform_func_qd(self, examples: Dict[str, List]) -> Dict[str, List]:
        examples = deepcopy(examples)
        
        ########################## using input str and demonstration str ##########################
        train_inputs = examples['train_input']
        train_demons = []
        for i in range(len(examples['demon_ids'])):
            dataset = examples['dataset'][i]
            for demonid in examples['demon_ids'][i]:
                train_demons.append(self.id_demon[dataset][demonid])
            # for train_nshot_demon in examples['demon_ids'][i]:
            #     train_demons.append([self.id_demon[dataset][train_demon] for train_demon in train_nshot_demon])

        assert(len(train_demons) == self.args.train_n_passages * len(train_inputs))
        step_size = self.args.train_n_passages
        batch_dict = {
            'train_inputs': train_inputs,
            'train_demons': [train_demons[idx:idx + step_size] for idx in range(0, len(train_demons), step_size)],
            'kd_labels': examples['scores']
        }


        ########################## using query doc str and demon_query demon_doc demon_label str ##########################
        # train_queries, train_docs = [], []
        # for dataset, train_input_id in zip(examples['dataset'], examples['train_input_id']):
        #     qid, docid, label = train_input_id.split('#')
        #     train_queries.append(self.id_demonquery[dataset][qid])
        #     train_docs.append(self.id_demondoc[dataset][docid])
        # # if 'selected_demonids' in examples: # for k-shot training
        # #     train_demons = []
        # #     for dataset, selected_demonids in zip(examples['dataset'], examples['selected_demonids']):
        # #         train_demons.append([self.id_demon[dataset][demonid] for demonid in selected_demonids])

        # demon_queries, demon_docs, demon_labels = [], [], []
        # for i in range(len(examples['demon_ids'])):
        #     dataset = examples['dataset'][i]
        #     for train_demon_id in examples['demon_ids'][i]:
        #         demon_qid, demon_docid, demon_label = train_demon_id.split('#')
        #         demon_queries.append(self.id_demonquery[dataset][demon_qid])
        #         demon_docs.append(self.id_demondoc[dataset][demon_docid])
        #         demon_labels.append('Yes' if demon_label == '1' else 'No')
        # assert(len(demon_queries) == self.args.train_n_passages * len(train_queries))
        # step_size = self.args.train_n_passages
        # batch_dict = {
        #     'train_queries': train_queries,
        #     'train_docs': train_docs,
        #     'demon_queries': [demon_queries[idx:idx + step_size] for idx in range(0, len(demon_queries), step_size)],
        #     'demon_docs': [demon_docs[idx:idx + step_size] for idx in range(0, len(demon_docs), step_size)],
        #     'demon_labels': [demon_labels[idx:idx + step_size] for idx in range(0, len(demon_labels), step_size)],
        #     'kd_labels': examples['scores']
        # }
        # # if 'selected_demonids' in examples: # for k-shot training
        # #     batch_dict['train_demons'] = train_demons
        
        # assert len(batch_dict['kd_labels']) == len(batch_dict['train_queries'])
        # assert len(batch_dict['kd_labels']) == len(batch_dict['demon_queries'])


        return batch_dict

class CrossEncoderDataLoader:

    def __init__(self, args: Arguments, tokenizer: PreTrainedTokenizerFast):
        self.args = args
        self.tokenizer = tokenizer
        self.train_dataset = self._get_transformed_datasets()

    def set_trainer(self, trainer: Trainer):
        if self.train_dataset is not None:
            self.train_dataset.trainer = trainer

    def _get_transformed_datasets(self) -> CrossEncoderDataset:
        train_dataset = None

        if self.args.train_file is not None:
            # train_input_files = get_input_files(self.args.train_file)
            # logger.info("Train files: {}".format(train_input_files))
            train_dataset = CrossEncoderDataset(
                args=self.args,
                tokenizer=self.tokenizer,
                input_files=self.args.train_file,
            )

        if self.args.do_train:
            assert train_dataset is not None, "Training requires a train dataset"

        return train_dataset
