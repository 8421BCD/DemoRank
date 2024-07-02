import copy
import json
import logging
import os
import random

import numpy as np

from transformers.utils.logging import enable_explicit_format, set_verbosity_info, set_verbosity_warning
from transformers.trainer_callback import PrinterCallback
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    set_seed,
    PreTrainedTokenizerFast
)
from logger_config import logger, LoggerCallback
from config import Arguments
from trainers.crossencoder_trainer import CrossencoderTrainer
from loaders import CrossEncoderDataLoader
from collators import CrossEncoderCollator
from models import Reranker
import torch

def _common_setup(args: Arguments):
    # set_verbosity_info()
    # if args.process_index > 0:
    #     logger.setLevel(logging.WARNING)
    #     set_verbosity_warning()
    # enable_explicit_format()
    set_seed(args.seed)

def make_train_data(args):
    all_train_data = []
    if args.uni_dataset: # whether to use msmarco+fever+nq+hotpotqa to train (in this paper, uni_dataset is always set as False)
        for file in os.listdir(args.uni_dataset_path):
            dataset = file.split('_')[0]
            # if dataset != 'nfcorpus':
            #     continue
            file_path = os.path.join(args.uni_dataset_path, file)
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    if data['selected_demonids'] != []:
                        continue
                    sorted_idxs = np.argsort(data['scores'])[::-1]
                    scores = [data['scores'][idx] for idx in sorted_idxs]
                    demon_ids = [data['demon_ids'][idx] for idx in sorted_idxs]
                    assert(len(scores) == 100)
                    data['scores'] = scores
                    data['demon_ids'] = demon_ids
                    data['dataset'] = dataset
                    data['selected_demonids'] = []
                    all_train_data.append(data)
        random.shuffle(all_train_data)
    else:
        dataset = os.path.basename(args.train_data_shards_file).split('_')[0]
        with open(args.train_data_shards_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                # greedy data
                if len(data['selected_demonids']) == 0:
                    continue
                demon_ids = data['selected_demonids'] + data['unselected_demonids']
                scores = list(range(len(data['selected_demonids']), 0, -1)) + [0] * len(data['unselected_demonids'])
                if len(demon_ids) != args.train_n_passages:
                    continue
                data['demon_ids'] = demon_ids
                data['scores'] = scores
                data['dataset'] = dataset
                all_train_data.append(data)

                # no greedy data
                # if len(data['demon_ids']) != args.train_n_passages:
                #     continue
                # sorted_idxs = np.argsort(data['scores'])[::-1]
                # # scores = [data['scores'][idx] for idx in sorted_idxs]
                # scores = list(range(len(data['demon_ids']), 0, -1))
                # demon_ids = [data['demon_ids'][idx] for idx in sorted_idxs]
                # data['scores'] = scores
                # data['demon_ids'] = demon_ids
                # data['dataset'] = dataset
                # all_train_data.append(data)

                # if len(all_train_data) == 5000: #lwh
                #     break
    # all_train_data = all_train_data[:50000] #lwh
    with open(args.train_file, 'w') as f: 
        for data in all_train_data:
            f.write(json.dumps(data) + '\n')
        print(f'all data written! total: {len(all_train_data)}')

def main():
    parser = HfArgumentParser((Arguments,))
    args: Arguments = parser.parse_args_into_dataclasses()[0]
    _common_setup(args)
    # logger.info('Args={}'.format(str(args)))

    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model: Reranker = Reranker.from_pretrained(
        all_args=args,
        pretrained_model_name_or_path=args.model_name_or_path,
        num_labels=1)
    model.hf_model.resize_token_embeddings(len(tokenizer))
    # logger.info(model)
    # logger.info('Vocab size: {}'.format(len(tokenizer)))

    data_collator = CrossEncoderCollator(
        args=args,
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if args.fp16 else None)
    if torch.distributed.get_rank() == 0:            
        make_train_data(args)

    reward_data_loader = CrossEncoderDataLoader(args=args, tokenizer=tokenizer)
    train_dataset = reward_data_loader.train_dataset # each query correspond to several passages (pos, neg1, neg2...)   lwh

    trainer: Trainer = CrossencoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset if args.do_train else None,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(LoggerCallback)
    reward_data_loader.trainer = trainer

    if args.do_train:
        train_result = trainer.train()
        trainer.save_model()

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    return


if __name__ == "__main__":
    main()
