import random
from trainers import BiencoderTrainer
import json
import logging
from transformers.utils.logging import enable_explicit_format, set_verbosity_info, set_verbosity_warning, enable_progress_bar
from transformers.trainer_callback import PrinterCallback
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    set_seed,
    PreTrainedTokenizerFast
)
import os
from logger_config import logger, LoggerCallback
from config import Arguments
from loaders import RetrievalDataLoader
from collators import BiencoderCollator
from models import BiencoderModel, BiencoderModel_new
import numpy as np
import torch

def _common_setup(args: Arguments):
    enable_progress_bar()
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
            # if dataset != 'msmarco':
            #     continue
            file_path = os.path.join(args.uni_dataset_path, file)
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    sorted_idxs = np.argsort(data['scores'])[::-1]
                    scores = [data['scores'][idx] for idx in sorted_idxs]
                    demon_ids = [data['demon_ids'][idx] for idx in sorted_idxs]
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
                assert len(data['scores']) == 50 and data['selected_demonids'] == []
                # if data['selected_demonids'] != []:
                #     continue
                if len(data['demon_ids']) != args.train_n_passages:
                    continue
                sorted_idxs = np.argsort(data['scores'])[::-1] # sort based on scores
                scores = [data['scores'][idx] for idx in sorted_idxs]
                demon_ids = [data['demon_ids'][idx] for idx in sorted_idxs]
                data['scores'] = scores
                data['demon_ids'] = demon_ids
                data['dataset'] = dataset
                all_train_data.append(data)

    with open(args.train_file, 'w') as f: 
        for data in all_train_data:
            f.write(json.dumps(data) + '\n')
        print(f'all data written! total: {len(all_train_data)}')

def main():
    parser = HfArgumentParser((Arguments,))
    args: Arguments = parser.parse_args_into_dataclasses()[0]
    _common_setup(args)
    # logger.info('Args={}'.format(str(args)))
    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = BiencoderModel_new.build(args=args)
    # logger.info(model)
    # logger.info('Vocab size: {}'.format(len(tokenizer)))

    data_collator = BiencoderCollator(
        args=args,
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if args.fp16 else None)
    
    if torch.distributed.get_rank() == 0:            
        make_train_data(args)

    retrieval_data_loader = RetrievalDataLoader(args=args, tokenizer=tokenizer)
    train_dataset = retrieval_data_loader.train_dataset
    trainer: Trainer = BiencoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset if args.do_train else None,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.remove_callback(PrinterCallback)
    trainer.add_callback(LoggerCallback)
    retrieval_data_loader.set_trainer(trainer)
    model.trainer = trainer

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
