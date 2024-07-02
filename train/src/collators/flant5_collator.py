import torch

from dataclasses import dataclass
from typing import List, Dict, Any, Union, Optional
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy
from transformers import T5Tokenizer, TrainingArguments

from logger_config import logger

@dataclass
class FlanT5Collator:
    tokenizer: T5Tokenizer
    args: TrainingArguments
    max_length: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features):
        self.tokenizer.padding_side = 'right'
        input_texts = [f['input'] for f in features]
        output_texts = [f['output'] for f in features]

        input_dict = self.tokenizer(input_texts, max_length = self.max_length, padding = True, truncation = True, return_tensors='pt').to(self.args.train_device)
        output_dict = self.tokenizer(output_texts, max_length = self.max_length, padding = True, truncation = True, return_tensors='pt').to(self.args.train_device)
        labels = output_dict['input_ids']
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        data_dict = {
            'input_ids': input_dict['input_ids'],
            'attention_mask': input_dict['attention_mask'],
            'labels': labels
        }
        return data_dict


