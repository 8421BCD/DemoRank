import torch

from dataclasses import dataclass
from typing import List, Dict, Any, Union, Optional
from transformers import BatchEncoding, PreTrainedTokenizerBase
from config import Arguments
from transformers.file_utils import PaddingStrategy



@dataclass
class CrossEncoderCollator:

    args: Arguments
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> BatchEncoding:
        ########################## using input str and demonstration str ##########################
        train_inputs: List[str] = [f['train_inputs'] for f in features]
        train_demons: List[str] = sum([f['train_demons'] for f in features], [])
        step_size = self.args.train_n_passages
        # input_texts = []
        # for i in range(len(train_demons)):
        #     input_texts.append(train_inputs[i // step_size] + ' [SEP] ' + '\n\n'.join(train_demons[i]))

        inputs, demons = [], []
        for i in range(len(train_demons)):
            inputs.append(train_inputs[i // step_size])
            demons.append(train_demons[i])

        merged_batch_dict = self.tokenizer(
            inputs,
            text_pair=demons,
            max_length=self.args.max_len,
            truncation=True,
            padding=self.padding,
            return_token_type_ids=False,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors)
        # dummy placeholder for field "labels", won't use it to compute loss
        labels = torch.zeros(len(train_inputs), dtype=torch.long)
        merged_batch_dict['labels'] = labels
        if 'kd_labels' in features[0]:
            kd_labels = torch.stack([torch.tensor(f['kd_labels']) for f in features], dim=0).float()
            merged_batch_dict['kd_labels'] = kd_labels
        return merged_batch_dict
    
        ########################## using query doc str and demon_query demon_doc demon_label str ##########################
        # train_queries = [f['train_queries'] for f in features]
        # train_docs = [f['train_docs'] for f in features]
        # demon_queries = sum([f['demon_queries'] for f in features], [])
        # demon_docs = sum([f['demon_docs'] for f in features], [])
        # demon_labels = sum([f['demon_labels'] for f in features], [])
        # step_size = self.args.train_n_passages
        # inputs = []
        # for i in range(len(demon_queries)):
        #     inputs.append(train_queries[i // step_size] + ' [ISEP] ' + train_docs[i // step_size] + ' [SEP] ' + demon_queries[i] + ' [DSEP] ' + demon_docs[i] + ' [DSEP] ' + demon_labels[i])
        
        # merged_batch_dict = self.tokenizer(
        #                             inputs,
        #                             max_length=self.args.max_len,
        #                             truncation=True,
        #                             padding=self.padding,
        #                             return_token_type_ids=False,
        #                             pad_to_multiple_of=self.pad_to_multiple_of,
        #                             return_tensors=self.return_tensors)

        # labels = torch.zeros(len(train_queries), dtype=torch.long)
        # merged_batch_dict['labels'] = labels
        # if 'kd_labels' in features[0]:
        #     kd_labels = torch.stack([torch.tensor(f['kd_labels']) for f in features], dim=0).float()
        #     merged_batch_dict['kd_labels'] = kd_labels
        # return merged_batch_dict

