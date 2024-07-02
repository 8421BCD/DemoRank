import torch

from dataclasses import dataclass
from typing import List, Dict, Any, Union, Optional
from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy

from config import Arguments


@dataclass
class BiencoderCollator:

    args: Arguments
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> BatchEncoding:
        ########################## using input str and demonstration str ##########################
        train_inputs: List[str] = [f['train_inputs'] for f in features]
        demons: List[str] = sum([f['train_demons'] for f in features], [])
        input_texts = train_inputs + demons
        merged_batch_dict = self.tokenizer(
            input_texts,
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
        # if 'train_demons' in features[0]:
        #     train_demons = [f['train_demons'] for f in features]
        # train_inputs = []
        # for i in range(len(train_queries)):
        #     input_str = ''
        #     if 'train_demons' in features[0]:
        #         for demon in train_demons[i][::-1]:
        #             input_str += demon + ' [SEP] '
        #     input_str += train_queries[i] + ' [SEP] ' + train_docs[i]
        #     train_inputs.append(input_str)

        # demon_queries = sum([f['demon_queries'] for f in features], [])
        # demon_docs = sum([f['demon_docs'] for f in features], [])
        # demon_labels = sum([f['demon_labels'] for f in features], [])
        # demon_inputs = []
        # for i in range(len(demon_queries)):
        #     demon_inputs.append(demon_queries[i] + ' [SEP] ' + demon_docs[i] + ' [SEP] ' + demon_labels[i])
        
        # merged_batch_dict = self.tokenizer(
        #     train_inputs + demon_inputs,
        #     max_length=self.args.max_len,
        #     truncation=True,
        #     padding=self.padding,
        #     return_token_type_ids=False,
        #     pad_to_multiple_of=self.pad_to_multiple_of,
        #     return_tensors=self.return_tensors)
        # # dummy placeholder for field "labels", won't use it to compute loss
        # labels = torch.zeros(len(train_queries), dtype=torch.long)
        # merged_batch_dict['labels'] = labels
        # if 'kd_labels' in features[0]:
        #     kd_labels = torch.stack([torch.tensor(f['kd_labels']) for f in features], dim=0).float()
        #     merged_batch_dict['kd_labels'] = kd_labels
        # return merged_batch_dict

@dataclass
class BiencoderCollator_feature:

    args: Arguments
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> BatchEncoding:
        # ----------- input is the whole demonstration -----------
        # train_inputs: List[str] = [f['train_inputs'] for f in features]
        # demons: List[str] = sum([f['demons'] for f in features], [])
        # input_texts = train_inputs + demons
        # merged_batch_dict = self.tokenizer(
        #     input_texts,
        #     max_length=self.args.max_len,
        #     truncation=True,
        #     padding=self.padding,
        #     return_token_type_ids=False,
        #     pad_to_multiple_of=self.pad_to_multiple_of,
        #     return_tensors=self.return_tensors)
        # # dummy placeholder for field "labels", won't use it to compute loss
        # labels = torch.zeros(len(train_inputs), dtype=torch.long)
        # merged_batch_dict['labels'] = labels

        # ----------- input is query (for ndcg_score) -----------
        # train_queries = [f['train_queries'] for f in features]
        # train_docs = [f['train_docs'] for f in features]
        # train_inputs = []
        # for i in range(len(train_queries)):
        #     train_inputs.append(' [SEP] '.join([train_queries[i]] + train_docs[i]))
        
        # ----------- input is query+doc -----------
        train_queries = [f['train_queries'] for f in features]
        train_docs = [f['train_docs'] for f in features]
        train_features = torch.stack([torch.tensor(f['train_features']) for f in features], dim=0).float()


        if 'train_demons' in features[0]:
            train_demons = [f['train_demons'] for f in features]
        train_inputs = []
        for i in range(len(train_queries)):
            input_str = ''
            if 'train_demons' in features[0]:
                for demon in train_demons[i][::-1]:
                    input_str += demon + ' [SEP] '
            input_str += train_queries[i] + ' [SEP] ' + train_docs[i]
            train_inputs.append(input_str)

        demon_queries = sum([f['demon_queries'] for f in features], [])
        demon_docs = sum([f['demon_docs'] for f in features], [])
        demon_labels = sum([f['demon_labels'] for f in features], [])
        demon_features = sum([f['demon_features'] for f in features], [])
        demon_features = torch.tensor(demon_features, dtype=torch.float)
        
        demon_inputs = []
        for i in range(len(demon_queries)):
            demon_inputs.append(demon_queries[i] + ' [SEP] ' + demon_docs[i] + ' [SEP] ' + demon_labels[i])
        
        merged_batch_dict = self.tokenizer(
            train_inputs + demon_inputs,
            max_length=self.args.max_len,
            truncation=True,
            padding=self.padding,
            return_token_type_ids=False,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors)
        # dummy placeholder for field "labels", won't use it to compute loss
        labels = torch.zeros(len(train_queries), dtype=torch.long)
        merged_batch_dict['labels'] = labels
        merged_batch_dict['train_features'] = train_features
        merged_batch_dict['demon_features'] = demon_features

        if 'kd_labels' in features[0]:
            kd_labels = torch.stack([torch.tensor(f['kd_labels']) for f in features], dim=0).float()
            merged_batch_dict['kd_labels'] = kd_labels
        return merged_batch_dict
