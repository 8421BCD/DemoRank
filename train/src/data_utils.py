import os
import random
import numpy as np

from typing import Dict, List
from collections import Counter
from datasets import load_dataset, Dataset, DownloadMode

from logger_config import logger
from util import save_dataset


def load_corpus(path: str) -> Dataset:
    assert path.endswith('.jsonl') or path.endswith('.jsonl.gz')

    # two fields: id, contents
    corpus = load_dataset('json', data_files=path)['train']
    logger.info('Load {} documents from {} with columns {}'.format(len(corpus), path, corpus.column_names))
    logger.info('A random document: {}'.format(random.choice(corpus)))
    return corpus

# sort the candidates based on their llm scores
def to_up_down_format(example):
    sorted_indices = np.argsort(example['scores'])[::-1]
    return {
        'demon_ids': [example['demon_ids'][idx] for idx in sorted_indices],
        'scores': [example['scores'][idx] for idx in sorted_indices]
    }

def sample_train_data(example: Dict, train_num: int) -> Dict:
    assert len(example['demon_ids']) == len(example['scores'])
    sampled_posidx = [0] # use top-1 as pos
    sampled_negidxs = random.sample(range(1, len(example['demon_ids'])), train_num - 1)
    sampled_idxs = sampled_posidx + sampled_negidxs
    sampled_demon_ids = [example['demon_ids'][idx] for idx in sampled_idxs]
    sampled_scores = [example['scores'][idx] for idx in sampled_idxs]
    sorted_indices: List[int] = np.argsort(sampled_scores)[::-1]
    assert sampled_scores[sorted_indices[0]] == example['scores'][0]
    return {
        'sampled_demon_ids': [sampled_demon_ids[idx] for idx in sorted_indices],
        'kd_labels': [sampled_scores[idx] for idx in sorted_indices]
    }

def to_positive_negative_format(example: Dict, topk_as_positive: int = 1, bottomk_as_negative: int = -1) -> Dict:
    # query_id / query / doc_ids / ndcg_10 / ndcg_50
    assert len(example['doc_ids']) == len(example['ndcg_50'])
    sorted_indices: List[int] = np.argsort(example['ndcg_50'])[::-1]
    positive_indices: List[int] = sorted_indices[:topk_as_positive]
    negative_indices: List[int] = sorted_indices[topk_as_positive:] if bottomk_as_negative <= 0 else sorted_indices[-bottomk_as_negative:]
    negative_indices = [idx for idx in negative_indices if idx not in positive_indices]
    np.random.shuffle(positive_indices)
    np.random.shuffle(negative_indices)
    return {
        'positives': {
            'doc_id': [example['doc_ids'][idx] for idx in positive_indices],
            'score': [example['ndcg_50'][idx] - example['none_ndcg_50'] for idx in positive_indices],
        },
        'negatives': {
            'doc_id': [example['doc_ids'][idx] for idx in negative_indices],
            'score': [example['ndcg_50'][idx] - example['none_ndcg_50'] for idx in negative_indices],
        },
    }

def to_positive_negative_format_new(example: Dict, topk_as_positive: int = 1, bottomk_as_negative: int = -1) -> Dict:
    assert len(example['demon_ids']) == len(example['scores'])
    sorted_indices: List[int] = np.argsort(example['scores'])[::-1]
    positive_indices: List[int] = sorted_indices[:topk_as_positive]
    negative_indices: List[int] = sorted_indices[topk_as_positive:] if bottomk_as_negative <= 0 else sorted_indices[-bottomk_as_negative:]
    negative_indices = [idx for idx in negative_indices if idx not in positive_indices]
    np.random.shuffle(positive_indices)
    np.random.shuffle(negative_indices)
    return {
        'positives': {
            'doc_id': [example['demon_ids'][idx] for idx in positive_indices],
            'score': [example['scores'][idx] for idx in positive_indices],
        },
        'negatives': {
            'doc_id': [example['demon_ids'][idx] for idx in negative_indices],
            'score': [example['scores'][idx] for idx in negative_indices],
        },
    }


def save_to_readable_format(in_path: str, corpus: Dataset, shuffle: bool = False, max_num_samples: int = 10000):
    out_path = '{}/readable_{}'.format(os.path.dirname(in_path), os.path.basename(in_path))
    out_path = out_path.replace('.jsonl.gz', '.json')
    dataset: Dataset = load_dataset('json', data_files=in_path, split='train', download_mode=DownloadMode.FORCE_REDOWNLOAD)
    if shuffle:
        dataset = dataset.shuffle()
    if len(dataset) > max_num_samples:
        dataset = dataset.select(range(max_num_samples))
    dataset = dataset.map(
        to_positive_negative_format,
        remove_columns=['doc_ids', 'ndcg_10', 'ndcg_50'],
        desc='to positive negative format'
    )

    max_to_keep = 5

    def _create_readable_field(samples: Dict[str, List]) -> List:
        readable_ex = []
        for idx in range(min(len(samples['doc_id']), max_to_keep)):
            doc_id = samples['doc_id'][idx]
            readable_ex.append({'doc_id': doc_id,
                                'contents': corpus[int(doc_id)]['contents'],
                                'score': samples['score'][idx],
                                'task_name': corpus[int(doc_id)]['task_name'],
                                })
        return readable_ex

    def _mp_func(ex: Dict) -> Dict:
        ex['positives'] = _create_readable_field(ex['positives'])
        ex['negatives'] = _create_readable_field(ex['negatives'])
        return ex
    dataset = dataset.map(_mp_func, desc='to readable format')

    dataset.to_json(out_path, force_ascii=False, lines=False, indent=4)
    logger.info('Done convert {} to readable format in {}'.format(in_path, out_path))


def save_llm_decoding_results(
        out_path: str,
        input_texts: List[str],
        decoded_texts: List[str],
        parsed_decoded_texts: List[str],
        options_list: List[List[str]],
        answer_texts: List[str]):
    assert len(input_texts) == len(decoded_texts)
    dataset = Dataset.from_dict({
        'input_text': input_texts,
        'decoded_text': decoded_texts,
        'parsed_decoded_text': parsed_decoded_texts,
        'options': options_list,
        'answer_text': answer_texts
    })
    save_dataset(dataset, out_path, shuffle=True)
    logger.info('Successfully save decoding results to {}'.format(out_path))


def log_task_statistics(ds: Dataset, split: str = 'train'):
    task_name_counter = Counter()
    for task_name in ds['task_name']:
        task_name_counter[task_name] += 1
    # log the number of examples per task
    for task_name, count in task_name_counter.most_common():
        logger.info('{} ({}): {}'.format(task_name, split, count))
    logger.info('{}: {} tasks, {} examples in total'.format(split, len(task_name_counter), len(ds)))
