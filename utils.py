import numpy as np
import os
import random
import torch
from prompts import DOC_FORMAT_DIC, PROMPT_DICT, yes_no_system_instruction, PROMPT_DICT_YES_NO
from typing import List, Union, Optional, Tuple, Mapping, Dict
import gc

def get_pointwise_prompt(query, doc, label):
    return 'Query:{}\nPassage:{}\nIs the Passage relevant to the Query?\nOutput:{}\n'.format(query, doc, label)

def convert_id_to_demon(demonid, id_demonquery, id_demondoc, index, model_name='flan-t5-xl'):
    qid, docid, label = demonid.split('#')
    demonstration = PROMPT_DICT_YES_NO[index][model_name].format_map({'qry': id_demonquery[qid], 'doc': id_demondoc[docid]}) + ('Yes' if label == '1' else 'No')
    return demonstration

def get_qg_demonstration(doc, query, tokenizer, doc_max_length, query_max_length, index, model_name_or_path):
    demonstration_str = ''
    doc = tokenizer.convert_tokens_to_string(tokenizer.tokenize(doc)[:doc_max_length])
    input_text = PROMPT_DICT[index][model_name_or_path].format_map({"doc": doc})
    demonstration_str += input_text + "\n"
    query = tokenizer.convert_tokens_to_string(tokenizer.tokenize(query)[:query_max_length])
    demonstration_str += query + '\n'
    return demonstration_str

def get_yes_no_prompt(dataset, query, doc, demonstrations=None, llm_name='flan-t5-xl'):
    prompt = []
    if yes_no_system_instruction[dataset] != '':
        prompt.append(yes_no_system_instruction[dataset])
    if demonstrations:
        for demonstration in demonstrations:
            prompt.append(demonstration)
    
    prompt.append(PROMPT_DICT_YES_NO[dataset][llm_name].format_map({'qry': query, 'doc': doc}))
    prompt = '\n\n'.join(prompt)
    return prompt

# for llama3
def get_yes_no_prompt_system_user(dataset, query, doc, demonstrations=None, llm_name='flan-t5-xl'):
    system = [yes_no_system_instruction[dataset]]
    # user = []
    if demonstrations:
        system[0] += f'Here are {len(demonstrations)}demonstrations:'
        for demonstration in demonstrations:
            system.append(demonstration)
    system = '\n\n'.join(system)
    user = PROMPT_DICT_YES_NO[dataset][llm_name].format_map({'qry': query, 'doc': doc})
    # user.append(PROMPT_DICT_YES_NO[dataset][llm_name].format_map({'qry': query, 'doc': doc}))
    # user = '\n\n'.join(user)
    return system, user


def get_yes_no_prompt_for_qd(dataset, qd, demonstrations=None):
    prompt = []
    if yes_no_system_instruction[dataset] != '':
        prompt.append(yes_no_system_instruction[dataset])
    if demonstrations:
        for demonstration in demonstrations:
            prompt.append(demonstration)
    prompt.append(qd)
    prompt = '\n\n'.join(prompt)
    return prompt

def set_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def delete_object(obj):
    del obj
    torch.cuda.empty_cache()  # 清空 GPU 缓存
    gc.collect()  # 执行垃圾回收

def clean_and_trunc_text(tokenizer, text, max_len):
    text = text.replace('\n\n', '') # in case of the noise for ICL
    text = tokenizer.convert_tokens_to_string(tokenizer.tokenize(text)[:max_len])
    # text = ' '.join(text.split(' ')[:max_len])
    return text

def mmr(qd_embed, demon_embeds, return_number, mmr_lambda=0.5):
    """
        qd_embed: tensor, [768]
        demon_embeds: tensor, [demon_num, 768]
        return_number: int
    """
    qd_embed_expanded = qd_embed.unsqueeze(0) # [1, 768]
    qd_demon_scores = torch.matmul(qd_embed_expanded, demon_embeds.T).squeeze(0) # [demon_num]
    qd_demon_scores = qd_demon_scores.cpu().numpy()
    # print(f'qd_demon_scores.shape: {qd_demon_scores.shape}')
    demon_demon_scores = torch.matmul(demon_embeds, demon_embeds.T) # [demon_num, demon_num]
    demon_demon_scores = demon_demon_scores.cpu().numpy()
    # print(f'demon_demon_scores.shape: {demon_demon_scores.shape}')
    remain = np.arange(1, demon_embeds.shape[0])
    selected = [0]
    for cnt in range(return_number - 1):
        mmr_score = -np.inf
        selected_index = -1
        for i in range(len(remain)):
            selected_scores_max = np.max(demon_demon_scores[remain[i]][selected])
            now_mmr_score = mmr_lambda * qd_demon_scores[remain[i]] - (1 - mmr_lambda) * selected_scores_max
            if now_mmr_score > mmr_score:
                mmr_score = now_mmr_score
                selected_index = i
        
        selected.append(remain[selected_index])
        remain = np.delete(remain, selected_index)
    assert(len(selected) == return_number)
    return selected[::-1]