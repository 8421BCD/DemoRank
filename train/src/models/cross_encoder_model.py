import torch
import torch.nn as nn

from typing import Optional, Dict
from transformers import (
    PreTrainedModel,
    AutoModelForSequenceClassification
)
from transformers.modeling_outputs import SequenceClassifierOutput

from config import Arguments
from torch.nn import BCELoss, BCEWithLogitsLoss
from itertools import product
import torch.nn.functional as F

def list_net(y_pred, y_true=None, padded_value_indicator=-100, eps=1e-10):
    """
        ListNet loss introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".
        :param y_pred: predictions from the model, shape [batch_size, slate_length]
        :param y_true: ground truth labels, shape [batch_size, slate_length]
        :param eps: epsilon value, used for numerical stability
        :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
        :return: loss value, a torch.Tensor
        """
    if y_true is None:
        y_true = torch.zeros_like(y_pred).to(y_pred.device)
        y_true[:, 0] = 1

    preds_smax = F.softmax(y_pred, dim=1)
    preds_smax = preds_smax + eps
    preds_log = torch.log(preds_smax)
    
    true_smax = F.softmax(y_true, dim=1)

    return torch.mean(-torch.sum(true_smax * preds_log, dim=1))


def rank_net(y_pred, y_true=None, padded_value_indicator=-100, weight_by_diff=False,
                weight_by_diff_powed=False):
    """
    RankNet loss introduced in "Learning to Rank using Gradient Descent".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param weight_by_diff: flag indicating whether to weight the score differences by ground truth differences.
    :param weight_by_diff_powed: flag indicating whether to weight the score differences by the squared ground truth differences.
    :return: loss value, a torch.Tensor
    """
    if y_true is None:
        y_true = torch.zeros_like(y_pred).to(y_pred.device)
        y_true[:, 0] = 1

    # here we generate every pair of indices from the range of document length in the batch
    document_pairs_candidates = list(product(range(y_true.shape[1]), repeat=2))

    pairs_true = y_true[:, document_pairs_candidates]
    selected_pred = y_pred[:, document_pairs_candidates]

    # here we calculate the relative true relevance of every candidate pair
    true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
    pred_diffs = selected_pred[:, :, 0] - selected_pred[:, :, 1]

    # here we filter just the pairs that are 'positive' and did not involve a padded instance
    # we can do that since in the candidate pairs we had symetric pairs so we can stick with
    # positive ones for a simpler loss function formulation
    the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs))

    pred_diffs = pred_diffs[the_mask]

    weight = None
    if weight_by_diff:
        abs_diff = torch.abs(true_diffs)
        weight = abs_diff[the_mask]
    elif weight_by_diff_powed:
        true_pow_diffs = torch.pow(pairs_true[:, :, 0], 2) - torch.pow(pairs_true[:, :, 1], 2)
        abs_diff = torch.abs(true_pow_diffs)
        weight = abs_diff[the_mask]

    # here we 'binarize' true relevancy diffs since for a pairwise loss we just need to know
    # whether one document is better than the other and not about the actual difference in
    # their relevancy levels
    true_diffs = (true_diffs > 0).type(torch.float32)
    true_diffs = true_diffs[the_mask]

    return BCEWithLogitsLoss(weight=weight)(pred_diffs, true_diffs)

class Reranker(nn.Module):
    def __init__(self, hf_model: PreTrainedModel, args: Arguments):
        super().__init__()
        self.hf_model = hf_model
        self.args = args

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, batch: Dict[str, torch.Tensor]) -> SequenceClassifierOutput:
        batch_input_dict = {k: v for k, v in batch.items() if k not in ['labels', 'kd_labels']}

        outputs: SequenceClassifierOutput = self.hf_model(**batch_input_dict, return_dict=True)
        outputs.logits = outputs.logits.view(-1, self.args.train_n_passages)

        # ce loss
        # positive_labels = batch['kd_labels'][:, 0].unsqueeze(1)
        # mask = (batch['kd_labels'] == positive_labels)
        # mask[:, 0] = False
        # masked_scores = outputs.logits.masked_fill(mask, float('-inf'))
        # ce_loss = self.cross_entropy(masked_scores, batch['labels'])

        # ranknet loss
        rank_gt = 1/torch.arange(1,1+outputs.logits.shape[1]).view(1,-1).repeat(outputs.logits.shape[0],1).to(outputs.logits.device)
        for i in range(rank_gt.shape[0]):
            for j in range(rank_gt.shape[1]):
                if j > 0 and batch['kd_labels'][i][j] == batch['kd_labels'][i][j - 1]:
                    rank_gt[i][j] = rank_gt[i][j - 1]
        rank_loss = rank_net(outputs.logits, rank_gt)

        # listnet loss
        # rank_gt = torch.where(batch['kd_labels'] == 0, torch.tensor(float('-inf')), batch['kd_labels'])
        # rank_loss = list_net(outputs.logits, rank_gt)
        # print(f'ce_loss: {self.args.kd_cont_loss_weight * ce_loss}, rank_loss: {rank_loss}')
        # loss = self.args.kd_cont_loss_weight * ce_loss + rank_loss
        loss = rank_loss
        outputs.loss = loss
        
        return outputs

    def gradient_checkpointing_enable(self):
        self.hf_model.gradient_checkpointing_enable()

    @classmethod
    def from_pretrained(cls, all_args: Arguments, *args, **kwargs):
        hf_model = AutoModelForSequenceClassification.from_pretrained(*args, **kwargs)
        return cls(hf_model, all_args)

    def save_pretrained(self, output_dir: str):
        self.hf_model.save_pretrained(output_dir)


class RerankerForInference(nn.Module):
    def __init__(self, hf_model: Optional[PreTrainedModel] = None):
        super().__init__()
        self.hf_model = hf_model
        self.hf_model.eval()

    @torch.no_grad()
    def forward(self, batch) -> SequenceClassifierOutput:
        return self.hf_model(**batch)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        hf_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path)
        return cls(hf_model)
