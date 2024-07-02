import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from torch import Tensor
from transformers import PreTrainedModel, AutoModel, AutoModelForSequenceClassification
from transformers.modeling_outputs import ModelOutput
from torch.nn import BCELoss, BCEWithLogitsLoss

from config import Arguments
from logger_config import logger
from util import dist_gather_tensor, select_grouped_indices, full_contrastive_scores_and_labels, pool, full_contrastive_scores_and_labels_with_inbatch
from itertools import product


@dataclass
class BiencoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    labels: Optional[Tensor] = None
    scores: Optional[Tensor] = None
    loss_dict: Optional[Dict] = None

def lambda_loss(y_pred, y_true=None, eps=1e-10, padded_value_indicator=-100, weighing_scheme=None, k=None,
                sigma=1., mu=10., reduction="mean", reduction_log="binary"):
    """
    LambdaLoss framework for LTR losses implementations, introduced in "The LambdaLoss Framework for Ranking Metric Optimization".
    Contains implementations of different weighing schemes corresponding to e.g. LambdaRank or RankNet.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param weighing_scheme: a string corresponding to a name of one of the weighing schemes
    :param k: rank at which the loss is truncated
    :param sigma: score difference weight used in the sigmoid function
    :param mu: optional weight used in NDCGLoss2++ weighing scheme
    :param reduction: losses reduction method, could be either a sum or a mean
    :param reduction_log: logarithm variant used prior to masking and loss reduction, either binary or natural
    :return: loss value, a torch.Tensor
    """
    y_pred = y_pred.clone().type(torch.float32)
    y_true = y_true.clone().type(torch.float32)

    if y_true is None:
        y_true = torch.zeros_like(y_pred).to(y_pred.device)
        y_true[:, 0] = 1

    device = y_pred.device

    # Here we sort the true and predicted relevancy scores.
    y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
    y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

    # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
    true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
    true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
    padded_pairs_mask = torch.isfinite(true_diffs)

    if weighing_scheme != "ndcgLoss1_scheme":
        padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)

    ndcg_at_k_mask = torch.zeros((y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device)
    ndcg_at_k_mask[:k, :k] = 1

    # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
    true_sorted_by_preds.clamp_(min=0.)
    y_true_sorted.clamp_(min=0.)

    # Here we find the gains, discounts and ideal DCGs per slate.
    pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
    D = torch.log2(1. + pos_idxs.float())[None, :]
    maxDCGs = torch.sum(((torch.pow(2, y_true_sorted) - 1) / D)[:, :k], dim=-1).clamp(min=eps)
    G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

    # Here we apply appropriate weighing scheme - ndcgLoss1, ndcgLoss2, ndcgLoss2++ or no weights (=1.0)
    if weighing_scheme is None:
        weights = 1.
    else:
        weights = globals()[weighing_scheme](G, D, mu, true_sorted_by_preds)  # type: ignore

    # We are clamping the array entries to maintain correct backprop (log(0) and division by 0)
    scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-1e8, max=1e8)
    scores_diffs.masked_fill(torch.isnan(scores_diffs), 0.)
    weighted_probas = (torch.sigmoid(sigma * scores_diffs).clamp(min=eps) ** weights).clamp(min=eps)
    if reduction_log == "natural":
        losses = torch.log(weighted_probas)
    elif reduction_log == "binary":
        losses = torch.log2(weighted_probas)
    else:
        raise ValueError("Reduction logarithm base can be either natural or binary")

    if reduction == "sum":
        loss = -torch.sum(losses[padded_pairs_mask & ndcg_at_k_mask])
    elif reduction == "mean":
        loss = -torch.mean(losses[padded_pairs_mask & ndcg_at_k_mask])
    else:
        raise ValueError("Reduction method can be either sum or mean")

    return loss


def lambda_mrr_loss(y_pred, y_true, eps=1e-10, padded_value_indicator=-1, reduction="mean"):
    """
    y_pred: FloatTensor [bz, topk]
    y_true: FloatTensor [bz, topk]
    """
    # print(y_pred.shape, y_true.shape)
    device = y_pred.device
    y_pred = y_pred.clone().type(torch.float32)
    y_true = y_true.clone().type(torch.float32)
    clamp_val = 1e8 if y_pred.dtype==torch.float32 else 1e4
    # print(f'clamp_val: {clamp_val}')
    padded_mask = y_true == padded_value_indicator
    y_pred[padded_mask] = float("-inf")
    y_true[padded_mask] = float("-inf")
    #assert torch.sum(padded_mask) == 0

    # Here we sort the true and predicted relevancy scores.
    y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)

    # After sorting, we can mask out the pairs of indices (i, j) containing index of a padded element.
    true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
    true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
    # print(f'true_diffs: {true_diffs}')
    padded_pairs_mask = torch.isfinite(true_diffs)
    # print(f'padded_pairs_mask1: {padded_pairs_mask}')
    padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)
    # print(f'padded_pairs_mask2: {padded_pairs_mask}')

    # Here we clamp the -infs to get correct gains and ideal DCGs (maxDCGs)
    true_sorted_by_preds.clamp_(min=0.)

    # Here we find the gains, discounts and ideal DCGs per slate.
    inv_pos_idxs = 1. / torch.arange(1, y_pred.shape[1] + 1).to(device)
    weights = torch.abs(inv_pos_idxs.view(1,-1,1) - inv_pos_idxs.view(1,1,-1)) # [1, topk, topk]
    # print(f'weights: {weights}')

    # We are clamping the array entries to maintain correct backprop (log(0) and division by 0)
    scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(min=-50, max=50)
    # print(f'scores_diffs: {scores_diffs}')
    # logger.info('scores_diffs:{}'.format(scores_diffs))
    # logger.info('torch.exp(-scores_diffs):{}'.format(torch.exp(-scores_diffs)))
    scores_diffs.masked_fill_(torch.isnan(scores_diffs), 0.)
    scores_diffs_exp = torch.exp(-scores_diffs)
    # print(f'scores_diffs_exp: {scores_diffs_exp}')
    scores_diffs_exp = scores_diffs_exp.clamp(min=-clamp_val, max=clamp_val)
    # logger.info('torch.max(scores_diffs):{}'.format(torch.max(scores_diffs)))
    # logger.info('torch.max(scores_diffs_exp):{}'.format(torch.max(scores_diffs_exp)))
    # logger.info('scores_diffs size:{}'.format(scores_diffs.size()))
    losses = torch.log(1. + scores_diffs_exp) * weights #[bz, topk, topk]
    # logger.info('weights:{}'.format(weights))
    # print(f'losses: {losses}')

    if reduction == "sum":
        loss = torch.sum(losses[padded_pairs_mask])
    elif reduction == "mean":
        loss = torch.mean(losses[padded_pairs_mask])
    else:
        raise ValueError("Reduction method can be either sum or mean")

    # logger.info('loss:{}'.format(loss))

    return loss

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


class BiencoderModel(nn.Module):
    def __init__(self, args: Arguments,
                 lm_q: PreTrainedModel,
                 lm_p: PreTrainedModel):
        super().__init__()
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.args = args

        from trainers import BiencoderTrainer
        self.trainer: Optional[BiencoderTrainer] = None

        self._freeze_position_embedding_if_needed(self.lm_q)
        self._freeze_position_embedding_if_needed(self.lm_p)

    def forward(self, batch_dict: Dict[str, Tensor]) -> BiencoderOutput:
        scores, labels, q_reps, p_reps, all_scores, all_labels = self._compute_scores(batch_dict)
        # start = self.args.process_index * q_reps.shape[0]
        # group_indices = select_grouped_indices(scores=scores,
        #                                        group_size=self.args.train_n_passages,
        #                                        start=start * self.args.train_n_passages)

        if not self.args.use_rankloss:
            # training biencoder from scratch
            loss = self.cross_entropy(scores, labels)
            loss_dict = {'ce_loss': loss}
            return BiencoderOutput(loss=loss, q_reps=q_reps, p_reps=p_reps,
                        labels=labels.contiguous(),
                        scores=scores[:, :].contiguous(),
                        loss_dict=loss_dict)

        else:
            # training biencoder with kd
            # batch_size x train_n_passage
            # if self.args.do_stable_kd:
            #     teacher_targets = F.softmax(batch_dict['kd_labels'], dim=-1)
            #     kd_loss = 0
            #     group_size = teacher_targets.shape[1] # train demonstration numbers
            #     mask = torch.zeros_like(scores)
            #     for i in range(group_size):
            #         # print(f'mask: {mask}')
            #         temp_target = labels + i
            #         # print(f'temp_target: {temp_target}')
            #         temp_scores = scores + mask
            #         # print(f'temp_scores: {temp_scores}')
            #         loss = F.cross_entropy(temp_scores, temp_target, reduction="none") # B
            #         kd_loss = kd_loss + torch.mean(teacher_targets[:, i] * loss)
            #         mask = torch.scatter(mask, dim=-1, index=temp_target.unsqueeze(-1), value=torch.finfo(scores.dtype).min)
            # else:
            # group_log_scores = F.log_softmax(scores, dim=-1)
            # kd_log_target = F.log_softmax(batch_dict['kd_labels'] / 0.2, dim=-1)
            # # print(f'group_log_scores: {group_log_scores}')
            # # print(f'kd_log_target: {kd_log_target}')
            # kd_loss = self.kl_loss_fn(input=group_log_scores, target=kd_log_target)
            ce_loss = self.cross_entropy(scores, labels)
            # loss = self.args.kd_cont_loss_weight * ce_loss + kd_loss
            # rank_gt = 1/torch.arange(1,1+scores.shape[1]).view(1,-1).repeat(scores.shape[0],1).to(scores.device)
            # rank_loss = lambda_mrr_loss(scores, rank_gt)
            # print(f'ce_loss:{ce_loss}, rank_loss:{rank_loss}')
            # loss = self.args.kd_cont_loss_weight * ce_loss + kd_loss
            # loss_dict = {'ce_loss': (self.args.kd_cont_loss_weight * ce_loss).item(), 'kd_loss': kd_loss.item()}
            # total_n_psg = self.args.world_size * q_reps.shape[0] * self.args.train_n_passages
            # ------------- ranking loss -------------
            rank_gt = 1/torch.arange(1,1+scores.shape[1]).view(1,-1).repeat(scores.shape[0],1).to(scores.device)
            for i in range(rank_gt.shape[0]):
                for j in range(rank_gt.shape[1]):
                    if j > 0 and batch_dict['kd_labels'][i][j] == batch_dict['kd_labels'][i][j - 1]:
                        rank_gt[i][j] = rank_gt[i][j - 1]
            rank_loss = rank_net(scores, rank_gt)
            loss = self.args.kd_cont_loss_weight * ce_loss + rank_loss
            loss_dict = {'ce_loss': (self.args.kd_cont_loss_weight * ce_loss).item(), 'rank_loss': rank_loss.item()}

            return BiencoderOutput(loss=loss, q_reps=q_reps, p_reps=p_reps,
                                labels=labels.contiguous(),
                                scores=scores[:, :].contiguous(),
                                loss_dict=loss_dict)

    def _compute_scores(self, batch_dict: Dict[str, Tensor]) -> Tuple:
        embeds = self._encode(self.lm_p, batch_dict)
        batch_size = batch_dict['input_ids'].shape[0] // (self.args.train_n_passages + 1)
        q_reps = embeds[:batch_size]
        p_reps = embeds[batch_size:]
        assert p_reps.shape[0] == q_reps.shape[0] * self.args.train_n_passages

        # all_q_reps = dist_gather_tensor(q_reps)
        # all_p_reps = dist_gather_tensor(p_reps)
        # assert all_p_reps.shape[0] == self.args.world_size * q_reps.shape[0] * self.args.train_n_passages

        all_scores, all_labels = full_contrastive_scores_and_labels(
            query=q_reps, key=p_reps,
            use_all_pairs=self.args.full_contrastive_loss,
        )
        if self.args.l2_normalize:
            all_scores = all_scores / self.args.t
        # start = self.args.process_index * q_reps.shape[0]
        # local_query_indices = torch.arange(start, start + q_reps.shape[0], dtype=torch.long).to(q_reps.device)
        # # batch_size x (world_size x batch_size x train_n_passage)
        # scores = all_scores.index_select(dim=0, index=local_query_indices)
        # labels = all_labels.index_select(dim=0, index=local_query_indices)
        scores = all_scores
        labels = all_labels

        return scores, labels, q_reps, p_reps, all_scores, all_labels

    def _encode(self, encoder: PreTrainedModel, input_dict: dict) -> Optional[torch.Tensor]:
        if not input_dict:
            return None
        outputs = encoder(**{k: v for k, v in input_dict.items() if k not in ['labels', 'kd_labels']}, return_dict=True)
        embeds = pool(last_hidden_states=outputs.last_hidden_state,
                      attention_mask=input_dict['attention_mask'],
                      pool_type=self.args.pool_type)
        if self.args.l2_normalize:
            embeds = F.normalize(embeds, dim=-1)
        return embeds.contiguous()

    def _freeze_position_embedding_if_needed(self, model: nn.Module):
        if self.args.freeze_position_embedding:
            for name, param in model.named_parameters():
                if 'position_embeddings' in name:
                    param.requires_grad = False
                    logger.info('Freeze {}'.format(name))

    def gradient_checkpointing_enable(self):
        self.lm_q.gradient_checkpointing_enable()

    @classmethod
    def build(cls, args: Arguments, **hf_kwargs):
        if os.path.isdir(args.model_name_or_path):
            logger.info(f'loading shared model weight from {args.model_name_or_path}')
        lm_q = AutoModel.from_pretrained(args.model_name_or_path)
        lm_p = lm_q

        model = cls(args=args, lm_q=lm_q, lm_p=lm_p)
        return model

    def save(self, output_dir: str):
        self.lm_q.save_pretrained(output_dir)


class BiencoderModel_new(nn.Module):
    def __init__(self, args: Arguments,
                 lm_q: PreTrainedModel,
                 lm_p: PreTrainedModel):
        super().__init__()
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.args = args

        from trainers import BiencoderTrainer
        self.trainer: Optional[BiencoderTrainer] = None

        self._freeze_position_embedding_if_needed(self.lm_q)
        self._freeze_position_embedding_if_needed(self.lm_p)

    def forward(self, batch_dict: Dict[str, Tensor]) -> BiencoderOutput:
        scores, labels, q_reps, p_reps, all_scores, all_labels = self._compute_scores(batch_dict)

        if self.args.in_batch:
            start = self.args.process_index * q_reps.shape[0]
            group_indices = select_grouped_indices(scores=scores,
                                                group_size=self.args.train_n_passages,
                                                start=start * self.args.train_n_passages)

        if not self.args.use_rankloss:
            # training biencoder from scratch
            loss = self.cross_entropy(scores, labels)
            loss_dict = {'ce_loss': loss}
            return BiencoderOutput(loss=loss, q_reps=q_reps, p_reps=p_reps,
                        labels=labels.contiguous(),
                        scores=scores[:, :].contiguous(),
                        loss_dict=loss_dict)

        else:
            # whether to use in-batch negatives
            if self.args.in_batch == False:
                # ------------- contrastive loss -------------
                # ignore the negtives whose ground truth scores equal to the ones of postives
                positive_labels = batch_dict['kd_labels'][:, 0].unsqueeze(1)
                mask = (batch_dict['kd_labels'] == positive_labels)
                mask[:, 0] = False
                masked_scores = scores.masked_fill(mask, float('-inf'))
                ce_loss = self.cross_entropy(masked_scores, labels)
                # ce_loss = self.cross_entropy(scores, labels)

                # ------------- ranking loss -------------
                rank_gt = 1/torch.arange(1,1+scores.shape[1]).view(1,-1).repeat(scores.shape[0],1).to(scores.device)
                for i in range(rank_gt.shape[0]):
                    for j in range(rank_gt.shape[1]):
                        if j > 0 and batch_dict['kd_labels'][i][j] == batch_dict['kd_labels'][i][j - 1]:
                            rank_gt[i][j] = rank_gt[i][j - 1]
                rank_loss = rank_net(scores, rank_gt)
                loss = self.args.kd_cont_loss_weight * ce_loss + rank_loss
                loss_dict = {'ce_loss': (self.args.kd_cont_loss_weight * ce_loss).item(), 'rank_loss': rank_loss.item()}
                return BiencoderOutput(loss=loss, q_reps=q_reps, p_reps=p_reps,
                                    labels=labels.contiguous(),
                                    scores=scores[:, :].contiguous(),
                                    loss_dict=loss_dict)

            else: 
                ce_loss = self.cross_entropy(scores, labels)
                group_scores = torch.gather(input=scores, dim=1, index=group_indices)
                assert group_scores.shape[1] == self.args.train_n_passages
                # ------------- ranking loss -------------
                rank_gt = 1/torch.arange(1,1+group_scores.shape[1]).view(1,-1).repeat(group_scores.shape[0],1).to(group_scores.device)
                for i in range(rank_gt.shape[0]):
                    for j in range(rank_gt.shape[1]):
                        if j > 0 and batch_dict['kd_labels'][i][j] == batch_dict['kd_labels'][i][j - 1]:
                            rank_gt[i][j] = rank_gt[i][j - 1]
                rank_loss = rank_net(group_scores, rank_gt)
                loss = self.args.kd_cont_loss_weight * ce_loss + rank_loss
                loss_dict = {'ce_loss': (self.args.kd_cont_loss_weight * ce_loss).item(), 'rank_loss': rank_loss.item()}

                total_n_psg = self.args.world_size * q_reps.shape[0] * self.args.train_n_passages
                return BiencoderOutput(loss=loss, q_reps=q_reps, p_reps=p_reps,
                                    labels=labels.contiguous(),
                                    scores=scores[:, :total_n_psg].contiguous(),
                                    loss_dict=loss_dict)

    def _compute_scores(self, batch_dict: Dict[str, Tensor]) -> Tuple:
        embeds = self._encode(self.lm_p, batch_dict)
        batch_size = batch_dict['input_ids'].shape[0] // (self.args.train_n_passages + 1)
        q_reps = embeds[:batch_size]
        p_reps = embeds[batch_size:]
        assert p_reps.shape[0] == q_reps.shape[0] * self.args.train_n_passages
            
        if self.args.in_batch == False:
            all_scores, all_labels = full_contrastive_scores_and_labels(
                query=q_reps, key=p_reps,
                use_all_pairs=self.args.full_contrastive_loss,
            )
        else:
            all_q_reps = dist_gather_tensor(q_reps)
            all_p_reps = dist_gather_tensor(p_reps)
            assert all_p_reps.shape[0] == self.args.world_size * q_reps.shape[0] * self.args.train_n_passages
            all_scores, all_labels = full_contrastive_scores_and_labels_with_inbatch(
                query=all_q_reps, key=all_p_reps,
                use_all_pairs=self.args.full_contrastive_loss
            )

        if self.args.l2_normalize:
            all_scores = all_scores / self.args.t

        if self.args.in_batch == False:
            scores = all_scores
            labels = all_labels
        else:
            start = self.args.process_index * q_reps.shape[0]
            local_query_indices = torch.arange(start, start + q_reps.shape[0], dtype=torch.long).to(q_reps.device)
            # batch_size x (world_size x batch_size x train_n_passage)
            scores = all_scores.index_select(dim=0, index=local_query_indices)
            labels = all_labels.index_select(dim=0, index=local_query_indices)

        return scores, labels, q_reps, p_reps, all_scores, all_labels

    def _encode(self, encoder: PreTrainedModel, input_dict: dict) -> Optional[torch.Tensor]:
        if not input_dict:
            return None
        outputs = encoder(**{k: v for k, v in input_dict.items() if k not in ['labels', 'kd_labels']}, return_dict=True)
        embeds = pool(last_hidden_states=outputs.last_hidden_state,
                      attention_mask=input_dict['attention_mask'],
                      pool_type=self.args.pool_type)
        if self.args.l2_normalize:
            embeds = F.normalize(embeds, dim=-1)
        return embeds.contiguous()

    def _freeze_position_embedding_if_needed(self, model: nn.Module):
        if self.args.freeze_position_embedding:
            for name, param in model.named_parameters():
                if 'position_embeddings' in name:
                    param.requires_grad = False
                    logger.info('Freeze {}'.format(name))

    def gradient_checkpointing_enable(self):
        self.lm_q.gradient_checkpointing_enable()

    @classmethod
    def build(cls, args: Arguments, **hf_kwargs):
        if os.path.isdir(args.model_name_or_path):
            logger.info(f'loading shared model weight from {args.model_name_or_path}')
        lm_q = AutoModel.from_pretrained(args.model_name_or_path)
        lm_p = lm_q

        model = cls(args=args, lm_q=lm_q, lm_p=lm_p)
        return model

    def save(self, output_dir: str):
        self.lm_q.save_pretrained(output_dir)

