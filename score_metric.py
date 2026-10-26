import torch
import numpy as np
from tqdm import tqdm
import not_in_use.algorithm.beat_tracking as bt


def get_cr_score_matrix(src_bc, tgt_bc):
    len_src, len_tgt = len(src_bc), len(tgt_bc)
    score_matrix = np.zeros((len_src, len_tgt), np.float32)

    for i, beat_chrom_src in enumerate(tqdm(src_bc)):
        for j, beat_chrom_tgt in enumerate(tgt_bc):

            score, _, _ = bt.get_cross_correlation_lag_pitch(beat_chrom_src, beat_chrom_tgt)

            score_matrix[i, j] = score

    return score_matrix


def get_dtw_score_matrix(src_bc, tgt_bc):
    len_src, len_tgt = len(src_bc), len(tgt_bc)
    score_matrix = np.zeros((len_src, len_tgt), np.float32)

    for i, beat_chrom_src in enumerate(tqdm(src_bc)):
        for j, beat_chrom_tgt in enumerate(tgt_bc):

            score_matrix[i, j] = bt.get_dtw_score(beat_chrom_src, beat_chrom_tgt)

    return score_matrix


def get_top1_acc(score_matrix, tgt_labels=None):
    ranked_top1 = np.argmax(score_matrix, axis=1)

    if tgt_labels == None:
        correct = np.arange(score_matrix.shape[0]) == ranked_top1 # src index = top1 idx
    else:
        tgt_labels = np.asarray(tgt_labels)
        correct = np.arange(score_matrix.shape[0]) == tgt_labels[ranked_top1] # src index = top1 idx

    top1 = np.mean(correct)

    return top1


def get_mrr(score_matrix, tgt_labels=None):
    """_summary_

    Args:
        score_matrix (_type_): _description_
        tgt_labels (ndarray, optional): numpy array of target labels. Defaults to None.
    Returns:
        _type_: _description_
    """
    reciprocal_ranks = []

    for i in range(score_matrix.shape[0]):
        ranked_indices = np.argsort(score_matrix[i])[::-1] # target index sorted according to highest score
        if tgt_labels is None:
            rank = np.where(ranked_indices == i)[0][0] + 1 # e.g. (array([2]),)[0][0] = 2
        else:
            tgt_labels = np.asarray(tgt_labels)
            ranked_labels = tgt_labels[ranked_indices]
            rank = np.where(ranked_labels == i)[0][0] + 1
            
        reciprocal_ranks.append(1.0 / rank)

    return np.mean(reciprocal_ranks)


def get_map(score_matrix, tgt_labels):
    average_precisions = []

    for i in range(score_matrix.shape[0]):
        ranked_indices = np.argsort(score_matrix[i])[::-1] # target index sorted according to highest score

        tgt_labels = np.asarray(tgt_labels)
        ranked_labels = tgt_labels[ranked_indices] # predicted index's label (it may be correct or wrong)
        relevant = (ranked_labels == i) # boolean array of match/unmatch
        num_relevant = np.sum(relevant)

        if num_relevant == 0: # no target case
            continue

        relevant_ranks = np.where(relevant)[0] + 1 # 1-based rank > +1
        precisions = (np.arange(1, num_relevant + 1) / relevant_ranks) # numerator: range from 1 to number or relevant | denominator: sorted rank

        average_precision = np.mean(precisions)
        average_precisions.append(average_precision)

    if len(average_precisions) == 0:
        return 0.0
    else:
        return np.mean(average_precisions)