# -*- coding: utf-8 -*-
"""Information Retrieval metrics
Useful Resources:
http://www.cs.utexas.edu/~mooney/ir-course/slides/Evaluation.ppt
http://www.nii.ac.jp/TechReports/05-014E.pdf
http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
http://hal.archives-ouvertes.fr/docs/00/72/67/60/PDF/07-busa-fekete.pdf
Learning to Rank for Information Retrieval (Tie-Yan Liu)


https://scikit-learn.org/stable/modules/classes.html

'accuracy_score', 'adjusted_mutual_info_score', 'adjusted_rand_score', 'auc', 'average_precision_score', 'balanced_accuracy_score', 'calinski_harabaz_score', 'calinski_harabasz_score', 'check_scoring', 'classification_report', 'cluster', 'cohen_kappa_score', 'completeness_score', 'confusion_matrix', 'consensus_score', 'coverage_error', 'davies_bouldin_score', 'euclidean_distances', 'explained_variance_score', 'f1_score', 'fbeta_score', 'fowlkes_mallows_score', 'get_scorer', 'hamming_loss', 'hinge_loss', 'homogeneity_completeness_v_measure', 'homogeneity_score', 'jaccard_score', 'jaccard_similarity_score', 'label_ranking_average_precision_score', 'label_ranking_loss', 'log_loss', 'make_scorer', 'matthews_corrcoef', 'max_error', 'mean_absolute_error', 'mean_squared_error', 'mean_squared_log_error', 'median_absolute_error', 'multilabel_confusion_matrix', 'mutual_info_score', 'normalized_mutual_info_score', 'pairwise_distances', 'pairwise_distances_argmin', 'pairwise_distances_argmin_min', 'pairwise_distances_chunked', 'pairwise_kernels', 'precision_recall_curve', 'precision_recall_fscore_support', 'precision_score', 'r2_score', 'recall_score', 'roc_auc_score', 'roc_curve', 'SCORERS', 'silhouette_samples', 'silhouette_score', 'v_measure_score', 'zero_one_loss', 'brier_score_loss'

http://rasbt.github.io/mlxtend/api_subpackages/mlxtend.evaluate/#proportion_difference

BootstrapOutOfBag
Methods
PredefinedHoldoutSplit
Methods
RandomHoldoutSplit
Methods
bias_variance_decomp
bootstrap
bootstrap_point632_score
cochrans_q
combined_ftest_5x2cv
confusion_matrix
feature_importance_permutation
ftest
lift_score
mcnemar
mcnemar_table
mcnemar_tables
paired_ttest_5x2cv
paired_ttest_kfold_cv
paired_ttest_resampled
permutation_test
proportion_difference
scoring

http://rasbt.github.io/mlxtend/api_subpackages/mlxtend.evaluate/#proportion_difference


https://www.nltk.org/api/nltk.metrics.html#module-nltk.metrics.scores


"""

import copy
import math
import os
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd
import scipy as sci

import sklearn

###############################################################################################################
from sklearn.metrics import *

from mlxtend.evaluate import *

###############################################################################################################




###############################################################################################################
def metrics_report(y_true, y_pred, y_pred_proba=None,
                                   metrics=["f1_macro", "accuracy", "precision_macro", "recall_macro"]):
    entries = []
    for metric in metrics:
        metric_val = globals()[metric]((y_true, y_pred)
                                      
        entries.append((metric, metric_val))
    cv_df = pd.DataFrame(entries, columns=["metric", 'metric_val'])
    return cv_df



######### Ranking Metrics ############################################################################################
def batch_binary_ndcg(x_pred, heldout_batch, k=100):
    """
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicat 0 relevance
    """
    batch_users = x_pred.shape[0]
    idx_topk_part = bn.argpartition(-x_pred, k, axis=1)
    topk_part = x_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)
    # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
    # topk predicted score
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    tp = 1. / np.log2(np.arange(2, k + 2))

    dcg = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk].toarray() * tp).sum(axis=1)
    idcg = np.array([(tp[:min(n, k)]).sum()
                     for n in heldout_batch.getnnz(axis=1)])
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0
    return ndcg


def batch_recall(x_pred, heldout_batch, k=100):
    batch_users = x_pred.shape[0]

    idx = bn.argpartition(-x_pred, k, axis=1)
    x_pred_binary = np.zeros_like(x_pred, dtype=bool)
    x_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    x_true_binary = (heldout_batch > 0).toarray()
    tmp = (np.logical_and(x_true_binary, x_pred_binary).sum(axis=1)).astype(np.float32)
    recall = tmp / np.minimum(k, x_true_binary.sum(axis=1))
    recall[np.isnan(recall)] = 0
    return recall

def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


def precision_r(r):
    """Score is precision after all relevant documents have been retrieved
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> r_precision(r)
    0.33333333333333331
    >>> r = [0, 1, 0]
    >>> r_precision(r)
    0.5
    >>> r = [1, 0, 0]
    >>> r_precision(r)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        R Precision
    """
    r = np.asarray(r) != 0
    z = r.nonzero()[0]
    if not z.size:
        return 0.
    return np.mean(r[:z[-1] + 1])


def precision(r, k=-1):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision(r, 1)
    0.0
    >>> precision(r, 2)
    0.0
    >>> precision(r, 3)
    0.33333333333333331
    >>> precision(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: The last element used to compute precision
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def precision_average(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def precision_mean_average(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def discounted_cumulative_gain(r, k=-1, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> discounted_cumulative_gain(r, 1)
    3.0
    >>> discounted_cumulative_gain(r, 1, method=1)
    3.0
    >>> discounted_cumulative_gain(r, 2)
    5.0
    >>> discounted_cumulative_gain(r, 2, method=1)
    4.2618595071429155
    >>> discounted_cumulative_gain(r, 10)
    9.6051177391888114
    >>> discounted_cumulative_gain(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def normalized_discounted_cumulative_gain(r, k=-1, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> normalized_discounted_cumulative_gain(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> normalized_discounted_cumulative_gain(r, 4)
    0.9203032077642922
    >>> normalized_discounted_cumulative_gain(r, 4, method=1)
    0.96519546960144276
    >>> normalized_discounted_cumulative_gain([0], 1)
    0.0
    >>> normalized_discounted_cumulative_gain([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = discounted_cumulative_gain(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return discounted_cumulative_gain(r, k, method) / dcg_max


def ztest():
    pass


if __name__ == "__main__":
    ztest()

                                       
                                       
                                       
                                       
