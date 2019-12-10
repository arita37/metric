# -*- coding: utf-8 -*-
import copy
import math
import os
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd
import scipy as sci

import sklearn


"""Information Retrieval metrics
Useful Resources:
http://www.cs.utexas.edu/~mooney/ir-course/slides/Evaluation.ppt
http://www.nii.ac.jp/TechReports/05-014E.pdf
http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
http://hal.archives-ouvertes.fr/docs/00/72/67/60/PDF/07-busa-fekete.pdf
Learning to Rank for Information Retrieval (Tie-Yan Liu)


"""




###############################################################################################################
###############################################################################################################
"""

https://scikit-learn.org/stable/modules/classes.html

'accuracy_score', 'adjusted_mutual_info_score', 'adjusted_rand_score', 'auc', 'average_precision_score', 'balanced_accuracy_score', 'calinski_harabaz_score', 'calinski_harabasz_score', 'check_scoring', 'classification_report', 'cluster', 'cohen_kappa_score', 'completeness_score', 'confusion_matrix', 'consensus_score', 'coverage_error', 'davies_bouldin_score', 'euclidean_distances', 'explained_variance_score', 'f1_score', 'fbeta_score', 'fowlkes_mallows_score', 'get_scorer', 'hamming_loss', 'hinge_loss', 'homogeneity_completeness_v_measure', 'homogeneity_score', 'jaccard_score', 'jaccard_similarity_score', 'label_ranking_average_precision_score', 'label_ranking_loss', 'log_loss', 'make_scorer', 'matthews_corrcoef', 'max_error', 'mean_absolute_error', 'mean_squared_error', 'mean_squared_log_error', 'median_absolute_error', 'multilabel_confusion_matrix', 'mutual_info_score', 'normalized_mutual_info_score', 'pairwise_distances', 'pairwise_distances_argmin', 'pairwise_distances_argmin_min', 'pairwise_distances_chunked', 'pairwise_kernels', 'precision_recall_curve', 'precision_recall_fscore_support', 'precision_score', 'r2_score', 'recall_score', 'roc_auc_score', 'roc_curve', 'SCORERS', 'silhouette_samples', 'silhouette_score', 'v_measure_score', 'zero_one_loss', 'brier_score_loss'



http://rasbt.github.io/mlxtend/api_subpackages/mlxtend.evaluate/#proportion_difference



"""
from sklearn.metrics import *




###############################################################################################################
###############################################################################################################
"""
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



"""
from mlxtend.evaluate import *





###############################################################################################################
###############################################################################################################
"""
https://www.nltk.org/api/nltk.metrics.html#module-nltk.metrics.scores



"""






###############################################################################################################
###############################################################################################################
def sk_model_eval_regression(clf, istrain=1, Xtrain=None, ytrain=None, Xval=None, yval=None):
    if istrain:
        clf.fit(Xtrain, ytrain)

    CV_score = -cross_val_score(clf, Xtrain, ytrain, scoring="neg_mean_absolute_error", cv=4)

    print("CV score: ", CV_score)
    print("CV mean: ", CV_score.mean())
    print("CV std:", CV_score.std())

    train_y_predicted_logReg = clf.predict(Xtrain)
    val_y_predicted_logReg = clf.predict(Xval)

    print("\n")
    print("Score on logReg training set:", mean_absolute_error(ytrain, train_y_predicted_logReg))
    print("Score on logReg validation set:", mean_absolute_error(yval, val_y_predicted_logReg))

    return clf, train_y_predicted_logReg, val_y_predicted_logReg


def sk_model_eval_classification(clf, istrain=1, Xtrain=None, ytrain=None, Xtest=None, ytest=None):
    if istrain:
        print("############# Train dataset  ####################################")
        clf.fit(Xtrain, ytrain)
        ytrain_proba = clf.predict_proba(Xtrain)[:, 1]
        ytrain_pred = clf.predict(Xtrain)
        sk_showmetrics(ytrain, ytrain_pred, ytrain_proba)

    print("############# Test dataset  #########################################")
    ytest_proba = clf.predict_proba(Xtest)[:, 1]
    ytest_pred = clf.predict(Xtest)
    sk_showmetrics(ytest, ytest_pred, ytest_proba)

    return clf, {"ytest_pred": ytest_pred}


def sk_metrics_eval(clf, Xtest, ytest, cv=1, metrics=["f1_macro", "accuracy", "precision_macro", "recall_macro"] ) :
  #
  entries = []
  model_name = clf.__class__.__name__
  for metric in  metrics :
    metric_val = cross_val_score(clf, Xtest, ytest, scoring= metric, cv=3)
    for i, metric_val_i in enumerate(metric_val):
       entries.append((model_name, i, metric, metric_val_i ))
  cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', "metric", 'metric_val'])
  return cv_df


def sk_model_eval(clf_list, Xtest, ytest, cv=1,
                  metrics=["f1_macro", "accuracy", "precision", "recall"]):
    df_list = []
    for clf in clf_list:
        df_clf_cv = sk_metrics_eval(clf, Xtest, ytest, cv=cv, metrics=metrics)
        df_list.append(df_clf_cv)
    
    return pd.concat(df_list, axis=0)





######### Ranking Metrics ############################################################################################
def ndcg_binary_at_k_batch(x_pred, heldout_batch, k=100):
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



def recall_at_k_batch(x_pred, heldout_batch, k=100):
    batch_users = x_pred.shape[0]

    idx = bn.argpartition(-x_pred, k, axis=1)
    x_pred_binary = np.zeros_like(x_pred, dtype=bool)
    x_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True

    x_true_binary = (heldout_batch > 0).toarray()
    tmp           = (np.logical_and(x_true_binary, x_pred_binary).sum(axis=1)).astype( np.float32)
    recall        = tmp / np.minimum(k, x_true_binary.sum(axis=1))
    recall[np.isnan(recall)] = 0
    return recall







###############################################################################################################
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


def r_precision(r):
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


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
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


def average_precision(r):
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
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
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


def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
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


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
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
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max



def ztest():
	pass

if __name__ == "__main__":
     ztest()









"""


sklearn.metrics: Metrics
See the Metrics and scoring: quantifying the quality of predictions section and the Pairwise metrics, Affinities and Kernels section of the user guide for further details.

The sklearn.metrics module includes score functions, performance metrics and pairwise metrics and distance computations.

Model Selection Interface
See the The scoring parameter: defining model evaluation rules section of the user guide for further details.

metrics.check_scoring(estimator[, scoring, …])

Determine scorer from user options.

metrics.get_scorer(scoring)

Get a scorer from string.

metrics.make_scorer(score_func[, …])

Make a scorer from a performance metric or loss function.

Classification metrics
See the Classification metrics section of the user guide for further details.

metrics.accuracy_score(y_true, y_pred[, …])

Accuracy classification score.

metrics.auc(x, y)

Compute Area Under the Curve (AUC) using the trapezoidal rule

metrics.average_precision_score(y_true, y_score)

Compute average precision (AP) from prediction scores

metrics.balanced_accuracy_score(y_true, y_pred)

Compute the balanced accuracy

metrics.brier_score_loss(y_true, y_prob[, …])

Compute the Brier score.

metrics.classification_report(y_true, y_pred)

Build a text report showing the main classification metrics

metrics.cohen_kappa_score(y1, y2[, labels, …])

Cohen’s kappa: a statistic that measures inter-annotator agreement.

metrics.confusion_matrix(y_true, y_pred[, …])

Compute confusion matrix to evaluate the accuracy of a classification.

metrics.dcg_score(y_true, y_score[, k, …])

Compute Discounted Cumulative Gain.

metrics.f1_score(y_true, y_pred[, labels, …])

Compute the F1 score, also known as balanced F-score or F-measure

metrics.fbeta_score(y_true, y_pred, beta[, …])

Compute the F-beta score

metrics.hamming_loss(y_true, y_pred[, …])

Compute the average Hamming loss.

metrics.hinge_loss(y_true, pred_decision[, …])

Average hinge loss (non-regularized)

metrics.jaccard_score(y_true, y_pred[, …])

Jaccard similarity coefficient score

metrics.log_loss(y_true, y_pred[, eps, …])

Log loss, aka logistic loss or cross-entropy loss.

metrics.matthews_corrcoef(y_true, y_pred[, …])

Compute the Matthews correlation coefficient (MCC)

metrics.multilabel_confusion_matrix(y_true, …)

Compute a confusion matrix for each class or sample

metrics.ndcg_score(y_true, y_score[, k, …])

Compute Normalized Discounted Cumulative Gain.

metrics.precision_recall_curve(y_true, …)

Compute precision-recall pairs for different probability thresholds

metrics.precision_recall_fscore_support(…)

Compute precision, recall, F-measure and support for each class

metrics.precision_score(y_true, y_pred[, …])

Compute the precision

metrics.recall_score(y_true, y_pred[, …])

Compute the recall

metrics.roc_auc_score(y_true, y_score[, …])

Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.

metrics.roc_curve(y_true, y_score[, …])

Compute Receiver operating characteristic (ROC)

metrics.zero_one_loss(y_true, y_pred[, …])

Zero-one classification loss.

Regression metrics
See the Regression metrics section of the user guide for further details.

metrics.explained_variance_score(y_true, y_pred)

Explained variance regression score function

metrics.max_error(y_true, y_pred)

max_error metric calculates the maximum residual error.

metrics.mean_absolute_error(y_true, y_pred)

Mean absolute error regression loss

metrics.mean_squared_error(y_true, y_pred[, …])

Mean squared error regression loss

metrics.mean_squared_log_error(y_true, y_pred)

Mean squared logarithmic error regression loss

metrics.median_absolute_error(y_true, y_pred)

Median absolute error regression loss

metrics.r2_score(y_true, y_pred[, …])

R^2 (coefficient of determination) regression score function.

metrics.mean_poisson_deviance(y_true, y_pred)

Mean Poisson deviance regression loss.

metrics.mean_gamma_deviance(y_true, y_pred)

Mean Gamma deviance regression loss.

metrics.mean_tweedie_deviance(y_true, y_pred)

Mean Tweedie deviance regression loss.

Multilabel ranking metrics
See the Multilabel ranking metrics section of the user guide for further details.

metrics.coverage_error(y_true, y_score[, …])

Coverage error measure

metrics.label_ranking_average_precision_score(…)

Compute ranking-based average precision

metrics.label_ranking_loss(y_true, y_score)

Compute Ranking loss measure

Clustering metrics
See the Clustering performance evaluation section of the user guide for further details.

The sklearn.metrics.cluster submodule contains evaluation metrics for cluster analysis results. There are two forms of evaluation:

supervised, which uses a ground truth class values for each sample.

unsupervised, which does not and measures the ‘quality’ of the model itself.

metrics.adjusted_mutual_info_score(…[, …])

Adjusted Mutual Information between two clusterings.

metrics.adjusted_rand_score(labels_true, …)

Rand index adjusted for chance.

metrics.calinski_harabasz_score(X, labels)

Compute the Calinski and Harabasz score.

metrics.davies_bouldin_score(X, labels)

Computes the Davies-Bouldin score.

metrics.completeness_score(labels_true, …)

Completeness metric of a cluster labeling given a ground truth.

metrics.cluster.contingency_matrix(…[, …])

Build a contingency matrix describing the relationship between labels.

metrics.fowlkes_mallows_score(labels_true, …)

Measure the similarity of two clusterings of a set of points.

metrics.homogeneity_completeness_v_measure(…)

Compute the homogeneity and completeness and V-Measure scores at once.

metrics.homogeneity_score(labels_true, …)

Homogeneity metric of a cluster labeling given a ground truth.

metrics.mutual_info_score(labels_true, …)

Mutual Information between two clusterings.

metrics.normalized_mutual_info_score(…[, …])

Normalized Mutual Information between two clusterings.

metrics.silhouette_score(X, labels[, …])

Compute the mean Silhouette Coefficient of all samples.

metrics.silhouette_samples(X, labels[, metric])

Compute the Silhouette Coefficient for each sample.

metrics.v_measure_score(labels_true, labels_pred)

V-measure cluster labeling given a ground truth.

Biclustering metrics
See the Biclustering evaluation section of the user guide for further details.

metrics.consensus_score(a, b[, similarity])

The similarity of two sets of biclusters.

Pairwise metrics
See the Pairwise metrics, Affinities and Kernels section of the user guide for further details.

metrics.pairwise.additive_chi2_kernel(X[, Y])

Computes the additive chi-squared kernel between observations in X and Y

metrics.pairwise.chi2_kernel(X[, Y, gamma])

Computes the exponential chi-squared kernel X and Y.

metrics.pairwise.cosine_similarity(X[, Y, …])

Compute cosine similarity between samples in X and Y.

metrics.pairwise.cosine_distances(X[, Y])

Compute cosine distance between samples in X and Y.

metrics.pairwise.distance_metrics()

Valid metrics for pairwise_distances.

metrics.pairwise.euclidean_distances(X[, Y, …])

Considering the rows of X (and Y=X) as vectors, compute the distance matrix between each pair of vectors.

metrics.pairwise.haversine_distances(X[, Y])

Compute the Haversine distance between samples in X and Y

metrics.pairwise.kernel_metrics()

Valid metrics for pairwise_kernels

metrics.pairwise.laplacian_kernel(X[, Y, gamma])

Compute the laplacian kernel between X and Y.

metrics.pairwise.linear_kernel(X[, Y, …])

Compute the linear kernel between X and Y.

metrics.pairwise.manhattan_distances(X[, Y, …])

Compute the L1 distances between the vectors in X and Y.

metrics.pairwise.nan_euclidean_distances(X)

Calculate the euclidean distances in the presence of missing values.

metrics.pairwise.pairwise_kernels(X[, Y, …])

Compute the kernel between arrays X and optional array Y.

metrics.pairwise.polynomial_kernel(X[, Y, …])

Compute the polynomial kernel between X and Y.

metrics.pairwise.rbf_kernel(X[, Y, gamma])

Compute the rbf (gaussian) kernel between X and Y.

metrics.pairwise.sigmoid_kernel(X[, Y, …])

Compute the sigmoid kernel between X and Y.

metrics.pairwise.paired_euclidean_distances(X, Y)

Computes the paired euclidean distances between X and Y

metrics.pairwise.paired_manhattan_distances(X, Y)

Compute the L1 distances between the vectors in X and Y.

metrics.pairwise.paired_cosine_distances(X, Y)

Computes the paired cosine distances between X and Y

metrics.pairwise.paired_distances(X, Y[, metric])

Computes the paired distances between X and Y.

metrics.pairwise_distances(X[, Y, metric, …])

Compute the distance matrix from a vector array X and optional Y.

metrics.pairwise_distances_argmin(X, Y[, …])

Compute minimum distances between one point and a set of points.

metrics.pairwise_distances_argmin_min(X, Y)

Compute minimum distances between one point and a set of points.

metrics.pairwise_distances_chunked(X[, Y, …])

Generate a distance matrix chunk by chunk with optional reduction


"""