# -*- coding: utf-8 -*-
"""Information Retrieval metrics
Useful Resources:
http://www.cs.utexas.edu/~mooney/ir-course/slides/Evaluation.ppt
http://www.nii.ac.jp/TechReports/05-014E.pdf
http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
http://hal.archives-ouvertes.fr/docs/00/72/67/60/PDF/07-busa-fekete.pdf
Learning to Rank for Information Retrieval (Tie-Yan Liu)



https://scikit-learn.org/stable/modules/classes.html

http://rasbt.github.io/mlxtend/api_subpackages/mlxtend.evaluate/#proportion_difference

https://www.nltk.org/api/nltk.metrics.html#module-nltk.metrics.scores



https://github.com/keras-team/keras/blob/master/keras/metrics.py



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
def metrics_report(y_true, y_pred, y_pred_proba=None,
                                   metrics=["f1_macro", "accuracy", "precision_macro", "recall_macro"]):
    entries = []
    for metric in metrics:
        metric_val = globals()[metric]((y_true, y_pred)
                                      
        entries.append((metric, metric_val))
    cv_df = pd.DataFrame(entries, columns=["metric", 'metric_val'])
    return cv_df



def docs():
  """
    Generate All the methods from scratch

  """  
  from util_inspect import * 
  metric_mod = Module(name_or_path="metric.metric") 
  
  l_fun  = metric_mod.get_functions() 
  l_class = metric_mod.get_classes() 
  l_all = {*l_fun, **l_class}
  for k,x in l_all():
    print(k)





                                       
###############################################################################################################
from sklearn.metrics import *
"""
##### Classification metrics	

accuracy_score(y_true, y_pred,...)	Accuracy classification score.
auc(x, y)	Compute Area Under the Curve (AUC) using the trapezoidal rule
average_precision_score(y_true, y_score)	Compute average precision (AP) from prediction scores
balanced_accuracy_score(y_true, y_pred)	Compute the balanced accuracy
brier_score_loss(y_true, y_prob,...)	Compute the Brier score.
classification_report(y_true, y_pred)	Build a text report showing the main classification metrics
cohen_kappa_score(y1, y2,...)	Cohenâ€™s kappa: a statistic that measures inter-annotator agreement.
confusion_matrix(y_true, y_pred,...)	Compute confusion matrix to evaluate the accuracy of a classification.
dcg_score(y_true, y_score , k, ...  )	Compute Discounted Cumulative Gain.
f1_score(y_true, y_pred,...)	Compute the F1 score, also known as balanced F-score or F-measure
fbeta_score(y_true, y_pred, beta,...)	Compute the F-beta score
hamming_loss(y_true, y_pred,...)	Compute the average Hamming loss.
hinge_loss(y_true, pred_decision,...)	Average hinge loss (non-regularized)
jaccard_score(y_true, y_pred,...)	Jaccard similarity coefficient score
log_loss(y_true, y_pred , eps, ...  )	Log loss, aka logistic loss or cross-entropy loss.
matthews_corrcoef(y_true, y_pred,...)	Compute the Matthews correlation coefficient (MCC)
multilabel_confusion_matrix(y_true, ...)	Compute a confusion matrix for each class or sample
ndcg_score(y_true, y_score , k, ...  )	Compute Normalized Discounted Cumulative Gain.
precision_recall_curve(y_true, ...)	Compute precision-recall pairs for different probability thresholds
precision_recall_fscore_support(...)	Compute precision, recall, F-measure and support for each class
precision_score(y_true, y_pred,...)	Compute the precision
recall_score(y_true, y_pred,...)	Compute the recall
roc_auc_score(y_true, y_score,...)	Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
roc_curve(y_true, y_score,...)	Compute Receiver operating characteristic (ROC)
zero_one_loss(y_true, y_pred,...)	Zero-one classification loss.



##### Regression metrics	

explained_variance_score(y_true, y_pred)	Explained variance regression score function
max_error(y_true, y_pred)	max_error metric calculates the maximum residual error.
mean_absolute_error(y_true, y_pred)	Mean absolute error regression loss
mean_squared_error(y_true, y_pred,...)	Mean squared error regression loss
mean_squared_log_error(y_true, y_pred)	Mean squared logarithmic error regression loss
median_absolute_error(y_true, y_pred)	Median absolute error regression loss
r2_score(y_true, y_pred,...)	R^2 (coefficient of determination) regression score function.
mean_poisson_deviance(y_true, y_pred)	Mean Poisson deviance regression loss.
mean_gamma_deviance(y_true, y_pred)	Mean Gamma deviance regression loss.
mean_tweedie_deviance(y_true, y_pred)	Mean Tweedie deviance regression loss.



##### Multilabel ranking metrics	

coverage_error(y_true, y_score,...)	Coverage error measure
label_ranking_average_precision_score(...)	Compute ranking-based average precision
label_ranking_loss(y_true, y_score)	Compute Ranking loss measure



##### Clustering metrics	

supervised, which uses a ground truth class values for each sample.	
unsupervised, which does not and measures the â€˜qualityâ€™ of the model itself.	

adjusted_mutual_info_score(...,...)	Adjusted Mutual Information between two clusterings.
adjusted_rand_score(labels_true, ...)	Rand index adjusted for chance.
calinski_harabasz_score(X, labels)	Compute the Calinski and Harabasz score.
davies_bouldin_score(X, labels)	Computes the Davies-Bouldin score.
completeness_score(labels_true, ...)	Completeness metric of a cluster labeling given a ground truth.
cluster.contingency_matrix(...,...)	Build a contingency matrix describing the relationship between labels.
fowlkes_mallows_score(labels_true, ...)	Measure the similarity of two clusterings of a set of points.
homogeneity_completeness_v_measure(...)	Compute the homogeneity and completeness and V-Measure scores at once.
homogeneity_score(labels_true, ...)	Homogeneity metric of a cluster labeling given a ground truth.
mutual_info_score(labels_true, ...)	Mutual Information between two clusterings.
normalized_mutual_info_score(...,...)	Normalized Mutual Information between two clusterings.
silhouette_score(X, labels,...)	Compute the mean Silhouette Coefficient of all samples.
silhouette_samples(X, labels , metric  )	Compute the Silhouette Coefficient for each sample.
v_measure_score(labels_true, labels_pred)	V-measure cluster labeling given a ground truth.


##### Biclustering metrics	

consensus_score(a, b , similarity  )	The similarity of two sets of biclusters.



##### Pairwise metrics	

pairwise.additive_chi2_kernel(X , Y  )	Computes the additive chi-squared kernel between observations in X and Y
pairwise.chi2_kernel(X , Y, gamma  )	Computes the exponential chi-squared kernel X and Y.
pairwise.cosine_similarity(X , Y, ...  )	Compute cosine similarity between samples in X and Y.
pairwise.cosine_distances(X , Y  )	Compute cosine distance between samples in X and Y.
pairwise.distance_metrics()	Valid metrics for pairwise_distances.
pairwise.euclidean_distances(X , Y, ...  )	Considering the rows of X (and Y=X) as vectors, compute the distance matrix between each pair of vectors.
pairwise.haversine_distances(X , Y  )	Compute the Haversine distance between samples in X and Y
pairwise.kernel_metrics()	Valid metrics for pairwise_kernels
pairwise.laplacian_kernel(X , Y, gamma  )	Compute the laplacian kernel between X and Y.
pairwise.linear_kernel(X , Y, ...  )	Compute the linear kernel between X and Y.
pairwise.manhattan_distances(X , Y, ...  )	Compute the L1 distances between the vectors in X and Y.
pairwise.nan_euclidean_distances(X)	Calculate the euclidean distances in the presence of missing values.
pairwise.pairwise_kernels(X , Y, ...  )	Compute the kernel between arrays X and optional array Y.
pairwise.polynomial_kernel(X , Y, ...  )	Compute the polynomial kernel between X and Y.
pairwise.rbf_kernel(X , Y, gamma  )	Compute the rbf (gaussian) kernel between X and Y.
pairwise.sigmoid_kernel(X , Y, ...  )	Compute the sigmoid kernel between X and Y.
pairwise.paired_euclidean_distances(X, Y)	Computes the paired euclidean distances between X and Y
pairwise.paired_manhattan_distances(X, Y)	Compute the L1 distances between the vectors in X and Y.
pairwise.paired_cosine_distances(X, Y)	Computes the paired cosine distances between X and Y
pairwise.paired_distances(X, Y , metric  )	Computes the paired distances between X and Y.
pairwise_distances(X , Y, metric, ...  )	Compute the distance matrix from a vector array X and optional Y.
pairwise_distances_argmin(X, Y,...)	Compute minimum distances between one point and a set of points.
pairwise_distances_argmin_min(X, Y)	Compute minimum distances between one point and a set of points.
pairwise_distances_chunked(X , Y, ...  )	Generate a distance matrix chunk by chunk with optional reduction



"""
                                       
                                       
                                       
                                       

###############################################################################################################
from mlxtend.evaluate import *
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
"""








######### Multi Class ############################################################################################
from metric.model.pycm import *
"""
Overall Statistics : 

95% CI                                                            (0.30439,0.86228)
ACC Macro                                                         0.72222
AUNP                                                              0.66667
AUNU                                                              0.69444
Bennett S                                                         0.375
CBA                                                               0.47778
CSI                                                               0.17778
Chi-Squared                                                       6.6
Chi-Squared DF                                                    4
Conditional Entropy                                               0.95915
Cramer V                                                          0.5244
Cross Entropy                                                     1.59352
F1 Macro                                                          0.56515
F1 Micro                                                          0.58333
Gwet AC1                                                          0.38931
Hamming Loss                                                      0.41667
Joint Entropy                                                     2.45915
KL Divergence                                                     0.09352
Kappa                                                             0.35484
Kappa 95% CI                                                      (-0.07708,0.78675)
Kappa No Prevalence                                               0.16667
Kappa Standard Error                                              0.22036
Kappa Unbiased                                                    0.34426
Lambda A                                                          0.16667
Lambda B                                                          0.42857
Mutual Information                                                0.52421
NIR                                                               0.5
Overall ACC                                                       0.58333
Overall CEN                                                       0.46381
Overall J                                                         (1.225,0.40833)
Overall MCC                                                       0.36667
Overall MCEN                                                      0.51894
Overall RACC                                                      0.35417
Overall RACCU                                                     0.36458
P-Value                                                           0.38721
PPV Macro                                                         0.56667
PPV Micro                                                         0.58333
Pearson C                                                         0.59568
Phi-Squared                                                       0.55
RCI                                                               0.34947
RR                                                                4.0
Reference Entropy                                                 1.5
Response Entropy                                                  1.48336
SOA1(Landis & Koch)                                               Fair
SOA2(Fleiss)                                                      Poor
SOA3(Altman)                                                      Fair
SOA4(Cicchetti)                                                   Poor
SOA5(Cramer)                                                      Relatively Strong
SOA6(Matthews)                                                    Weak
Scott PI                                                          0.34426
Standard Error                                                    0.14232
TPR Macro                                                         0.61111
TPR Micro                                                         0.58333
Zero-one Loss                                                     5

"""







######### Ranking Metrics ############################################################################################
def ndcg_batch_binary(x_pred, heldout_batch, k=100):
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


def recall_recall(x_pred, heldout_batch, k=100):
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
    return ndcg(r, k=-1, method=0)                                   
                                    
def ndcg(r, k=-1, method=0):
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


                                       
##############################################################################################################                                       
def ztest():
    pass


if __name__ == "__main__":
    ztest()

                                       
                                       
                                       
                                       
