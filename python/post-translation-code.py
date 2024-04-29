# using chatGPT, we translated our julia code into the following python code

import numpy as np
import pandas as pd
import random

from numpy import linalg as LA
from scipy.linalg import cholesky
from sklearn.linear_model import LassoCV, Lasso


# `obs_data` and `intv_data` are matrices
def zscore(obs_data, intv_data):
    n, p = intv_data.shape
    assert obs_data.shape[1] == p, "Both inputs should have same number of variables"
    # observational data's mean and std
    mu = np.mean(obs_data, axis=0)
    sigma = np.std(obs_data, axis=0, ddof=1) # sample std
    # compute squared Z scores for each patient in interventional data
    zscore_intv = np.zeros((n, p))
    for i in range(n):
        zs = [(abs((intv_data[i, j] - mu[j]) / sigma[j]))**2 for j in range(p)]
        zscore_intv[i,:] = zs
    return zscore_intv

# `z` is a vector
def compute_permutations(z, threshold=2, nshuffles=1):
    # subset of variables indices that are abberent
    idx_abberent = [i for i, x in enumerate(z) if x > threshold]
    idx_abberent_copy = idx_abberent.copy()
    # subset of variables indices that are normal
    idx_normal = list(set(range(len(z))) - set(idx_abberent))
    # generate feasible permutations accroding to our theoretical results
    perms = []
    for i in range(len(idx_abberent)):
        for _ in range(nshuffles):
            random.shuffle(idx_normal)
            random.shuffle(idx_abberent)
            target = idx_abberent_copy[i]
            index = idx_abberent.index(target)
            idx_abberent[0], idx_abberent[index] = idx_abberent[index], idx_abberent[0]
            perms.append(idx_normal + idx_abberent)
    return perms


# `Xobs` is a matrix, `Xint` a vector, `perm` a permutation vector
def root_cause_discovery(Xobs, Xint, perm):
    if isinstance(perm, np.ndarray)==False:
        perm = np.array(perm)
    n, p = Xobs.shape
    assert p == len(Xint), "dimensions mismatch!"
    assert sorted(perm) == list(range(0, p)), "perm is not a permutation vector"
    # permute Xobs and Xint
    Xobs_perm = Xobs[:, perm]
    Xint_perm = Xint[perm]
    # estimate covariance and mean
    mu = np.mean(Xobs_perm, axis=0)
    if n > p:
        sigma = np.cov(Xobs_perm.transpose())
    else:
        raise Exception("covariance shrinkage not implemented")
    # ad-hoc way to ensure PSD
    min_eigenvalue = min(LA.eigvals(sigma))
    if min_eigenvalue < 1e-6:
        sigma = sigma + abs(min_eigenvalue) + 1e-6
    # compute cholesky
    L = LA.cholesky(sigma)
    # solve for Xtilde in L*Xtilde = Xint - mu
    X_tilde = np.linalg.solve(L, Xint - mu)
    # undo the permutations
    X_tilde = X_tilde[perm.argsort()]
    return abs(X_tilde)


def root_cause_discovery_one_subject_all_perm(X_obs, X_int, threshold, nshuffles=1):
    n, p = X_obs.shape
    assert p == len(X_int), "dimensions mismatch!"
    # compute Z scores
    z = zscore(X_obs, np.matrix(X_int))[0]
    # compute permutations to try
    permutations = compute_permutations(z, threshold=threshold, nshuffles=nshuffles)
    # try all permutations
    print("Trying", len(permutations), "permutations")
    X_tilde_all = np.zeros((p, len(permutations)))
    for i, perm in enumerate(permutations):
        X_tilde = root_cause_discovery(X_obs, X_int, perm)
        X_tilde_all[:, i] = X_tilde
    return X_tilde_all


# `Xall` is a vector of vectors
def find_largest(Xall):
    largest = [sorted(X, reverse=True)[0] for X in Xall]
    largest_idx = [np.argmax(X) for X in Xall]
    return largest, largest_idx


# `Xall` is a vector of vectors
def find_second_largest(Xall):
    second_largest = [sorted(X, reverse=True)[1] for X in Xall]
    return second_largest


def reduce_genes(patient_id, y_idx, Xobs, Xint, ground_truth, method):
    n, p = Xobs.shape
    # response and design matrix for Lasso
    y = Xobs[:, y_idx]
    X = np.delete(Xobs, y_idx, axis=0)
    # fit lasso
    beta_final = None
    if method == "cv":
        lasso_cv = LassoCV().fit(X, y)
        beta_final = lasso_cv.coef_
    elif method == "largest_support":
        lasso = Lasso().fit(X, y)
        beta_final = lasso.coef_
    else:
        raise ValueError("method should be `cv` or `largest_support`")
    nz = np.count_nonzero(beta_final)
    print("Lasso found ", nz, " non-zero entries")
    # for non-zero idx, find the original indices in Xobs, and don't forget to include y_idx
    selected_idx = np.nonzero(beta_final)[0]
    selected_idx = np.array([idx + 1 if idx >= y_idx else idx for idx in selected_idx])
    selected_idx = np.append(selected_idx, y_idx)
    # return the subset of variables of Xobs that were selected
    Xobs_new = Xobs[:, selected_idx]
    i = ground_truth[ground_truth["Patient ID"] == patient_id]
    Xint_sample_new = Xint[i, selected_idx]
    # return
    return Xobs_new, Xint_sample_new, selected_idx

