# using chatGPT, we translated our julia code into the following python code

import numpy as np
import pandas as pd
import random

from numpy import linalg as LA
from scipy.linalg import cholesky
from sklearn.linear_model import LassoCV, Lasso, lasso_path


# `obs_data` and `intv_data` are matrices
def zscore(obs_data, intv_data):
    if intv_data.ndim == 1:
        return zscore_vec(obs_data, intv_data)
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

# `obs_data` is matrix, `intv_data` is vector
def zscore_vec(obs_data, intv_data):
    ngenes = len(intv_data)
    assert obs_data.shape[1] == ngenes, "Number of genes mismatch"
    # observational data's mean and std
    mu = np.mean(obs_data, axis=0)
    sigma = np.std(obs_data, axis=0)
    # compute squared Z scores for each patient in interventional data
    zs = [(abs((intv_data[i] - mu[i]) / sigma[i]))**2 for i in range(ngenes)]
    return zs

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
    # solve for Xtilde in L*Xtilde = Xint_perm - mu
    X_tilde = np.linalg.solve(L, Xint_perm - mu)
    # undo the permutations
    X_tilde = X_tilde[perm.argsort()]
    return abs(X_tilde)


def root_cause_discovery_one_subject_all_perm(Xobs, Xint, threshold, nshuffles=1, verbose=True):
    p = Xobs.shape[1]
    assert p == len(Xint), "Number of genes mismatch"
    # compute z scores
    z = zscore(Xobs, Xint)
    # compute permutations to try
    permutations = compute_permutations(z, threshold=threshold, nshuffles=nshuffles)
    if verbose: 
        print("Trying", len(permutations), "permutations")
    # try all permutations
    Xtilde_all = []
    for perm in permutations:
        Xtilde = root_cause_discovery(Xobs, Xint, perm)
        Xtilde_all.append(Xtilde)
    # assign the final root cause score for each variable
    root_cause_score = np.zeros(p)
    for i in range(len(Xtilde_all)):
        sorted_X = sorted(Xtilde_all[i])
        nonzero_quant_ratio = (sorted_X[-1] - sorted_X[-2]) / sorted_X[-2]
        max_index = np.argmax(Xtilde_all[i])
        if root_cause_score[max_index] < nonzero_quant_ratio:
            root_cause_score[max_index] = nonzero_quant_ratio
    # assign final root cause score for variables that never have maximal Xtilde_i
    idx1 = np.where(root_cause_score != 0)[0]
    idx2 = np.where(root_cause_score == 0)[0]
    max_RC_score_idx2 = np.min(root_cause_score[idx1]) - 0.0001
    z_array = np.array(z)
    root_cause_score[idx2] = z_array[idx2] / (np.max(z_array[idx2]) / max_RC_score_idx2)
    return root_cause_score


# `Xall` is a vector of vectors
def find_largest(Xall):
    largest = [sorted(X, reverse=True)[0] for X in Xall]
    largest_idx = [np.argmax(X) for X in Xall]
    return largest, largest_idx


# `Xall` is a vector of vectors
def find_second_largest(Xall):
    second_largest = [sorted(X, reverse=True)[1] for X in Xall]
    return second_largest


# this is same function as reduce_gene
def reduce_dimension(y_idx, Xobs, Xint, method, verbose=True):
    n, p = Xobs.shape
    # response and design matrix for Lasso
    y = Xobs[:, y_idx]
    X = np.delete(Xobs, y_idx, axis=1)
    # fit lasso
    if method == "cv":
        lasso_cv = LassoCV().fit(X, y)
        beta_final = lasso_cv.coef_
    elif method == "largest_support":
        _, coef_path, _ = lasso_path(X, y)
        beta_final = coef_path[:, -1]
    else:
        raise ValueError("method should be `cv` or `largest_support`")
    nz = np.count_nonzero(beta_final)
    if verbose:
        print("Lasso found ", nz, " non-zero entries")
    # for non-zero idx, find the original indices in Xobs, and don't forget to include y_idx
    selected_idx = np.nonzero(beta_final)[0]
    selected_idx = np.array([idx + 1 if idx >= y_idx else idx for idx in selected_idx])
    selected_idx = np.append(selected_idx, y_idx)
    # return the subset of variables of Xobs that were selected
    Xobs_new = Xobs[:, selected_idx]
    Xint_sample_new = Xint[selected_idx]
    # return
    return Xobs_new, Xint_sample_new, selected_idx

def root_cause_discovery_high_dimensional(
        Xobs, 
        Xint,
        method,
        y_idx_z_threshold=1.5,
        permutation_thresholds=np.arange(0.1, 5, 0.2),
        nshuffles=1,
        verbose=True):
    n, p = Xobs.shape
    z = zscore(Xobs, Xint)
    y_indices = np.where(z > y_idx_z_threshold)[0]
    if verbose:
        print(f"Trying {len(y_indices)} y_indices")
    # check for desired pattern
    record_match = np.zeros(len(y_indices))
    for (i, y_idx) in enumerate(y_indices):
        best_permutation_score = 0.0
        best_Xtilde = []
        # treat one column of Xobs as response
        Xobs_new, Xint_sample_new, _ = reduce_dimension(
            y_idx, Xobs, Xint, method
        )
        # try different permutations
        for thrs in permutation_thresholds:
            cholesky_score = root_cause_discovery_one_subject_all_perm(Xobs_new, 
                                                            Xint_sample_new, 
                                                            threshold=thrs,
                                                            nshuffles=nshuffles,
                                                            verbose=verbose)
            sorted_X = sort(cholesky_score)
            current_score = (sorted_X[-1] - sorted_X[-2]) / sorted_X[-2]
            if current_score > best_permutation_score:
                best_permutation_score = current_score
                best_Xtilde = cholesky_score
        # check if the discovered "root cause" in best_Xtilde matches y_idx
        if np.argmax(best_Xtilde) == y_idx:
            record_match[i] = 1
    # compare z score and record_match to determine rank of each var
    original_rank = z.argsort().argsort()
    offset = len(setdiff1d(range(p), y_indices)) + np.sum(record_match == 0)
    variable_rank = []
    for (i, zi) in enumerate(z):
        used_as_response = i in y_indices
        used_as_response_and_matched = (used_as_response and record_match[np.where(y_indices == i)] == 1)[0]
        if used_as_response:
            

    # todo: 

    return best_Xtilde, y_indices
