import numpy as np
import random
from numpy import linalg as LA
from sklearn.linear_model import LassoCV, lasso_path


# `X_obs` and `X_int` are matrices
def zscore(X_obs, X_int):
    if X_int.ndim == 1:
        return zscore_vec(X_obs, X_int)
    n, p = X_int.shape
    assert X_obs.shape[1] == p, "Both inputs should have same number of variables"
    # observational data's mean and std
    mu = np.mean(X_obs, axis=0)
    sigma = np.std(X_obs, axis=0, ddof=1) # sample std
    # compute squared Z scores for each patient in interventional data
    zscore_intv = np.zeros((n, p))
    for i in range(n):
        zs = [(abs((X_int[i, j] - mu[j]) / sigma[j]))**2 for j in range(p)]
        zscore_intv[i,:] = zs
    return zscore_intv

# `X_obs` is matrix, `X_int` is vector
def zscore_vec(X_obs, X_int):
    ngenes = len(X_int)
    assert X_obs.shape[1] == ngenes, "Number of genes mismatch"
    # observational data's mean and std
    mu = np.mean(X_obs, axis=0)
    sigma = np.std(X_obs, axis=0)
    # compute squared Z scores for each patient in interventional data
    zs = [(abs((X_int[i] - mu[i]) / sigma[i]))**2 for i in range(ngenes)]
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


# Basic root cause discovery function with 'perm' as an input
# `X_obs` is a matrix, `X_int` a vector, `perm` a permutation vector
def root_cause_discovery(X_obs, X_int, perm):
    if isinstance(perm, np.ndarray)==False:
        perm = np.array(perm)
    n, p = X_obs.shape
    assert p == len(X_int), "dimensions mismatch!"
    assert sorted(perm) == list(range(0, p)), "perm is not a permutation vector"
    # permute X_obs and X_int
    X_obs_perm = X_obs[:, perm]
    X_int_perm = X_int[perm]
    # estimate covariance and mean
    mu = np.mean(X_obs_perm, axis=0)
    if n > p:
        sigma = np.cov(X_obs_perm.transpose())
    else:
        raise Exception("covariance shrinkage not implemented")
    # ad-hoc way to ensure PSD
    min_eigenvalue = min(LA.eigvals(sigma))
    if min_eigenvalue < 1e-6:
        sigma = sigma + abs(min_eigenvalue) + 1e-6
    # compute cholesky
    L = LA.cholesky(sigma)
    # solve for Xtilde in L*Xtilde = X_int_perm - mu
    X_tilde = np.linalg.solve(L, X_int_perm - mu)
    # undo the permutations
    X_tilde = X_tilde[perm.argsort()]
    return abs(X_tilde)

# main root cause discovery function (Algo 3 in the paper)
def root_cause_discovery_main(X_obs, X_int, thresholds, nshuffles=1, verbose=True):
    p = X_obs.shape[1]
    assert p == len(X_int), "Number of genes mismatch"
    # compute z scores
    z = zscore(X_obs, X_int)

    root_cause_score = np.zeros(p)
    for threshold in thresholds:
        # compute permutations to try
        permutations = compute_permutations(z, threshold=threshold, nshuffles=nshuffles)
        if verbose:
            print("Trying", len(permutations), "permutations for threshold", threshold)

        # try all permutations to calculate 'Xtilde'
        Xtilde_all = []
        for perm in permutations:
            Xtilde = root_cause_discovery(X_obs, X_int, perm)
            Xtilde_all.append(Xtilde)
        # update root cause scores
        for i in range(len(Xtilde_all)):
            sorted_X = sorted(Xtilde_all[i])
            OneNonZero_quantification = (sorted_X[-1] - sorted_X[-2]) / sorted_X[-2]
            max_index = np.argmax(Xtilde_all[i])
            if root_cause_score[max_index] < OneNonZero_quantification:
                root_cause_score[max_index] = OneNonZero_quantification
    # assign final root cause score for variables that never had maximal Xtilde_i
    idx2 = np.where(root_cause_score == 0)[0]
    if len(idx2) != 0:
        idx1 = np.where(root_cause_score != 0)[0]
        max_RC_score_idx2 = np.min(root_cause_score[idx1]) - 0.0001
        z_array = np.array(z)
        root_cause_score[idx2] = z_array[idx2] / (np.max(z_array[idx2]) / max_RC_score_idx2)
    return root_cause_score


# this is same function as 'reduce_gene'
def reduce_dimension(y_idx, X_obs, X_int, method, verbose=True):
    n, p = X_obs.shape
    # response and design matrix for Lasso
    y = X_obs[:, y_idx]
    X = np.delete(X_obs, y_idx, axis=1)
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
    # for non-zero idx, find the original indices in X_obs, and don't forget to include y_idx
    selected_idx = np.nonzero(beta_final)[0]
    selected_idx = np.array([idx + 1 if idx >= y_idx else idx for idx in selected_idx])
    selected_idx = np.append(selected_idx, y_idx)
    # return the subset of variables of X_obs that were selected
    X_obs_new = X_obs[:, selected_idx]
    X_int_sample_new = X_int[selected_idx]
    # return
    return X_obs_new, X_int_sample_new, selected_idx

def root_cause_discovery_high_dimensional(
        X_obs,
        X_int,
        method,
        y_idx_z_threshold=1.5,
        permutation_thresholds=np.arange(0.1, 5, 0.2),
        nshuffles=1,
        verbose=True):
    n, p = X_obs.shape
    z = zscore(X_obs, X_int)
    y_indices = np.where(z > y_idx_z_threshold)[0]
    if verbose:
        print(f"Trying {len(y_indices)} y_indices")
    # check for desired pattern
    record_match = np.zeros(len(y_indices))
    for (i, y_idx) in enumerate(y_indices):
        best_permutation_score = 0.0
        best_Xtilde = []
        # treat one column of X_obs as response
        X_obs_new, X_int_sample_new, _ = reduce_dimension(
            y_idx, X_obs, X_int, method
        )
        # try different permutations
        for thrs in permutation_thresholds:
            cholesky_score = root_cause_discovery_one_subject_all_perm(X_obs_new,
                                                            X_int_sample_new,
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
    # for (i, zi) in enumerate(z):
    #     used_as_response = i in y_indices
    #     used_as_response_and_matched = (used_as_response and record_match[np.where(y_indices == i)] == 1)[0]
    #     if used_as_response:
            

    # todo: 

    return best_Xtilde, y_indices
