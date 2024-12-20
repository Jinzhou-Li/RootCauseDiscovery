import numpy as np
import random
from numpy import linalg as LA
from sklearn.linear_model import LassoCV, lasso_path
import warnings  # ignore the warnings
from sklearn.covariance import ShrunkCovariance
from joblib import Parallel, delayed
from lingam import DirectLiNGAM
from tqdm import tqdm

# generate thresholds to determine aberrant set, make sure no repeated permutations later.
def get_aberrant_thresholds(z_vec, thre_min=0.1, thre_max=5, thre_seq=0.2):
    thresholds_raw = np.arange(thre_min, thre_max, thre_seq)
    thre_new = list()
    count_temp = -1
    for threshold in thresholds_raw:
        if np.sum(z_vec >= threshold) > 0:
            if np.sum(z_vec <= threshold) != count_temp:
                count_temp = np.sum(z_vec <= threshold)
                thre_new.append(threshold)
    thre_new = np.array(thre_new)
    return thre_new

# `X_obs` and `X_int` are matrices
def zscore(X_obs, X_int):
    if X_int.ndim == 1:
        return zscore_vec(X_obs, X_int)
    n, p = X_int.shape
    assert X_obs.shape[1] == p, "Both inputs should have same number of variables"
    # observational data's mean and std
    mu = np.mean(X_obs, axis=0)
    sigma = np.std(X_obs, axis=0, ddof=1)  # sample std
    # compute squared Z scores for each patient in interventional data
    zscore_intv = np.zeros((n, p))
    for i in range(n):
        zs = [(abs((X_int[i, j] - mu[j]) / sigma[j])) ** 2 for j in range(p)]
        zscore_intv[i, :] = zs
    return zscore_intv


# `X_obs` is matrix, `X_int` is a vector
def zscore_vec(X_obs, X_int):
    ngenes = len(X_int)
    assert X_obs.shape[1] == ngenes, "Number of genes mismatch"
    # observational data's mean and std
    mu = np.mean(X_obs, axis=0)
    sigma = np.std(X_obs, axis=0, ddof=1)
    # compute squared Z scores for each patient in interventional data
    zs = np.array([(abs((X_int[i] - mu[i]) / sigma[i])) ** 2 for i in range(ngenes)])
    return zs


# `z` is a vector
def compute_permutations(z, threshold=1.5, nshuffles=5):
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
    if not isinstance(perm, np.ndarray):
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
        # it is possible to use other estimators here
        cov = ShrunkCovariance().fit(X_obs_perm)
        sigma = cov.covariance_

    # ensure positive-definite
    min_eigenvalue = min(LA.eigvals(sigma))
    if min_eigenvalue < 1e-6:
        #sigma = sigma + abs(min_eigenvalue) + 1e-6
        sigma = sigma + (abs(min_eigenvalue) + 1e-6)*np.eye(p)
    # compute cholesky
    L = LA.cholesky(sigma)
    # solve for Xtilde in L*Xtilde = X_int_perm - mu
    X_tilde = np.linalg.solve(L, X_int_perm - mu)
    # undo the permutations
    X_tilde = X_tilde[perm.argsort()]
    return abs(X_tilde)


# Main root cause discovery function (Algo 2 in the paper)
def root_cause_discovery_main(X_obs, X_int, nshuffles=1, thresholds=None, verbose=True):
    p = X_obs.shape[1]
    assert p == len(X_int), "Number of variables mismatch"
    # compute z scores
    z = zscore(X_obs, X_int)

    if thresholds is None:
        thresholds = get_aberrant_thresholds(z, thre_min=0.1, thre_max=5, thre_seq=0.2)

    root_cause_score = np.zeros(p)
    for threshold in thresholds:
        permutations = compute_permutations(z, threshold=threshold, nshuffles=nshuffles)
        if verbose:
            print("Trying", len(permutations), "permutations for threshold", threshold)

        # try all permutations to calculate 'Xtilde'
        for perm in permutations:
            Xtilde = root_cause_discovery(X_obs, X_int, perm)
            sorted_X = sorted(Xtilde)
            OneNonZero_quantification = (sorted_X[-1] - sorted_X[-2]) / sorted_X[-2]
            max_index = np.argmax(Xtilde)
            if root_cause_score[max_index] < OneNonZero_quantification:
                root_cause_score[max_index] = OneNonZero_quantification

    # assign final root cause score for variables that never had maximal Xtilde_i
    idx2 = np.where(root_cause_score == 0)[0]
    if len(idx2) != 0:
        idx1 = np.where(root_cause_score != 0)[0]
        max_RC_score_idx2 = np.min(root_cause_score[idx1])/2
        root_cause_score[idx2] = z[idx2] / (np.max(z[idx2]) / max_RC_score_idx2)
    return root_cause_score


def reduce_dimension(y_idx, X_obs, X_int, verbose=True):
    # response and design matrix for Lasso
    y = X_obs[:, y_idx]
    X = np.delete(X_obs, y_idx, axis=1)
    n = len(y)

    # fit CV-lasso
    with warnings.catch_warnings():  # ignore the convergence warnings from 'enet_path'
        warnings.simplefilter("ignore")
        lasso_cv = LassoCV().fit(X, y)
        beta_final = lasso_cv.coef_
        if np.sum(beta_final != 0) <= 1:  # in this case return n/2 variables
            _, coef_path, _ = lasso_path(X, y)
            num_nonzeros = np.sum((coef_path != 0), axis=0)
            alpha_idx = np.argmin(np.abs(num_nonzeros - n / 2))
            beta_final = coef_path[:, alpha_idx]

    nz = np.count_nonzero(beta_final)
    if verbose:
        print("Treat", y_idx, "as response, found ", nz, " non-zero entries")
    # for non-zero idx, find the original indices in X_obs, and don't forget to include y_idx
    selected_idx = np.nonzero(beta_final)[0]
    selected_idx = np.array([idx + 1 if idx >= y_idx else idx for idx in selected_idx])
    selected_idx = np.append(selected_idx, y_idx)
    # return the subset of variables of X_obs that were selected
    X_obs_new = X_obs[:, selected_idx]

    if len(X_int.shape) == 2:
        X_int_sample_new = X_int[:, selected_idx]
    elif len(X_int.shape) == 1:
        X_int_sample_new = X_int[selected_idx]
    # return
    return X_obs_new, X_int_sample_new, selected_idx


# Function to parallel
def process_y_idx_rcd(
        y_idx,
        X_obs,
        X_int,
        nshuffles=1,
        verbose=True,
        Precision_mat=None):
    if Precision_mat is None:
        X_obs_new, X_int_sample_new, selected_idx = reduce_dimension(y_idx, X_obs, X_int, verbose)
        select_len_y = len(selected_idx)
    else:
        MB = np.where(Precision_mat[:, y_idx] != 0)[0]
        MB = np.delete(MB, np.where(MB == y_idx)[0][0])
        MB = np.append(MB, y_idx)  # put y_idx in the end
        X_obs_new = X_obs[:, MB]
        X_int_sample_new = X_int[MB]
        select_len_y = len(MB)

    z_new = zscore(X_obs_new, X_int_sample_new)

    thresholds = get_aberrant_thresholds(z_new, thre_min=0.1, thre_max=5, thre_seq=0.2)
    for threshold in thresholds:
        permutations = compute_permutations(z_new, threshold=threshold, nshuffles=nshuffles)
        if verbose:
            print("Trying", len(permutations), "permutations for threshold", threshold)

        # try all permutations to calculate 'Xtilde' and update 'best_OneNonZero_quantification'
        root_cause_score_y = 0
        for perm in permutations:
            Xtilde = root_cause_discovery(X_obs_new, X_int_sample_new, perm)
            sorted_X = sorted(Xtilde)
            OneNonZero_quantification = (sorted_X[-1] - sorted_X[-2]) / sorted_X[-2]

            max_index = np.argmax(Xtilde)
            if max_index == (len(Xtilde)-1):  # recall that the last variable is the one treated as response
                if root_cause_score_y < OneNonZero_quantification:
                    root_cause_score_y = OneNonZero_quantification

    return y_idx, root_cause_score_y, select_len_y

# Root cause discovery function for high-dimensional case (Algo 3 in the paper)
def root_cause_discovery_highdim_parallel(
        X_obs,
        X_int,
        n_jobs,
        y_idx_z_threshold=1.5,
        nshuffles=1,
        verbose=True,
        Precision_mat=None):  # Parameter to specify the number of cores
    n, p = X_obs.shape
    z = zscore(X_obs, X_int)
    y_indices = np.where(z > y_idx_z_threshold)[0]

    # Parallelize the processing of y_indices
    results = Parallel(n_jobs=n_jobs)(delayed(process_y_idx_rcd)(y_idx, X_obs, X_int, nshuffles,
                                                                 verbose, Precision_mat) for y_idx in y_indices)
    root_cause_score = np.zeros(p)
    select_len = np.zeros(p)
    for y_idx, score, length in results:
        root_cause_score[y_idx] = score
        select_len[y_idx] = length

    # assign final root cause score for variables that never had maximal Xtilde_i
    idx2 = np.where(root_cause_score == 0)[0]
    idx1 = np.where(root_cause_score != 0)[0]
    if len(idx2) != 0:
        if len(idx1) != 0:
            max_RC_score_idx2 = np.min(root_cause_score[idx1])/2
            root_cause_score[idx2] = z[idx2] / (np.max(z[idx2]) / max_RC_score_idx2)
        else:
            root_cause_score = z

    return root_cause_score, select_len

# LiNGAM-based method 1 (aberrant set + causal ordering): for comparison in simulations
def get_rank_LiNGAM(X_obs, X_int, RC):
    Zscores = zscore(X_obs, X_int)

    # Fit LiNGAM
    model = DirectLiNGAM()
    model.fit(X_obs)
    causal_order_lingam = np.array(model.causal_order_, dtype=int)

    thre_abe_opt = Zscores[RC]  # this is the 'optimal threshold': gaurantees better than z-score (unavailable in practice)
    thre_abe_1 = 2
    thre_abe_2 = 5
    thre_abe_3 = 10

    rank_LiNGAM_opt = get_rank_LiNGAM_one_thre(Zscores, thre_abe_opt, causal_order_lingam)
    rank_LiNGAM_1 = get_rank_LiNGAM_one_thre(Zscores, thre_abe_1, causal_order_lingam)
    rank_LiNGAM_2 = get_rank_LiNGAM_one_thre(Zscores, thre_abe_2, causal_order_lingam)
    rank_LiNGAM_3 = get_rank_LiNGAM_one_thre(Zscores, thre_abe_3, causal_order_lingam)

    return rank_LiNGAM_opt, rank_LiNGAM_1, rank_LiNGAM_2, rank_LiNGAM_3, thre_abe_opt

def get_rank_LiNGAM_one_thre(Zscores, thre_abe, causal_order_lingam):
    p = len(Zscores)
    abe_set = np.where(Zscores >= thre_abe)[0]

    remain_set = np.setdiff1d(np.array(range(p)), abe_set)

    Zscores_remian_set = Zscores[list(remain_set)]
    rank_remain_set = list(remain_set[np.argsort(-Zscores_remian_set)])

    abe_rank = []
    for i in causal_order_lingam:
        if i in abe_set:
            abe_rank.append(i)

    Lin_rank = abe_rank + rank_remain_set
    return np.array(Lin_rank, dtype=int)

# LiNGAM-based method 2 (invariant conditional on Pa): for comparison in simulations
def get_InvaScore(X_obs, X_int, Oracle=False, B=None, sigma_error=None):
    p = X_obs.shape[1]
    if Oracle == True:
        B_est = B
    else:
        # Fit LiNGAM
        model = DirectLiNGAM()
        model.fit(X_obs)
        B_est = model.adjacency_matrix_

    InvaScore = np.zeros(p)
    for j in range(p):
        y_train = X_obs[:, j]
        y_true = X_int[j]

        Pa_est = list(np.where(B_est[j, :] != 0)[0])

        if len(Pa_est) == 0:
            if Oracle == True:
                InvaScore[j] = np.abs(y_true - np.mean(y_train)) / sigma_error[j]
            else:
                InvaScore[j] = np.abs(y_true - np.mean(y_train)) / np.std(y_train)
        else:
            X_train = X_obs[:, Pa_est]
            X_test = X_int[Pa_est].reshape(1, -1)
            with warnings.catch_warnings():  # ignore the convergence warnings
                warnings.simplefilter("ignore")
                lasso_cv = LassoCV(cv=10)
                lasso_cv.fit(X_train, y_train)
            y_test = lasso_cv.predict(X_test)

            if Oracle == True:
                InvaScore[j] = np.abs(y_true - y_test[0]) / sigma_error[j]
            else:
                std_est_Pa = np.std(y_train - lasso_cv.predict(X_train))
                InvaScore[j] = np.abs(y_true - y_test[0]) / std_est_Pa
    return InvaScore