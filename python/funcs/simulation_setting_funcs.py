import numpy as np
from scipy import linalg

np.set_printoptions(linewidth=200, threshold=np.inf)


# functions for simulations
################### Randomly generate lower triangular matrix B
def B_random(p, s_B, B_value_min, B_value_max):
    lowertri_index = np.nonzero(np.tril(np.ones((p, p)), -1))
    num_lowertri_entry = len(lowertri_index[0])
    num_nonzero_entry = int(s_B * num_lowertri_entry)
    nonzero_index = np.random.choice(num_lowertri_entry, num_nonzero_entry, replace=False)
    B = np.zeros((p, p))
    B[lowertri_index[0][nonzero_index], lowertri_index[1][nonzero_index]] = np.random.uniform(B_value_min, B_value_max,
                                                                                              num_nonzero_entry)
    return B


################### Randomly generate lower triangular matrix B corresponds to a hub graph
def B_hub_func(num_hub, size_up_block, size_low_block, intersect_prop, s_B, B_value_min, B_value_max):
    if num_hub < 2:
        intersect_prop = 0
    p = num_hub + num_hub * (size_up_block + size_low_block)
    B = np.zeros((p, p))
    B_upper_list, B_lower_list, hub_B_in_vec, hub_B_out_vec = [], [], [], []
    hub_size_in = size_up_block + int(size_up_block * intersect_prop)  # point to hubs
    hub_size_out = size_low_block + int(size_low_block * intersect_prop)  # hubs point to other nodes

    for i in range(num_hub):
        B_upper_list.append(B_random(size_up_block, s_B, B_value_min, B_value_max))
        B_lower_list.append(B_random(size_low_block, s_B, B_value_min, B_value_max))
        hub_B_in_vec.append(np.random.uniform(B_value_min, B_value_max, hub_size_in))
        hub_B_out_vec.append(np.random.uniform(B_value_min, B_value_max, hub_size_out))

    for i in range(num_hub):
        start_index_up = i * size_up_block
        B[start_index_up:(start_index_up + size_up_block), start_index_up:(start_index_up + size_up_block)] = \
            B_upper_list[i]

    for i in range(num_hub):
        start_index_in_row = num_hub * size_up_block + i
        start_index_in_col = max(0, i * size_up_block - int(size_up_block * intersect_prop))
        B[start_index_in_row, start_index_in_col:(start_index_in_col + len(hub_B_in_vec[i]))] = hub_B_in_vec[i]

        start_index_out_row = num_hub * size_up_block + num_hub + max(0, i * size_low_block - int(
            size_low_block * intersect_prop))
        start_index_out_col = num_hub * size_up_block + i
        B[start_index_out_row:(start_index_out_row + len(hub_B_out_vec[i])), start_index_out_col] = hub_B_out_vec[i]

    for i in range(num_hub):
        start_index_low = num_hub * size_up_block + num_hub + i * size_low_block
        B[start_index_low:(start_index_low + size_low_block), start_index_low:(start_index_low + size_low_block)] = \
            B_lower_list[i]

    return B


################ Rescale B while keeping its support, so that the variance of X is close to the given one
def rescale_B_func(B, var_X_design, sigma2_error, tol, step_size, max_count):
    p = B.shape[1]
    I = np.identity(p)
    assert step_size < 1, "step_size must be smaller than 1"

    # Using [] will point to memory so the value of the input 'sigma2_error' will also change, so we generate a copy
    sigma2_error_copy = sigma2_error.copy()
    for i in range(p):
        # if it is a source node
        if np.sum(B[i] != 0) == 0:
            sigma2_error_copy[i] = var_X_design[i]
        # if it is not a source node
        elif np.sum(B[i] != 0) > 0:
            IB_inv_temp = linalg.solve(I - B, np.eye(p))
            var_temp = (IB_inv_temp[i] * sigma2_error_copy) @ IB_inv_temp[i].T
            rescale_temp = np.sqrt(np.abs(var_temp - sigma2_error_copy[i]))

            count_while = 0
            B_temp = B.copy()
            while abs(var_temp - var_X_design[i]) > tol / 50:
                if var_temp - var_X_design[i] > 0:
                    rescale_temp = rescale_temp * (1 + step_size)
                    B_temp = B.copy()
                    B_temp[i] = B[i] / rescale_temp

                    IB_inv_temp = linalg.solve(I - B_temp, np.eye(p))
                    var_temp = (IB_inv_temp[i] * sigma2_error_copy) @ IB_inv_temp[i].T
                else:
                    rescale_temp = rescale_temp * (1 - step_size)
                    B_temp = B.copy()
                    B_temp[i] = B[i] / rescale_temp

                    IB_inv_temp = linalg.solve(I - B_temp, np.eye(p))
                    var_temp = (IB_inv_temp[i] * sigma2_error_copy) @ IB_inv_temp[i].T
                count_while += 1
                if count_while > max_count:
                    break
            B = B_temp  # update B

    return B, sigma2_error_copy


# Generate a random or hub DAG for simulation, with permuted variable ordering
def generate_setting(dag_type, s_B, B_value_min, B_value_max, err_min, err_max, var_X_min, var_X_max,
                     p=0, num_hub=0, size_up_block=0, size_low_block=0, intersect_prop=0,
                     tol=10, step_size=0.2, max_count=100):
    if dag_type == "random":
        if p == 0:
            raise ValueError("p is needed for random dag")
        B_unscaled = B_random(p, s_B, B_value_min, B_value_max)
    elif dag_type == "hub":
        if num_hub == 0 or size_up_block == 0 or size_low_block == 0 or intersect_prop == 0:
            raise ValueError("num_hub, size_up_block, size_low_block, and intersect_prop are needed for hub dag")
        p = num_hub + num_hub * (size_up_block + size_low_block)
        B_unscaled = B_hub_func(num_hub, size_up_block, size_low_block, intersect_prop, s_B, B_value_min, B_value_max)

    sigma2_error_raw = np.random.uniform(err_min, err_max, p)  # rep(var_error,p) # do not make error variance the same!
    var_X_design = np.random.uniform(var_X_min, var_X_max, p)  # preset variance of X we want to get based on the SEM
    B_scaled, sigma2_error_new = rescale_B_func(B_unscaled, var_X_design, sigma2_error_raw, tol=tol,
                                                step_size=step_size,
                                                max_count=max_count)

    # # check that the variance of X is indeed close to the preset one
    I = np.identity(p)
    var_X_new = np.diag(np.dot(linalg.solve(I - B_scaled, np.diag(sigma2_error_new)), linalg.inv(I - B_scaled).T))
    max_diff = np.max(np.abs(var_X_new - var_X_design))
    if max_diff > tol:
        print(f"the max difference between var_X_new and var_X_design is {max_diff}")

    ordering = np.random.permutation(np.arange(p))
    Permut_mat = np.eye(p)[ordering]
    B = Permut_mat @ B_scaled @ Permut_mat.T
    sigma2_error = np.diag(sigma2_error_new[ordering])

    b = np.random.uniform(-5, 5, p)  # intercept
    return B, sigma2_error, b


# Generate n observartional and m interventional data
def generate_data(n, m, p, B, sigma2_error, b, delta_r):
    # True root causes
    RC = np.random.choice(np.arange(p), size=m, replace=True)
    I = np.identity(p)

    X_obs = np.zeros((n, p))
    X_int = np.zeros((m, p))
    for i in range(n):
        error = np.random.multivariate_normal(np.repeat(0, p), sigma2_error, 1)
        X_obs[i, :] = linalg.solve(I - B, (b + error).T).reshape(p)
    for i in range(m):
        delta = np.repeat(0, p)
        delta[RC[i]] = delta_r
        error = np.random.multivariate_normal(np.repeat(0, p), sigma2_error, 1)
        X_int[i, :] = linalg.solve(I - B, (b + error + delta).T).reshape(p)

    return X_obs, X_int, RC


def generate_data_latent(n, m, p, latent_proportion, B, sigma2_error, b, delta_r):
    # randomly sample some variables as latent
    latent_idx = np.random.choice(np.arange(p), size=int(latent_proportion * p), replace=False)
    non_latent_idx = np.setdiff1d(np.arange(p), latent_idx)

    # randomly sample true root causes from non-latent variables
    RC = np.random.choice(non_latent_idx, size=m, replace=True)

    I = np.identity(p)

    X_obs = np.zeros((n, p))
    X_int = np.zeros((m, p))
    for i in range(n):
        error = np.random.multivariate_normal(np.repeat(0, p), sigma2_error, 1)
        X_obs[i, :] = linalg.solve(I - B, (b + error).T).reshape(p)
    for i in range(m):
        delta = np.repeat(0, p)
        delta[RC[i]] = delta_r
        error = np.random.multivariate_normal(np.repeat(0, p), sigma2_error, 1)
        X_int[i, :] = linalg.solve(I - B, (b + error + delta).T).reshape(p)

    # remove latent variables
    X_obs_new = X_obs[:, non_latent_idx]
    X_int_new = X_int[:, non_latent_idx]
    # update RC_idx
    RC_new = np.zeros(m)
    for i in range(m):
        RC_new[i] = RC[i] - np.sum(latent_idx < RC[i])

    return X_obs_new, X_int_new, RC_new


# Generate n observartional and m interventional data, with different error types
def generate_data_errorType(n, m, p, B, b, delta_r, error_type, sigma2_error=None):
    # True root causes
    RC = np.random.choice(np.arange(p), size=m, replace=True)
    I = np.identity(p)

    X_obs = np.zeros((n, p))
    X_int = np.zeros((m, p))
    for i in range(n):
        if error_type == "Gaussian":
            error = np.random.multivariate_normal(np.repeat(0, p), sigma2_error, 1)
        elif error_type == "Uniform":
            error = np.random.uniform(1, 10, p)
        X_obs[i, :] = linalg.solve(I - B, (b + error).T).reshape(p)
    for i in range(m):
        delta = np.repeat(0, p)
        delta[RC[i]] = delta_r
        if error_type == "Gaussian":
            error = np.random.multivariate_normal(np.repeat(0, p), sigma2_error, 1)
        elif error_type == "Uniform":
            error = np.random.uniform(1, 10, p)
        X_int[i, :] = linalg.solve(I - B, (b + error + delta).T).reshape(p)

    return X_obs, X_int, RC
