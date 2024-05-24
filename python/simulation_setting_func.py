import numpy as np
from scipy import linalg

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


################ Rescale B, but keep its support, so that the variance of X is close to the given one
def rescale_B_func(B, var_X_design, sigma2_error, tol, step_size, max_count):
    p = B.shape[1]
    I = np.identity(p)
    assert step_size < 1, "step_size must be smaller than 1"

    # Note that using [] will point to memory so the value of the input 'sigma2_error' will also change, so we generate a copy
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
            while abs(var_temp - var_X_design[i]) > tol:
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


# Generate n observartional data and m interventional data

def generate_data(n, m, p, B, sigma2_error, b, int_mean, int_sd):
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
        delta[RC[i]] = np.random.normal(int_mean, int_sd, 1)
        error = np.random.multivariate_normal(np.repeat(0, p), sigma2_error, 1)
        X_int[i, :] = linalg.solve(I - B, (b + error + delta).T).reshape(p)

    return X_obs, X_int, RC