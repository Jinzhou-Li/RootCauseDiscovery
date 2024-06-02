import numpy as np
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from funcs.root_cause_discovery_funcs import *
from funcs.simulation_setting_funcs import *

# parameters for simulations (run in different cores)
s_B = float(sys.argv[1])      # 0.2, 0.4, or 0.6
int_mean = int(sys.argv[2])   # 10, 15, or 20
dag_type = sys.argv[3]
dimreduce_method = sys.argv[4]
ncore = int(sys.argv[5])
seed_B = int(sys.argv[6])  # 1 to 10
seed_m = int(sys.argv[7])  # 1 to 10
outdir = sys.argv[8]          # output directory
if not os.path.isdir(outdir):
    os.mkdir(outdir)

# other fixed parameters
n = 100
nshuffles = 5

num_hub = 10 # same number of upper and lower blocks
size_up_block = 30
size_low_block = 20
intersect_prop = 0.3

verbose = False
y_idx_z_threshold = 0

int_sd = 1
B_value_min = -1
B_value_max = 1
err_min = 1
err_max = 5
var_X_min = 10
var_X_max = 50

# generate data
np.random.seed(seed_B)
B, sigma2_error, b = generate_setting(dag_type, s_B, B_value_min, B_value_max, err_min, err_max, var_X_min, var_X_max,
                                      num_hub=num_hub, size_up_block=size_up_block,
                                      size_low_block=size_low_block, intersect_prop=intersect_prop)
p = len(b)

# Precision matrix to get true MB
I = np.eye(p)
Precision_mat = (I - B).T @ np.diag(1/np.diag(sigma2_error)) @ (I - B)

### Start simulations
np.random.seed(seed_m)
X_obs, X_int, RC = generate_data(n, 1, p, B, sigma2_error, b, int_mean, int_sd)
X_int = X_int[0,:]

# z score method
Zscores = zscore(X_obs, X_int)

# our main score method
CholScores_highdim, select_len = root_cause_discovery_highsim_parallel(X_obs, X_int, dimreduce_method, ncore, y_idx_z_threshold, nshuffles, verbose)
CholScores_highdim_MB, select_len_MB = root_cause_discovery_highsim_parallel(X_obs, X_int, dimreduce_method, ncore, y_idx_z_threshold, nshuffles, verbose, Precision_mat)

# save simulation result
outfile = os.path.join(outdir, "hd" + dag_type + dimreduce_method + "_s" + str(int(s_B*10)) +
                       'int' + str(int_mean) + "seedB" + str(seed_B) +
                       "seedm" + str(seed_m) + '.npz')
np.savez(outfile, array1=RC, array2=Zscores, array3=CholScores_highdim, array4=CholScores_highdim_MB,
         array5=select_len, array6=select_len_MB)