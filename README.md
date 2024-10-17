# RootCauseDiscovery

This repository contains Python and Julia codes for implementing the root cause discovery methods in paper ["Root cause discovery via permutations and Cholesky decomposition"](https://arxiv.org/abs/2410.12151). We use the proposed root cause discovery method to discover the disease-causing gene of rare disease patients affected by a monogenic disorder.

## Python code for Root Cause Discovery

We used python 3.12.5 for running the python code. You need to have the following python packages installed: 
+ `numpy`
+ `scipy`
+ `random`
+ `sklearn`
+ `warnings`
+ `joblib`
+ `lingam`
+ `tqdm`

After these packages are installed, you can load the root cause discovery python codes located in `python/funcs`. 

## Julia package for Root Cause Discovery

To use our Julia package, download [Julia](https://julialang.org/downloads/). Within Julia, install the package via
```
using Pkg
Pkg.add(PackageSpec(url="https://github.com/biona001/RootCauseDiscovery.git", subdir="julia"))
```

We used this package for our real data analysis due to Julia's computational efficiency.

## Examples

+ **Python users:** See `python/Example.ipynb` for 2 simulated examples. 
+ **Julia users:** See `julia/notebook/example.ipynb` for a simulated example and a real-data example.

