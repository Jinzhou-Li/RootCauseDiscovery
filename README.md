# RootCauseDiscovery

To use this package, download [Julia](https://julialang.org/downloads/). Within Julia, install the package via
```
using Pkg
Pkg.develop(url="https://github.com/biona001/RootCauseDiscovery.jl")
```

## Obtain QC'd gene expression data
```julia
using RootCauseDiscovery

# download data
download_data()

# process downloaded data
genecounts = process_data()
root_cause_df = process_root_cause_truth(genecounts)

# run QC
genecounts_normalized_ground_truth,
    genecounts_normalized_obs,
    root_cause_df_new = QC_gene_expression_data()
```