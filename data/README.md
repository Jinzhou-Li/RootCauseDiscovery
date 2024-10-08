# Data

## Obtaining QC'd gene expression data

To obtain real data used in our paper:
1. Download and install [Julia](https://julialang.org/downloads). 
2. Within Julia, install our package via
    ```
    using Pkg
    Pkg.add(PackageSpec(url="https://github.com/Jinzhou-Li/RootCauseDiscovery.git", subdir="julia"))
    ```
3. After the package is installed, copy and paste the following into Julia
    ```julia
    using RootCauseDiscovery

    # Download, read, filter, QC, and log (base 2) transform data.
    transform_int, transform_obs, ground_truth = QC_gene_expression_data(
        low_count = 10,
        threshold = 0.1, 
        max_cor = 0.999, 
    )

    # check dimension
    size(transform_int) # (19736, 60)
    size(transform_obs) # (19736, 365)
    size(ground_truth) # (70, 6)

    # convert to numeric matrices with rows as samples and columns as genes
    Xobs = transform_obs[:, 2:end] |> Matrix |> transpose
    Xint = transform_int[:, 2:end] |> Matrix |> transpose

    # get gene IDs
    gene_ids = transform_obs[:, 1]
    ```

To see how `QC_gene_expression_data` is implemented, see `https://github.com/Jinzhou-Li/RootCauseDiscovery/blob/main/julia/src/get_data.jl`.

## Real data analysis 

Due to Julia's computational efficiency, we used julia code located at `https://github.com/Jinzhou-Li/RootCauseDiscovery/tree/main/julia/src` for real data analysis. To reproduce the results, see the notebooks located at `https://github.com/Jinzhou-Li/RootCauseDiscovery/tree/main/julia/notebook`. 

## Ground truth data

This repo contains 2 files needed for our real data analysis:
+ **13073_2022_1019_MOESM1_ESM.csv**: we treat the gene listed under 'genetic diagnosis' in this table as the 'ground truth' root cause for each patient
+ **gene_name_mapping_v29.tsv**: file mapping gene IDs to gene names (e.g. `ENSG00000223972.5_2` maps to `DDX11L1`)

## Gene expression data

The raw gene-expression data can be downloaded and combined from the following two links:
+ https://zenodo.org/records/4646823
+ https://zenodo.org/records/4646827

To untar the `.tar.gz` files:
```
tar -xvzf fib_ns--hg19--gencode34.tar.gz -C .
tar -xvzf fib_ss--hg19--gencode34.tar.gz -C .
```

## References

All data from the links above and ground truth table originate from the following paper 

> Yepez, V. A., Gusic, M., Kopajtich, R., Mertes, C., et al. (2021). Clinical implementation of RNA sequencing for Mendelian disease diagnostics. medRxiv. doi:10.1101/2021.04.01.21254633.

