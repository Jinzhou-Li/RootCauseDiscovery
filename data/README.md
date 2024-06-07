# Data

For exact codes used to download and preprocess these data, see codes in file `https://github.com/biona001/RootCauseDiscovery.jl/blob/main/julia/src/get_data.jl`.

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

