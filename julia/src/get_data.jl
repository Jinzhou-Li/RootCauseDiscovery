function download_data(outdir::String=datadir())
    isdir(outdir) || error("$outdit does not exist")

    # raw gene count data
    ns_file = joinpath(outdir, "fib_ns--hg19--gencode34.tar.gz")
    Downloads.download(
        "https://zenodo.org/records/4646823/files/fib_ns--hg19--gencode34.tar.gz",
        ns_file
    )
    ss_file = joinpath(outdir, "fib_ss--hg19--gencode34.tar.gz")
    Downloads.download(
        "https://zenodo.org/records/4646827/files/fib_ss--hg19--gencode34.tar.gz",
        ss_file
    )

    # ground truth root cause table can be downloaded as below
    # but it is not needed since the S2, S3, and S4 tables are already 
    # manually convert to csv and uploaded onto github.
    # xlsx_file = joinpath(outdir, "13073_2022_1019_MOESM1_ESM.xlsx")
    # Downloads.download(
    #     "https://static-content.springer.com/esm/art%3A10.1186%2Fs13073-022-01019-9/MediaObjects/13073_2022_1019_MOESM1_ESM.xlsx",
    #     xlsx_file
    # )

    # untar
    run(`tar -xvzf $ns_file -C $outdir`)
    run(`tar -xvzf $ss_file -C $outdir`)
end

"""
    process_data()

# Output
+ Each column is a sample
+ Each row is a gene
"""
function process_data(
        data_dir::String=datadir(), 
        concatenate="all", # "all", "ss", or "ns"
    )
    # data should exist in `data_dir`
    ss_dir = joinpath(data_dir, "fib_ss--hg19--gencode34")
    ns_dir = joinpath(data_dir, "fib_ns--hg19--gencode34")
    isdir(ss_dir) || error("Did not find `fib_ss--hg19--gencode34` directory in $data_dir, please rerun download_data()")
    isdir(ss_dir) || error("Did not find `fib_ns--hg19--gencode34` directory in $data_dir, please rerun download_data()")

    # read data
    ss_genecounts = CSV.read(joinpath(ss_dir, "geneCounts.tsv.gz"), DataFrame)
    ns_genecounts = CSV.read(joinpath(ns_dir, "geneCounts.tsv.gz"), DataFrame)

    # remove ".15_5" after geneID
    ss_genecounts[!, "geneID"] .= [split(gene, '.')[1] for gene in ss_genecounts[!, "geneID"]]
    ns_genecounts[!, "geneID"] .= [split(gene, '.')[1] for gene in ns_genecounts[!, "geneID"]]

    if concatenate == "all"
        genecounts = hcat(ss_genecounts, ns_genecounts[:, 2:end])
        # @show size(genecounts) # 62492×424
        return genecounts
    elseif concatenate == "ns"
        return ns_genecounts # 62492×155
    elseif concatenate == "ss"
        return ss_genecounts # 62492×270
    else
        error("concatenate must be `all`, `ns`, or `ss`")
    end
end

"""
    process_root_cause_truth(genecounts::DataFrame)

Assumes "gene_name_mapping_v29.tsv" and "13073_2022_1019_MOESM1_ESM_S{2..4}.csv" 
exist in `data_dir`
"""
function process_root_cause_truth(
        genecounts::DataFrame,
        data_dir::String=RootCauseDiscovery.datadir(),
    )
    S2_file = "13073_2022_1019_MOESM1_ESM_S2.csv"
    S3_file = "13073_2022_1019_MOESM1_ESM_S3.csv"
    S4_file = "13073_2022_1019_MOESM1_ESM_S4.csv"
    df2 = process_one_root_cause_ground_truth_table(S2_file, genecounts, data_dir)
    df3 = process_one_root_cause_ground_truth_table(S3_file, genecounts, data_dir)
    df4 = process_one_root_cause_ground_truth_table(S4_file, genecounts, data_dir)

    # for table S3, rename "Candidate gene" to "Genetic diagnosis"
    rename!(df3, "Candidate gene" => "Genetic diagnosis")

    # concatenate all 3 
    df = vcat(df2, df3, df4)

    # "RNA detect" actually indicate whether the gene is AE
    # so we change that columnname to indicate this
    is_AE = [!ismissing(x) && x == "x" for x in df[!, "RNA defect"]]
    df = df[!, ["Patient ID", "Genetic diagnosis", "gene_id", 
        "patient column index in genecounts", "root cause row index in genecounts"]]
    df[!, "is AE"] = is_AE

    # strip trailing white spaces from patient IDs
    df[!, "Patient ID"] .= strip.(df[!, "Patient ID"])

    return df
end

function process_one_root_cause_ground_truth_table(
        table::String,
        genecounts::DataFrame,
        data_dir::String=RootCauseDiscovery.datadir(),
    )
    # determine the root-cause column name in `table`
    if table == "13073_2022_1019_MOESM1_ESM_S2.csv"
        RC_col_name = "Genetic diagnosis"
    elseif table == "13073_2022_1019_MOESM1_ESM_S3.csv"
        RC_col_name = "Candidate gene"
    elseif table == "13073_2022_1019_MOESM1_ESM_S4.csv"
        RC_col_name = "Genetic diagnosis"
    else
        error("unknow table $table!")
    end

    # read ground truth
    df = CSV.read(joinpath(data_dir, table), DataFrame, 
        header=2)

    # pre-process data
    has_entry = findall(!ismissing, df[!, "Patient ID"])
    df = df[has_entry, :]

    # for each root cause gene, find the corresponding name in the gene_name_mapping_v29.tsv file
    map_file = CSV.read(joinpath(data_dir, "gene_name_mapping_v29.tsv"), DataFrame, delim=',')
    map_dict = Dict{String, String}()
    for i in 1:size(map_file, 1)
        genename = map_file[i, "gene_name"]
        geneID = split(map_file[i, "gene_id"], '.')[1]
        map_dict[genename] = geneID
    end

    # first version of processed data frame (underlying truth)
    gene_ID = String[]
    for i in 1:size(df, 1)
        push!(gene_ID, map_dict[df[i, RC_col_name]])
    end
    df[!, "gene_id"] = gene_ID

    # keep only necessary columns. Note: due to the csv format conversion, 
    # "RNA detect" actually indicate whether the gene is AE
    root_cause_df = df[!, ["Patient ID", RC_col_name, "gene_id", "RNA defect"]] 

    # add patient index and root cause index to processed df
    col_idx = Int[]
    row_idx = Int[]
    keep_rows = Int[]
    genecounts_col_names = names(genecounts)
    genecounts_row_names = genecounts[!, "geneID"]
    for (i, row) in enumerate(eachrow(root_cause_df))
        patient_id = row["Patient ID"]
        patient_rootcause_gene = row["gene_id"]
        c = findfirst(x -> x == strip(patient_id), genecounts_col_names)
        r = findfirst(x -> x == patient_rootcause_gene, genecounts_row_names)
        if !isnothing(c) && !isnothing(r)
            push!(col_idx, c)
            push!(row_idx, r)
            push!(keep_rows, i)
        end
    end
    root_cause_df = root_cause_df[keep_rows, :]
    root_cause_df[!, "patient column index in genecounts"] = col_idx
    root_cause_df[!, "root cause row index in genecounts"] = row_idx
    return root_cause_df
end

function estimate_size_factor(K::AbstractMatrix)
    n, m = size(K)
    s = zeros(m)
    storage = Float64[]
    @showprogress for j in 1:m
        empty!(storage)
        for i in 1:n
            push!(storage, K[i, j] / geomeanNZ(K[i, :]))
        end
        s[j] = median(storage)
    end
    return s
end
function geomeanNZ(v::AbstractVector)
    m = length(v)
    if all(iszero, v)
        return 0
    else
        return exp( sum(log.(v[findall(x -> x > 0, v)])) / m )
    end
end

"""
    QC_gene_expression_data(;low_count=10, threshold=0.2)

keeps rows (i.e. genes) if there is at least `threshold` 
number of columns with values greater than `low_count`, and then 
normalize gene count file. 

# Normalizing gene counts file
First compute "size factor" to normalize columns of gene counts 
(somehow taking care of read depth issue), see eq5 of 
https://genomebiology.biomedcentral.com/articles/10.1186/gb-2010-11-10-r106

The actual normalization follows eq6 of 
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6288422/#:~:text=OUTRIDER%20is%20open%20source%20and,%2Drate%2Dadjusted%20p%20values
`x_ij = log((k_ij + 1) / s_j)` where `s` are the size-factors.
"""
function QC_gene_expression_data(;
    low_count::Int = 10, 
    threshold::Float64 = 0.1,
    max_cor::Float64 = 0.999,
    concatenate::String = "all" # "all", "ss", or "ns"
    )
    # low_count = 10
    # threshold = 0.1
    # max_cor = 0.999
    # concatenate = "all"
    # import RootCauseDiscovery.estimate_size_factor
    # import RootCauseDiscovery.has_linearly_dependent_col

    # raw data
    genecounts = process_data(RootCauseDiscovery.datadir(), concatenate)
    root_cause_ground_truth = process_root_cause_truth(genecounts)

    # first column is gene ID
    count_data = @view(genecounts[:, 2:end])

    # process gene count file
    nsamples = size(count_data, 2)
    keep_rows = Int[]
    for (i, row) in enumerate(eachrow(count_data))
        proportion = count(x -> x > low_count, row) / nsamples
        if proportion > threshold
            push!(keep_rows, i)
        end
    end
    genecounts_filtered = genecounts[keep_rows, :]

    # remove linearly dependent columns
    Σcor = cor(Matrix(genecounts_filtered[:, 2:end])')
    while has_linearly_dependent_col(Σcor, max_cor)
        idx = findall(x -> abs(x) > max_cor, UpperTriangular(Σcor))
        idx = filter!(x -> x[1] != x[2], idx)
        to_remove = [x[2] for x in idx]
        to_keep = setdiff(1:size(genecounts_filtered, 1), to_remove)
        genecounts_filtered = genecounts_filtered[to_keep, :]
        Σcor = cor(Matrix(genecounts_filtered[:, 2:end])')
    end

    # Make sure the truth root cause gene is preserved
    pass_qc_genes = unique(genecounts_filtered[!, 1]) |> Vector{String}
    root_cause_genes = unique(root_cause_ground_truth[!, "gene_id"]) |> Vector{String}
    length(root_cause_genes) == length(intersect(root_cause_genes, pass_qc_genes)) ||
        error("At least one true root cause gene is removed after QC, aborting")

    # estimate size factor 
    s = estimate_size_factor(Matrix(genecounts_filtered[:, 2:end]))

    # normalize (j index over samples, i index over genes)
    genecounts_normalized = DataFrame("geneID" => genecounts_filtered[!, "geneID"])
    col_names = names(genecounts_filtered)
    for j in 2:size(genecounts_filtered, 2) # loop over samples
        col = Float64[]
        for i in 1:size(genecounts_filtered, 1) # loop over genes
            push!(col, log((genecounts_filtered[i, j] + 1) / s[j - 1]) )
        end
        genecounts_normalized[!, col_names[j]] = col
    end

    # update the ground truth root cause table after deleting some genes 
    # (which causes the position of root cause genes to change)
    root_cause_ground_truth_new = deepcopy(root_cause_ground_truth)
    full_geneID = genecounts_filtered[!, "geneID"]
    for i in 1:size(root_cause_ground_truth, 1)
        gene = root_cause_ground_truth[i, "gene_id"]
        new_idx = findfirst(x -> x == gene, full_geneID)
        root_cause_ground_truth_new[i, "root cause row index in genecounts"] = new_idx
    end

    # finally, we create 2 subsets:
    #   1. One includes only AE samples for which root cause gene is known 
    #   2. The other includes all other samples (observational data)
    known_AE_patient_id = root_cause_ground_truth_new[
        root_cause_ground_truth_new[!, "is AE"], "Patient ID"
    ]
    subset1_idx = indexin(known_AE_patient_id, names(genecounts_normalized))
    subset2_idx = setdiff(1:size(genecounts_normalized, 2), subset1_idx)
    genecounts_normalized_int = genecounts_normalized[:, vcat(1, subset1_idx)]
    genecounts_normalized_obs = genecounts_normalized[:, subset2_idx]

    return genecounts_normalized_int, 
        genecounts_normalized_obs, 
        root_cause_ground_truth_new
end

# input is a correlation matrix
function has_linearly_dependent_col(Σcor::AbstractMatrix, max_cor::Float64)
    idx = findall(x -> x > max_cor, UpperTriangular(Σcor))
    idx = filter!(x -> x[1] != x[2], idx)
    return length(idx) > 0
end
