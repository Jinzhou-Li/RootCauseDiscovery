module RootCauseDiscovery

using CSV
using DataFrames
using Downloads
using ProgressMeter
using Statistics
using GLMNet
using CovarianceEstimation
using LinearAlgebra
using Random

export download_data, 
    process_data,
    process_root_cause_truth,
    QC_gene_expression_data,
    root_cause_discovery_high_dimensional,
    root_cause_discovery_reduced_dimensional,
    reduce_genes,
    zscore

include("get_data.jl")
include("utilities.jl")

datadir(parts...) = joinpath(@__DIR__, "../..", "data", parts...)

function check_data_exist()
    files = readdir(RootCauseDiscovery.datadir())
    if "fib_ss--hg19--gencode34" ∉ files
        @info "Downloading data..."
        # download data
        download_data()
        # process downloaded data
        process_root_cause_truth(process_data())
    end
end
check_data_exist()

end # module RootCauseDiscovery
