"""
    zscore(obs_data::AbstractMatrix{T}, intv_data::AbstractMatrix{T})

Runs the Z-score method. Input 1 is n*p, input 2 is m*p
"""
function zscore(obs_data::AbstractMatrix{T}, intv_data::AbstractMatrix{T}) where T
    npatients, ngenes = size(intv_data)
    size(obs_data, 2) == ngenes || error("Both input should have same number of genes (columns)")

    # observational data's mean and std
    μ = mean(obs_data, dims=1)
    σ = std(obs_data, dims=1)

    # compute squared Z scores for each patient in interventional data
    zscore_intv = zeros(npatients, ngenes)
    for patient in 1:npatients
        zs = Float64[]
        for i in 1:ngenes
            push!(zs, abs2((intv_data[patient, i] - μ[i]) / σ[i]) )
        end
        zscore_intv[patient, :] .= zs
    end
    
    return zscore_intv
end

"""
    compute_permutations(z::Vector{T}; threshold=2, nshuffles=1)

Given z-score vector, returns a list of permutations that will be used for 
the Cholesky method. 

# Inputs
+ `z`: Z-score vector

# Optional inputs
+ `threshold`: z-score threshold used for finding the abberent set (smaller
    means more permutations)
+ `nshuffles`: how many permutations to try for each "middle guy"
"""
function compute_permutations(
        z::AbstractVector{T}; 
        threshold=2,
        nshuffles=1
    ) where T
    # subset of genes indices that are abberent
    idx1 = findall(x -> x > threshold, z)
    idx1_copy = copy(idx1)

    # these are the normal genes
    idx2 = setdiff(1:length(z), idx1)

    # one way of generating length(idx1) permutations, fixing idx2
    perms = Vector{Int}[]
    for i in eachindex(idx1)
        for _ in 1:nshuffles
            shuffle!(idx2)
            shuffle!(idx1)
            target = idx1_copy[i]
            index = findfirst(x -> x == target, idx1)
            idx1[1], idx1[index] = target, idx1[1]
            push!(perms, vcat(idx2, idx1))
        end
    end
    return perms
end

# note: Xobs is n by p (i.e. p genes, must transpose Xobs and Xint prior to input)
"""
    root_cause_discovery(Xobs, Xint, perm)

Given one permutation, run main algorithm for root cause discovery.
"""
function root_cause_discovery(
        Xobs::AbstractMatrix{T}, 
        Xint::AbstractVector{T},
        perm::AbstractVector{Int}; 
        verbose=true,
    ) where T
    n, p = size(Xobs)
    p == length(Xint) || error("dimension mismatch!")
    all(sort(perm) .== 1:p) || error("perm is not a permutation vector")
    
    # permute Xobs and Xint
    Xobs_perm = Xobs[:, perm]
    Xint_perm = Xint[perm]

    # estimate covariance and mean
    μ̂ = mean(Xobs_perm, dims=1) |> vec
    if n > p
        Σ̂ = cov(Xobs_perm)
#         Σ̂ = cov(LinearShrinkage(CommonCovariance(), :ss), Xobs_perm)
#         Σ̂ = cov(LinearShrinkage(DiagonalUnequalVariance(), :ss), Xobs_perm)
    else
        Σ̂ = cov(LinearShrinkage(DiagonalUnequalVariance(), :lw), Xobs_perm)
    end

    # ad-hoc way to ensure PSD
    min_eval = minimum(eigvals(Σ̂))
    if min_eval < 1e-6
        for i in 1:size(Σ̂, 1)
            Σ̂[i, i] += abs(min_eval) + 1e-6
        end
    end
    
    # compute cholesky
    L = cholesky(Σ̂)

    # solve for X̃ in LX̃ = Xint - μ̂ 
    X̃ = zeros(p)
    ldiv!(X̃, L.L, Xint - μ̂)
    
    # undo the permutations
    invpermute!(X̃, perm)

    return abs.(X̃)
end

# used in simulation
# this is a main RootCauseDiscovery algorith, without dimensional reduction
function root_cause_discovery_one_subject_all_perm(
        transform_obs::DataFrame,  # col 1 is gene name, every other cols are different samples
        transform_int::DataFrame,  # this MUST be a 2 column dataframe, col1 = gene name, col2 = gene counts for a patient
        threshold::Float64;
        verbose::Bool=true,
        nshuffles::Int=1
        )
    # check for errors
    ngenes = size(transform_obs, 1)
    gene_names = transform_obs[!, 1]
    gene_names == transform_int[!, 1] || error("col1 of transform_obs and transform_int should be the same")
    size(transform_int, 2) == 2 || error("transform_int MUST be a 2 column dataframe")
    
    # compute Z scores (1st column of both dataframe is gene name)
    Xobs = transform_obs[:, 2:end] |> Matrix |> transpose
    Xint = transform_int[:, 2]'
    z = zscore(Xobs, Xint) |> vec
    
    # compute permutations to try
    permutations = compute_permutations(z, threshold=threshold, nshuffles=nshuffles)
    println("Trying $(length(permutations)) permutations")

    # try all permutations
    X̃all = zeros(ngenes, length(permutations))
    for (i, perm) in enumerate(permutations)
        X̃ = root_cause_discovery(Xobs_perm, Xint_perm, perm, verbose=verbose)
        X̃all[:, i] .= X̃
    end

    return X̃all
end

function find_largest(X̃all::Vector{Vector{Float64}})
    p = length(X̃all)
    largest = Float64[]
    largest_idx = Int[]
    for X̃ in X̃all
        push!(largest, sort(X̃)[end])
        push!(largest_idx, findmax(X̃)[2])
    end
    return largest, largest_idx
end

function find_second_largest(X̃all::Vector{Vector{Float64}})
    p = length(X̃all)
    second_largest = Float64[]
    for X̃ in X̃all
        push!(second_largest, sort(X̃)[end-1])
    end
    return second_largest
end

"""
    reduce_genes(patient_id, y_idx, Xobs, Xint, ground_truth, method)

Assuming root cause gene for patient `patient_id` is unknown, we run Lasso on the each gene, 
pretending it is the root-cause-gene (to select a subset of genes from Xobs)
"""
function reduce_genes(
        patient_id, 
        y_idx::Int, # this col will be treated as response in Xobs when we run lasso
        Xobs::AbstractMatrix{Float64}, 
        Xint::AbstractMatrix{Float64},
        ground_truth::DataFrame, # this is only used to access sample ID in Xint
        method::String, # either "cv" or "largest_support"
    )
    n, p = size(Xobs)

    # response and design matrix for Lasso
    y = Xobs[:, y_idx]
    X = Xobs[:, setdiff(1:p, y_idx)]

    # fit lasso
    beta_final = nothing
    if method == "cv"
        cv = glmnetcv(X, y)
        beta_final = GLMNet.coef(cv)
    elseif method == "largest_support"
        path = glmnet(X, y)
        beta_final = path.betas[:, end]
    else
        error("method should be `cv` or `largest_support`")
    end
    nz = count(!iszero, beta_final)
    println("Lasso found $nz non-zero entries")

    # for non-zero idx, find the original indices in Xobs, and don't forget to include y_idx
    # back to the set of selected variables
    selected_idx = findall(!iszero, beta_final)
    selected_idx[findall(x -> x ≥ y_idx, selected_idx)] .+= 1
    append!(selected_idx, y_idx)

    # return the subset of variables of Xobs that were selected
    Xobs_new = Xobs[:, selected_idx]
    i = findfirst(x -> x == patient_id, ground_truth[!, "Patient ID"])
    Xint_sample_new = Xint[i, selected_idx]

    return Xobs_new, Xint_sample_new, selected_idx
end

"""
Todo: combine with `root_cause_discovery_one_subject_all_perm`
"""
function root_cause_discovery_reduced_dimensional(
        Xobs_new::AbstractMatrix{Float64}, 
        Xint_sample_new::AbstractVector{Float64};
        nshuffles::Int = 1,
    )
    # first compute z score (needed to compute permutations)
    z = zscore(Xobs_new, Xint_sample_new') |> vec

    # root cause discovery by trying lots of permutations
    largest, largest_idx = Float64[], Int[]
    second_largest = Float64[]
    @showprogress for threshold in 0.1:0.2:5
        permutations = compute_permutations(z; threshold=threshold, nshuffles=nshuffles)
        X̃all = Vector{Float64}[]
        for perm in permutations
            push!(X̃all, root_cause_discovery(Xobs_new, Xint_sample_new, perm; verbose=true))
        end
        largest_cur, largest_idx_cur = find_largest(X̃all)
        append!(largest, largest_cur)
        append!(largest_idx, largest_idx_cur)
        append!(second_largest, find_second_largest(X̃all))
    end

    # return table (sorted by largest - 2nd largest)
    diff = largest-second_largest
    diff_normalized = diff ./ second_largest
    perm = sortperm(diff_normalized)
    result = [largest second_largest diff diff_normalized largest_idx z[largest_idx]][perm, :]
    result = [result [size(Xobs_new, 2) for _ in 1:size(result, 1)]] # add correct to result purely for easier visualization
    return result
end

"""
    root_cause_discovery_high_dimensional

High level wrapper for Root cause discovery, for p>n data. 
First guess many root causes, then runs lasso, then run root cause discovery algorithm.

Still need a way to check if guesses are "root causes"
"""
function root_cause_discovery_high_dimensional(
        patient_id::String,
        Xobs::AbstractMatrix{Float64}, 
        Xint::AbstractMatrix{Float64},
        ground_truth::DataFrame, # this is only used to access sample ID in Xint
        method::String; # either "cv" or "largest_support"
        y_idx_z_threshold=1.5,
        nshuffles::Int = 1,
    )
    # compute some guesses for root cause index
    i = findfirst(x -> x == patient_id, ground_truth[!, "Patient ID"])
    Xint_sample = Xint[i, :]
    z = zscore(Xobs, Xint_sample') |> vec
    y_indices = compute_y_idx(z, z_threshold=y_idx_z_threshold)
    println("Trying $(length(y_indices)) y_idx")
    
    results = Matrix{Float64}[]
    for y_idx in y_indices
        # run lasso, select gene subset to run root cause discovery
        Xobs_new, Xint_sample_new, _ = reduce_genes(
            patient_id, y_idx, Xobs, Xint, ground_truth, method
        )

        # compute current guess of root cause index
        result = root_cause_discovery_reduced_dimensional(Xobs_new, 
            Xint_sample_new, nshuffles=nshuffles)

        push!(results, result)
    end
    
    # 4th col is (largest - 2nd largest) / (2nd largest)
    # 5th col is index of the largest element
    # 7th col is the index that is used as response in lasso regression
    # when 5th and 7th column match, we have "found" the root cause index
    # otherwise we didn't find the root cause index
    return results, y_indices
end

"""
    compute_y_idx(z::AbstractVector{Float64})

Given some Z scores, compute a set of variables that will be treated as "root causes", and enter the 
root-cause-discovery algorirthm
"""
function compute_y_idx(z::AbstractVector{Float64}; z_threshold=1.5)
    return findall(x -> x > z_threshold, z)
end
