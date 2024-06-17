"""
    zscore(obs_data::AbstractMatrix{T}, intv_data::AbstractMatrix{T})
    zscore(obs_data::AbstractMatrix{T}, intv_data::AbstractVector{T})

Runs the Z-score method. Input 1 is n*p, input 2 is m*p (or p*1)
"""
function zscore(
        obs_data::AbstractMatrix{T}, 
        intv_data::AbstractMatrix{T}
    ) where T
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

function zscore(
        obs_data::AbstractMatrix{T}, 
        intv_data::AbstractVector{T}
    ) where T
    ngenes = length(intv_data)
    size(obs_data, 2) == ngenes || error("Number of genes mismatch")

    # observational data's mean and std
    μ = mean(obs_data, dims=1)
    σ = std(obs_data, dims=1)

    # compute squared Z scores for each patient in interventional data
    zs = Float64[]
    for i in 1:ngenes
        push!(zs, abs2((intv_data[i] - μ[i]) / σ[i]) )
    end
    return zs
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
+ `nshuffles`: how many permutations to try for a fixed "middle variable"
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
    if length(idx1) == 0 # all z score are below threshold
        for _ in 1:nshuffles
            shuffle!(idx2)
            push!(perms, copy(idx2))
        end
    else
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
    end
    return perms
end

"""
    root_cause_discovery(Xobs, Xint, perm)

Given one permutation, run our main algorithm for root cause discovery.

# Note
`Xobs` is n by p (i.e. p genes, must transpose Xobs and Xint prior to input)
"""
function root_cause_discovery(
        Xobs::AbstractMatrix{T}, 
        Xint::AbstractVector{T},
        perm::AbstractVector{Int}; 
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

    # solve for X̃ in LX̃ = Xint_perm - μ̂ 
    X̃ = zeros(p)
    ldiv!(X̃, L.L, Xint_perm - μ̂)
    
    # undo the permutations
    invpermute!(X̃, perm)

    return abs.(X̃)
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
    reduce_genes(y_idx, Xobs, Xint, method, verbose)

Treat each gene as the response and run CVLasso to select a subset of genes 
(estimated Markov blanket). Return new datasets with only these genes. The
gene used as response is returned as the last element of `selected_idx`

Later we will use the returned data to run our algorithm. 
"""
function reduce_genes(
        y_idx::Int, # this col will be treated as response in Xobs when we run lasso
        Xobs::AbstractMatrix{Float64}, 
        Xint_sample::AbstractVector{Float64},
        method::String = "cv", # either "cv" or "nhalf"
        verbose::Bool = true,
        nhalf_threshold = 5 # when CVLasso selects too few, run method=nhalf
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
        if count(!iszero, beta_final) < nhalf_threshold
            return reduce_genes(y_idx, Xobs, Xint_sample, "nhalf", verbose)
        end
    elseif method == "nhalf" # ad-hoc method to choose ~n/2 number of non-zero betas
        # compute default lambda path
        r = X'*y
        lambdamax = maximum(abs, r) / sqrt(n)
        lambdamin = 0.0000001lambdamax
        lambda = exp.(range(log(lambdamin), log(lambdamax), length=100)) |> reverse!
        # run lasso
        path = glmnet(X, y, lambda=lambda)
        beta_final = path.betas[:, 1]
        best_ratio = abs(0.5 - count(!iszero, beta_final) / n)
        for beta in eachcol(path.betas)
            new_ratio = abs(0.5 - count(!iszero, beta) / n)
            if new_ratio < best_ratio
                best_ratio = new_ratio
                beta_final = beta
            end
        end
        beta_final = Vector(beta_final)
    else
        error("method should be `cv` or `nhalf`")
    end
    nz = count(!iszero, beta_final)
    verbose && println("Lasso found $nz non-zero entries")

    # for non-zero idx, find the original indices in Xobs
    # and include y_idx back to the set of selected variables
    selected_idx = findall(!iszero, beta_final)
    selected_idx[findall(x -> x ≥ y_idx, selected_idx)] .+= 1
    append!(selected_idx, y_idx)

    # return data containing the subset of selected variables
    Xobs_new = Xobs[:, selected_idx]
    Xint_sample_new = Xint_sample[selected_idx]

    return Xobs_new, Xint_sample_new, selected_idx
end

"""
    get_abberant_thresholds(z_vec::AbstractVector{T}, [threshold_min], 
        [threshold_max], [threshold_seq])

Computes a list of threshold values (defaults to 0.1:0.2:5) that are 
non-redundant based on the input z scores
"""
function get_abberant_thresholds(
        z_vec::AbstractVector{T};
        threshold_min=0.1, 
        threshold_max=5.0,
        threshold_seq=0.2
    ) where T
    threshold_raw = collect(threshold_min:threshold_seq:threshold_max)
    count_temp = -1
    threshold_new = T[]
    for threshold in threshold_raw
        if count(x -> x >= threshold, z_vec) > 0
            if count(x -> x <= threshold, z_vec) != count_temp
                count_temp = count(x -> x <= threshold, z_vec)
                push!(threshold_new, threshold)
            end
        end
    end
    if length(threshold_new) == 0
        threshold_new = [threshold_min]
    end
    return threshold_new
end

"""
    root_cause_discovery_reduced_dimensional(Xobs_new, Xint_sample_new, [nshuffles], [thresholds])

Run the root cause discovery algorithm on data with reduced dimension.
Update the Cholesky score for the variable treated as response
"""
function root_cause_discovery_reduced_dimensional(
        Xobs_new::AbstractMatrix{Float64}, 
        Xint_sample_new::AbstractVector{Float64};
        nshuffles::Int = 1,
        thresholds::Union{Nothing, Vector{Float64}} = nothing
    )
    # first compute z score (needed to compute permutations)
    z = zscore(Xobs_new, Xint_sample_new)

    # if no thresholds are provided, compute default
    if isnothing(thresholds)
        thresholds = get_abberant_thresholds(z)
    end

    # root cause discovery by trying lots of permutations
    largest, largest_idx = Float64[], Int[]
    second_largest = Float64[]
    for threshold in thresholds
        permutations = compute_permutations(z; threshold=threshold, nshuffles=nshuffles)
        X̃all = Vector{Vector{Float64}}(undef, length(permutations))
        for i in eachindex(permutations)
            perm = permutations[i]
            X̃all[i] = root_cause_discovery(
                Xobs_new, Xint_sample_new, perm
            )
        end
        largest_cur, largest_idx_cur = find_largest(X̃all)
        append!(largest, largest_cur)
        append!(largest_idx, largest_idx_cur)
        append!(second_largest, find_second_largest(X̃all))
        if any(x -> x < 0, largest - second_largest)
            println("largest = $largest")
            println("second_largest = $second_largest")
            println("X̃all = $(X̃all)")
            error("largest - 2nd largest is negative! Shouldn't happen!")
        end
    end

    diff = largest-second_largest
    diff_normalized = diff ./ second_largest
    perm = sortperm(diff_normalized)

    # return root cause (cholesky) score for the current variable that is treated as response
    root_cause_score_y = 0.0
    for per in Iterators.reverse(perm)
        matched = size(Xobs_new, 2) == largest_idx[per]
        if matched
            root_cause_score_y = diff_normalized[per]
            break
        end
    end
    return root_cause_score_y
end

"""
    root_cause_discovery_high_dimensional

The main root cause discovery for high-dimensional data.
This includes treating each variable as response, 
applying Laaso to reduce dimension, 
and running our root cause discovery algorithm.
"""
function root_cause_discovery_high_dimensional(
        Xobs::AbstractMatrix{Float64}, 
        Xint_sample::AbstractVector{Float64},
        method::String = "cv"; # either "cv" or "nhalf"
        y_idx_z_threshold=1.5, # To save computational time, we only treat variables that are abberant as response (have large enough z-score)
        nshuffles::Int = 1,
        verbose = true,
        thresholds::Union{Nothing, Vector{Float64}} = nothing,
        y_indices = compute_y_idx(Xobs, Xint_sample, y_idx_z_threshold)
    )
    p = size(Xobs, 2)
    verbose && println("Trying $(length(y_indices)) y_idx")

    # assign root cause score one by one
    root_cause_scores = zeros(p)
    Threads.@threads for y_idx in y_indices
        # run lasso, select gene subset to run root cause discovery
        # note: it is possible that reduce_genes (i.e. CVLasso) select no features
        Xobs_new, Xint_sample_new, _ = reduce_genes(
            y_idx, Xobs, Xint_sample, method, verbose
        )
        # run our root cause discovery algorithm on reduced data
        root_cause_scores[y_idx] = root_cause_discovery_reduced_dimensional(
            Xobs_new, Xint_sample_new, nshuffles=nshuffles, thresholds=thresholds
        )
    end

    # assign root cause score for variables whose current score are 0
    z = zscore(Xobs, Xint_sample)
    idx2 = findall(iszero, root_cause_scores)
    idx1 = findall(!iszero, root_cause_scores)
    if length(idx2) != 0
        if length(idx1) != 0
            max_RC_score_idx = minimum(root_cause_scores[idx1]) - 0.00001
            root_cause_scores[idx2] .= z[idx2] ./ (maximum(z[idx2]) ./ max_RC_score_idx)
        else
            # use z scores when all scores are 0
            root_cause_scores = z
        end
    end

    return root_cause_scores
end

"""
    compute_y_idx(z::AbstractVector{Float64})

Compute a set of abberant variables whose z-score are larger than some threshold.
These variables will be treated as response in function 'root_cause_discovery_high_dimensional'
"""
function compute_y_idx(Xobs, Xint_sample, z_threshold=1.5)
    z = zscore(Xobs, Xint_sample)
    return findall(x -> x > z_threshold, z)
end
