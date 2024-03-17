# This file is based in the julia's example of the 2023 edition

"""
    build_searchgraph(dist::SemiMetric, db::AbstractDatabase; minrecall=0.9)

Creates a `SearchGraph` index on the database `db`

- online documentation: <https://sadit.github.io/SimilaritySearch.jl/dev/>
- joss paper: _SimilaritySearch. jl: Autotuned nearest neighbor indexes for Julia
ES Tellez, G Ruiz - Journal of Open Source Software, 2022_ <https://joss.theoj.org/papers/10.21105/joss.04442.pdf>
- arxiv paper: 
```
Similarity search on neighbor's graphs with automatic Pareto optimal performance and minimum expected quality setups based on hyperparameter optimization
ES Tellez, G Ruiz - arXiv preprint arXiv:2201.07917, 2022
```
"""
function build_searchgraph(dist::SemiMetric, db::AbstractDatabase; minrecall=0.9)
    algo = "SearchGraph"
    logbase = 2
    ctx = SearchGraphContext(;
                             hyperparameters_callback = OptimizeParameters(MinRecall(minrecall)),
                             neighborhood = Neighborhood(; logbase),
                            )

    params = "r=$minrecall b=$logbase"
    G = SearchGraph(; db, dist)
    buildtime = @elapsed G = index!(G, ctx)
    optimtime = @elapsed optimize_index!(G, ctx)
    @show params, buildtime, optimtime
    meta = Dict(
        "buildtime" => buildtime,
        "matrix_size" => size(db.matrix),
        "optimtime" => optimtime,
        "algo" => algo,
        "params" => params
    )

    G, meta
end

"""
    run_search(idx::SearchGraph, queries::AbstractDatabase, k::Integer, meta, resfile_::AbstractString)

Solve `queries` with the give index (it will iterate on some parameter to find similar setups)

- `k` the number of nearest neighbors to retrieve
- `meta` metadata to be stored with results
- `resfile_` base name to create result files
"""

function save_results(knns::Matrix, dists::Matrix, meta, resfile::AbstractString)
    jldsave(resfile;
        knns, dists,
        algo=meta["algo"],
        buildtime=meta["buildtime"] + meta["optimtime"],
        querytime=meta["querytime"],
        preprocessingtime=meta["preprocessingtime"],
        params=meta["params"],
        size=meta["size"]
    )
end

function create_rp_model(dist, file::String; nbits::Int)
    dim = nbits ÷ 32 
    jldopen(file) do f
        X = f["emb"]
        m, n = size(X)
        fit(GaussianRandomProjection{Float32}, m => dim)
    end, SqL2Distance(), "GaussianRandomProjection-$(dim)"
end

function create_pca_model(dist, file::String; nbits::Int)
    dim = nbits ÷ 32 
    A = h5open(file) do f
        X = f["emb"]
        m, n = size(X)
        n2 = min(10^6, n ÷ 3)
        X[:, 1:n2]
    end
    @show size(A) typeof(A)
    fit(PCAProjection, A, dim), SqL2Distance(), "PCA-$(dim)"
end

function create_binperms_model(dist, file::String; nbits::Int, nrefs::Int=2048)
    A = h5open(file) do f
        X = f["emb"]
        m, n = size(X)
        n2 = min(10^6, n ÷ 3)
        X[:, 1:n2]
    end

    @show size(A) typeof(A)
    refs = let
        C = fft(dist, MatrixDatabase(A), nrefs) # select `nrefs` distant elements -- kcenters using farthest first traversal
        MatrixDatabase(A[:, C.centers])
    end

    fit(BinPerms, dist, refs, nbits), BinaryHammingDistance(), "BinPerms-$nbits"
end


function predict_h5(model::Union{PCAProjection,GaussianRandomProjection}, file::String; nbits, block::Int=10^5)
    dim = nbits ÷ 32
    h5open(file) do f
        X = f["emb"]
        m, n = size(X)
        B = Matrix{Float32}(undef, dim, n)
        for group in Iterators.partition(1:n, block)
            @info "encoding $group of $n -- $(Dates.now())"
            B[:, group] .= predict(model, MatrixDatabase(X[:, group])).matrix
        end

        StrideMatrixDatabase(B)
    end
end

function predict_h5(model::BinPerms, file::String; nbits, block::Int=10^5)
    h5open(file) do f
        X = f["emb"]
        m, n = size(X)
        B = Matrix{UInt64}(undef, nbits ÷ 64, n)
        for group in Iterators.partition(1:n, block)
            @info "encoding $group of $n -- $(Dates.now())"
            B[:, group] .= predict(model, MatrixDatabase(X[:, group])).matrix
        end

        StrideMatrixDatabase(B)
    end
end

