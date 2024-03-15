using SimilaritySearch, SurrogatedDistanceModels, HDF5, JLD2, CSV, Glob, LinearAlgebra, Dates

include("common.jl")

# This file is based in the julia's example of the 2023 edition

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

function run_search_task3(idx::SearchGraph, queries::AbstractDatabase, k::Integer, meta, resfile_::String)
    resfile_ = replace(resfile_, ".h5" => "")
    step = 1.05f0
    delta = idx.search_algo.Δ / step^3
    params = meta["params"]

    # produces result files for different search hyperparameters
    while delta < 2f0
        idx.search_algo.Δ = delta
        dt = "delta=$(round(delta; digits=3))"
        resfile = "$resfile_-$dt.h5"
        @info "searching $resfile"
        meta["params"] = "$params $dt"
        querytime = @elapsed knns, dists = searchbatch(idx, queries, k)
        meta["querytime"] = querytime
        save_results(knns, dists, meta, resfile)
        delta *= step
    end
end

function task3(;
        dbsize,
        dfile="data2024/laion2B-en-clip768v2-n=$dbsize.h5",
        qfile="data2024/public-queries-2024-laion2B-en-clip768v2-n=10k.h5",
        k=30,
        outdir="results-task3/$dbsize/$(Dates.format(Dates.now(), "yyyymmdd-HHMMSS"))"
    )

    mkpath(outdir) 
    dist = NormalizedCosineDistance()  # 1 - dot(·, ·)
    nbits = 8 * 4 * 128  # memory eq to 128 fp32 
    #model, dist_proj, nick = create_binperms_model(dist, dfile; nbits)
    #model, dist_proj, nick = create_rp_model(dist, dfile; nbits)
    model, dist_proj, nick = create_pca_model(dist, dfile; nbits)
    db = predict_h5(model, dfile; nbits)
    queries = predict_h5(model, qfile; nbits)

    # loading or computing knns
    @info "indexing, this can take a while!"
    G, meta = build_searchgraph(dist_proj, db)
    meta["size"] = dbsize
    meta["params"] = "$(meta["params"]) $nick"
    resfile = joinpath(outdir, "searchgraph-$nick-k=$k")
    run_search_task3(G, queries, k, meta, resfile)
end

# functions for each database; these should have all required hyperparameters

if !isinteractive()
    if length(ARGS) != 1 || ARGS[1] ∉ ("300K", "10M", "100M")
        throw(ArgumentError("this script must be called with one of the following arguments: 300K, 10M or 100M"))
    end

    task3(dbsize=ARGS[1])
end
