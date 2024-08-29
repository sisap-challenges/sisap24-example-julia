using SimilaritySearch, SurrogatedDistanceModels, JLD2, CSV, Glob, LinearAlgebra, Dates

include("common.jl")

# This file is based in the julia's example of the 2023 edition

"""
    load_database(file)

Loads a dataset stored in `file`n
"""
function load_database(file)
    @info "loading clip768 (converting Float16 -> Float32)"
    X = jldopen(file) do f
        Matrix{Float32}(f["emb"])
    end

    #=for col in eachcol(X)
        normalize!(col)
    end=#

    StrideMatrixDatabase(X)
end

function run_search_task1(idx, queries::AbstractDatabase, k::Integer, meta, resfile_::AbstractString)
    resfile_ = replace(resfile_, ".h5" => "")
    resfile = "$resfile_.h5"
    @info "searching $resfile"
    querytime = @elapsed knns, dists = searchbatch(idx, queries, k)
    meta["querytime"] = querytime
    save_results(knns, dists, meta, resfile)
end

"""
    task1(; kwargs...)

Runs an entire beenchmark

- `dbsize`: string denoting the size of the dataset (e.g., "300K", "100M"), million scale should not be used in GitHub Actions.
- `k`: the number of neighbors to find 
"""
function task1(;
        dbsize,
        dfile="data2024/laion2B-en-clip768v2-n=$dbsize.h5",
        #qfile="data2024/public-queries-2024-laion2B-en-clip768v2-n=10k.h5",
        qfile="data2024/private-queries-2024-laion2B-en-clip768v2-n=10k-epsilon=0.2.h5",
        k=30,
        outdir="results-task1-bruteforce/$dbsize/$(Dates.format(Dates.now(), "yyyymmdd-HHMMSS"))"
    )

    mkpath(outdir)

    dist = NormalizedCosineDistance()  # 1 - dot(·, ·)
    @info "loading $qfile and $dfile"
    @time db = load_database(dfile)
    @time queries = load_database(qfile)
    
    # loading or computing knns
    @info "indexing, this can take a while!"
    G = ExhaustiveSearch(; dist, db)
    meta = Dict(
        "buildtime" => 0.0,
        "matrix_size" => size(db.matrix),
        "optimtime" => 0.0,
        "algo" => "Bruteforce",
        "params" => ""
    )
    meta["size"] = dbsize
    meta["modelingtime"] = 0.0
    meta["encdatabasetime"] = 0.0
    meta["encqueriestime"] = 0.0
    meta["buildtime"] = 0.0
    resfile = joinpath(outdir, "bruteforce-k=$k")
    run_search_task1(G, queries, k, meta, resfile)
end

if !isinteractive()
    if length(ARGS) == 0 || any(dbsize -> dbsize ∉ ("300K", "10M", "100M"), ARGS)
        throw(ArgumentError("this script must be called with a list of the following arguments: 300K, 10M or 100M"))
    end

    for dbsize in ARGS
        task1(; dbsize)
    end
end
