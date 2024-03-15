using SimilaritySearch, SurrogatedDistanceModels, HDF5, JLD2, CSV, Glob, LinearAlgebra, Dates

include("common.jl")

# This file is based in the julia's example of the 2023 edition

function binperms_model(dist, file::String; nbits::Int=1024, nrefs::Integer=2028)
    jldopen(file) do f
        X = f["emb"]
        m, n = size(X)
        refs = MatrixDatabase(Matrix{Float32}(X[:, 1:nrefs])) # taking the first 2048 vectors as references
        fit(BinPerms, dist, refs, nbits)  # 1024 bits
    end
end

function predict_h5(model, file::String; nbits::Int=1024, block::Int=10^5)
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

function postprocessing(idx, queries, knns, dists, k, dfile, qfile)
    m, n = size(knns)
    dist = NormalizedCosineDistance()
    knns_ = zeros(Int32, k, n)
    dists_ = zeros(Float32, k, n)
    ctx = getcontext(idx)
     
    h5open(dfile) do fX
        h5open(qfile) do fQ
            X, Q = fX["emb"], fQ["emb"]
            L = Threads.SpinLock() 

            Threads.@threads :static for i in eachindex(queries)
                res = getknnresult(k, ctx)
                K = sort!(knns[:, i])
                q = lock(L) do
                    Q[:, i]
                end

                for j in 1:m
                    #objID = Int(knns[j, i])
                    objID = K[j]
                    objID == 0 && continue
                    v = lock(L) do
                        X[:, objID]
                    end
                    d = evaluate(dist, q, v)
                    push_item!(res, objID, d)
                end

                knns_[:, i] .= IdView(res)
                dists_[:, i] .= DistView(res)
            end
        end
    end
    
    knns_, dists_
end

function run_search_task2(idx::SearchGraph, queries::AbstractDatabase, k::Integer, meta, resfile_::String, dfile, qfile)
    resfile_ = replace(resfile_, ".h5" => "")
    params = meta["params"]

    # produces result files for different search hyperparametersfor k2 delta < 2f0
    for kspan in [k, 3k, 10k, 30k]
        dt = "kspan=$kspan"
        resfile = "$resfile_-$dt.h5"
        @info "searching $resfile"
        meta["params"] = "$params $dt"
        querytime = @elapsed knns, dists = searchbatch(idx, queries, kspan)
        if k != kspan
            querytime += @elapsed knns, dists = postprocessing(idx, queries, knns, dists, k, dfile, qfile)
        end
        meta["querytime"] = querytime
        save_results(knns, dists, meta, resfile)
    end
end

function task2(;
        dbsize,
        dfile="data2024/laion2B-en-clip768v2-n=$dbsize.h5",
        qfile="data2024/public-queries-2024-laion2B-en-clip768v2-n=10k.h5",
        k=30,
        k2=300,
        outdir="results-task2/$dbsize/$(Dates.format(Dates.now(), "yyyymmdd-HHMMSS"))"
    )

    mkpath(outdir) 
    dist = NormalizedCosineDistance()  # 1 - dot(·, ·)
    nbits = 8 * 4 * 72 # same than 72 FP32 
    model = binperms_model(dist, dfile; nbits)
    db = predict_h5(model, dfile; nbits)
    queries = predict_h5(model, qfile; nbits)
    distH = BinaryHammingDistance()

    # loading or computing knns
    @info "indexing, this can take a while!"
    G, meta = build_searchgraph(distH, db)
    meta["size"] = dbsize
    meta["params"] = "$(meta["params"]) binperms-$nbits"
    resfile = joinpath(outdir, "searchgraph-binperms-nbits=$nbits-k=$k")
    run_search_task2(G, queries, k, meta, resfile, dfile, qfile)
end

# functions for each database; these should have all required hyperparameters
if !isinteractive()
    if length(ARGS) != 1 || ARGS[1] ∉ ("300K", "10M", "100M")
        throw(ArgumentError("this script must be called with one of the following arguments: 300K, 10M or 100M"))
    end

    task2(dbsize=ARGS[1])
end
