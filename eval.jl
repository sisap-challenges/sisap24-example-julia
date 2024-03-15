using JLD2, SimilaritySearch, DataFrames, CSV, Glob

function evaluate_results(gfile, resultfiles, k)
    gold_knns = jldopen(f->f["knns"][1:k, :], gfile)
    res = DataFrame(size=[], algo=[], buildtime=[], querytime=[], params=[], recall=[])
    for resfile in resultfiles
        @info resfile
        reg = jldopen(resfile) do f
            knns = f["knns"][1:k, :]
            recall = macrorecall(gold_knns, knns)
            push!(res, (f["size"], f["algo"], f["buildtime"], f["querytime"], f["params"], recall))
        end

    end

    res
end

if !isinteractive()
    goldsuffix = "public-queries-2024-laion2B-en-clip768v2-n=10k.h5"
    k = 30
    for path in glob("results-task?/*")
        task, dbsize = splitpath(path)
        lastpath = glob(joinpath(path, "*")) |> sort! |> last
        gfile = joinpath("data2024", "gold-standard-dbsize=$dbsize--$goldsuffix")
        files = glob(joinpath(lastpath, "*.h5"))
        D = evaluate_results(gfile, files, k)
        display(gfile => files)
        display(D)
        CSV.write("$task-$dbsize.csv", D)
    end
end
