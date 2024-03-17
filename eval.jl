using JLD2, SimilaritySearch, DataFrames, CSV, Glob, UnicodePlots

function evaluate_results(gfile, resultfiles, k)
    gold_knns = jldopen(f->f["knns"][1:k, :], gfile)
    res = DataFrame(size=[], algo=[], preprocessingtime=[], buildtime=[], querytime=[], params=[], recall=[])
    for resfile in resultfiles
        @info resfile
        reg = jldopen(resfile) do f
            knns = f["knns"][1:k, :]
            recall = macrorecall(gold_knns, knns)
            push!(res, (f["size"], f["algo"], f["preprocessingtime"], f["buildtime"], f["querytime"], f["params"], recall))
        end

    end
    sort!(res, :recall)
    res
end

if !isinteractive()
    goldsuffix = "public-queries-2024-laion2B-en-clip768v2-n=10k.h5"
    k = 30
    open("results-summary.txt", "w") do f
        for path in glob("results-task?/*")
            task, dbsize = splitpath(path)
            lastpath = glob(joinpath(path, "*")) |> sort! |> last
            gfile = joinpath("data2024", "gold-standard-dbsize=$dbsize--$goldsuffix")
            files = glob(joinpath(lastpath, "*.h5"))
            D = evaluate_results(gfile, files, k)
            println(f, "\n\n=== results for $dbsize $goldsuffix ===")
            println(f, gfile => files)
            show(f, "text/plain", gfile => files); println()
            show(f, "text/plain", D); println()
            display(gfile => files)
            display(D)
            p = lineplot(D.recall; ylim=(0, 1), title=String(D.algo[1]), ylabel="recall", xlabel="$(D.params[1]) ... $(D.params[end])")
            display(p)
            show(f, "text/plain", p); println()
            CSV.write("$task-$dbsize.csv", D)
        end
    end
end
