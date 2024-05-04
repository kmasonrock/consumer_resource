using JLD2
using Glob
using ProgressBars
include("cr_inline.jl")

function plot_extinction_statistics(gd)
    fig, ax = PyPlot.subplots()
    num_extinct = reduce(vcat,[length.(gd[g][2]) for g in keys(gd)])
    max_bin = maximum(num_extinct)

    bins = collect(range(0.5, max_bin + 0.5, max_bin*2 + 1))

    ax.hist(num_extinct, bins, density = false)
    ax.set_xlabel("Number of Resultant Extinctions")
    ax.set_ylabel("Number of Corresponding Experiments")
    PyPlot.gcf()
end

function get_df(dir)
    f = jldopen(dir)
    graph_dict = f["graph_dict"]
    close(f)
    return graph_dict
end

function extinction_statistics(dir)
    graph_dict = get_df(dir)
    plot_extinction_statistics(graph_dict)
end


function num_extinctions(graph_dir,system_dir)
    graphs = glob(graph_dir*"/*.jld2")
    all_systems = glob(system_dir*"/*.jld2")
    
    num_extinct = []
    tl = []

    for graph in ProgressBar(graphs)
        graph_num = String(split(graph,"/")[end])[1:end-5]
        gf = jldopen(graph)
        p = cr_params(gf["g"])
        trophics = get_trophic(gf["g"])
        close(gf)
        systems = all_systems[occursin.(graph_num*"_", all_systems)]
        for system in systems
            node = parse(Int64,split(system,"_")[end][1:end-5])
            sf = jldopen(system)
            ic = sf["init_cons"][:,1]
            prob = ODEProblem(cr_log_f, ic, [0 5000], p)
            sol = solve(prob, Vern9(); verbose = false)
            append!(num_extinct, sum(sol[end] .< -20))
            append!(tl, trophics[node])
            close(sf)
        end
    end
    return num_extinct, tl
end




