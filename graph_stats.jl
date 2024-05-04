using JLD2
using Graphs
using LinearAlgebra
using PyPlot
using Infiltrator

import StatsBase: fit, Histogram, normalize
include("ecosystem_model_functions.jl")

# This struct holds all the statistics for a given graph
mutable struct GraphStats{T <: Float64}
    degree::Vector{T} #degree of all nodes
    tl::Vector{T} # trophic level of the nodes
    sl_frac::Vector{T} #fraction of nodes that are self loops
    connected::Vector{T}
end


#This function returns a dictionary where keys are the file not_same_size
#of graphs in a given directory (dir). The value of each key is a structure
#called GraphStats, containting all relevant statistics for the given graph.

function get_graph_statistics(dir)
    
    # This function builds the GraphStat structre
    function graph_stats(g)
        gs = GraphStats(
            Float64.(degree(g)),
            get_trophic(g),
            [num_self_loops(g)/ne(g)],
            [ne(g)/(nv(g))^2]
        )
        return gs
    end


    stat_dict = Dict()
    for file in readdir(dir)
        key = file[1: end - 5] #File name of the graph (minus the .jld2)
        f = jldopen(dir*"/"*file)
        stat_dict[key] = graph_stats(f["g"]) #Assigns a GraphStruct to a key
        close(f)
    end

    return stat_dict #Returns a dictionary of structures containing graph statistics
end


#Given a series of graphs, this function will compare their statistics. This function
#is mainly used to diseminate any differences between graphs that are generated in
#similar (but different) fashions.
function compare_graph_stats(dirs::Vector{String})
    fig,ax = PyPlot.subplots(2,2)

    all_dict = Dict() # In the same form from before, we will save a file name 
                      # as a key to compare all of our different graph constructors

    naughty_list = []
    for dir in dirs

        # We can actually use the same GraphStruct for all the statistics
        # of the differently generated graphs due to how I constructed it.

        all_gs = GraphStats(Float64[],
        Float64[],
        Float64[],
        Float64[])

        sd = get_graph_statistics(dir)
        for (i,gs) in sd
            if !(i in naughty_list)
                append!(all_gs.degree,gs.degree)
                if any(x -> x > 20, gs.tl)
                    @infiltrate
                    println(dir)
                    naughty_list = vcat(naughty_list, i)
                    continue
                end
                append!(all_gs.tl,gs.tl)
                append!(all_gs.sl_frac,gs.sl_frac)
                append!(all_gs.connected,gs.connected)
            end
        end

        all_dict[dir] = all_gs
    end
    
    max_mean = zeros(length(fieldnames(GraphStats)))

    for (key, gs) in all_dict
        for (i,n) in enumerate(fieldnames(typeof(gs)))
            max_mean[i] += maximum(getfield(gs,n))
        end
    end

    max_mean ./= length(all_dict)

    for (key, gs) in all_dict
        for (i,n) in enumerate(fieldnames(typeof(gs)))
            h = fit(Histogram,getfield(gs,n), [i for i in range(0,max_mean[i], 25)])
            e = (h.edges[1][1:end-1] + h.edges[1][2:end])/2
            h = normalize(h, mode=:pdf)
            ax[i].scatter(e,h.weights, label = string(key[1:3]), s = 6)
            ax[i].set_title(string(n))
        end
    end
    PyPlot.legend()
    PyPlot.tight_layout()
    gcf()
    return all_dict, naughty_list
end