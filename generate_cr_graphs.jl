using JLD2
using ProgressBars
include("bottum_up_cr.jl")
include("ecosystem_model_functions.jl")
include("cr_inline.jl")

function generate_graph(n_graphs, size, spc_loc)

    function make_loc(loc)
        if ~isdir(loc)
            mkdir(loc)
        end
    end

    make_loc(spc_loc)

    for i in ProgressBar(1:n_graphs)
        #kmr_cr = ground_up_niche_model(size, 1, 0.32) #Returns a graph
        spc_cr = bottum_up_niche(size,1,0.32, 100) #Returns a structure that contains the graph (NicheStruct.G)
        jldsave("$spc_loc/graph_$(i).jld2"; g = spc_cr.G)
        #jldsave("$kmr_loc/graph_$(i).jld2"; g = kmr_cr)
        # if i + 61 > 88
        #     kmr_cr,_,_,_,_ = ground_up_niche_model(size) 
        #     jldsave("$kmr_loc/graph_$(i + 61).jld2"; g = kmr_cr)
        # end
    end

    nothing
end