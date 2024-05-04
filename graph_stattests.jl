using JLD2
using Graphs
using MetaGraphs
using Glob
using PyPlot
using StatsBase
using LinearAlgebra

function get_trophic(g)

    copy_g = SimpleDiGraph(copy(g))

    self_loops = simplecycles_limited_length(copy_g,1)

    for i in self_loops
        rem_edge!(copy_g,i[1],i[1])
    end

    identity_m = Matrix(Diagonal(ones(length(vertices(copy_g)))))


    for i in [v for v in vertices(copy_g)]
    prey = indegree(copy_g,i)

        for j in inneighbors(copy_g,i)
            identity_m[i,j] = identity_m[i,j] - (1/prey)
        end
    end


    return [round(i; digits = 2) for i in inv(identity_m)*ones(length(vertices(copy_g)))]

end

files = glob("strong_allee_graphs_1overh/*.jld2")

indeg = []
introph = []
outdeg = []
outtroph = []
n_nodes = []

invasive_troph = []
invasive_in = []
invasive_out = []

for file in files
    f = jldopen(file, "r")
    g = f["g"]
    node = f["node"]

    append!(n_nodes, nv(g))
    troph = get_trophic(g)

    append!(invasive_troph, troph[node])
    f_in = [indegree(g,i) for i in 1:nv(g)]
    append!(indeg, f_in)
    append!(invasive_in, f_in[node])
    f_out = [outdegree(g,i) for i in 1:nv(g)]
    append!(outdeg, f_out)
    append!(invasive_out, f_out[node])

end


H_in,e_in,_ = hist(indeg, 1:8)
e_in = (e_in[1:end-1] + e_in[2:end])/2

H_out,e_out,_ = hist(outdeg, 1:8)
e_out = (e_out[1:end-1] + e_out[2:end])/2

fig, ax  = subplots(1,2)
ax[1].scatter(e_in, H_in)
ax[1].set_title("In-degree Distribution")
ax[2].scatter(e_out,H_out)
ax[2].set_title("Out-degree Distribution")

println("Avg In-degree: $(mean(indeg))")
println("Avg Out-degree: $(mean(outdeg))")
println("Avg Nodes: $(mean(n_nodes))")

println("Avg In-Degree of Invasive: $(mean(invasive_in))")
println("Avg Out-Degree of Invasive: $(mean(invasive_out))")
println("Avg Trophic Level of Invasive: $(mean(invasive_troph))")





gcf()




