using NetworkLayout
using Colors
import StatsBase: countmap
include("ecosystem_model_functions.jl")
include("bottum_up_cr.jl")

#This function gives the position at which we should place our nodes
# in order to get a food web in a tree format
function TrophicTree(G)
    tl = round.(get_trophic(G)) .+ 0.01*(rand())
    norm_tl = tl./maximum(tl)
    a = 2/sqrt(3)
    θ = tand(60)

    num_tl = countmap(norm_tl)

    pin = Dict()

    [pin[i] = 0 for i in 1:length(tl)]

    for level in unique(norm_tl)
        nodes = findall(x->x == level, norm_tl)
        num_node = num_tl[level]
        b = level/θ
        r = range(b, a - b, num_node + 1)
        x = (r[2:end] + r[1:end-1])/2

        for (i,n) in enumerate(nodes)
            pin[n] = (x[i], level)
        end
    end

    return pin
end

function compare_graphs(size)
    local sean_g
    my_g = ground_up_niche_model(size)
    not_same_size = true

    while not_same_size
        sean_g = bottum_up_niche(size,1,0.32)
        if nv(my_g[1]) == nv(sean_g.G)
            not_same_size = false
        end
    end

    return my_g[1], sean_g.G
end

function plot_comparison(g1, g2)

    f = GLMakie.Figure(resolution = (800,500))
    ax1 = f[1,1] = Axis(f; title = "My Graphs")
    ax2 = f[1,2] = Axis(f; title = "Seans Graphs")

    for ax in [ax1, ax2]
        xlims!(ax, 0,1.5); ylims!(ax,0,1.2); hidespines!(ax); hidedecorations!(ax)
    end

    p1 = graphplot!(ax1, g1; layout = Stress(;initialpos = TrophicTree(g1), pin = TrophicTree(g1)), node_size = [24 for _ in 1:nv(g1)], edge_color = [RGBA(0,0,0,0.7) for _ in 1:ne(g1)])
    p2 = graphplot!(ax2, g2; layout = Stress(;initialpos = TrophicTree(g2), pin = TrophicTree(g2)), node_size = [24 for I in 1:nv(g2)], edge_color = [RGBA(0,0,0,0.7) for _ in 1:ne(g2)])

    return f
end

