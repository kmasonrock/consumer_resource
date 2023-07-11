using DifferentialEquations, ModelingToolkit
using OrdinaryDiffEq
using DynamicalSystems
using Symbolics
using PyPlot
using LinearAlgebra
using Graphs
using Colors
using GraphPlot
using InvertedIndices
include("/Users/kmrock/Documents/network_control_project/network_control/DynamicsControl.jl")
include("/Users/kmrock/Documents/network_control_project/network_control/network_gen.jl")

# mutable struct Params
#     graph:: MetaDiGraph

# end

# function consumer_resource(du,u,p,t)



# end

#g = niche_model_graph(25,3.85)

# for i in 1:25

#     if outdegree(g,i) == 0
#         set_props!(g,i,Dict(:is_basal => true))
#     else
#         set_props!(g,i,Dict(:is_basal => false))
#     end

# end

global n_nodes = 25

# function condition(out,u,t,integrator)
#     for i in 1:n_nodes
#         out[i] = u[i] - 1e-10
#     end
# end

# function affect!(integrator, idx)
#     for id in idx
#         integrator.u[id] = 0
#     end
# end

# cb = VectorContinuousCallback(condition, affect!, n_nodes)

function get_assim_eff(g, herb, carn)
    basal = findall(x->x==0, indegree(g))

    e = zeros(length(vertices(g)),length(vertices(g)))

    for edge in edges(g)
        if indexin(edge.src,basal)[1] !== nothing
            e[edge.dst,edge.src] = herb
        else
            e[edge.dst,edge.src] = carn
        end
    end

    return e
end

function get_Ω(g)
    Ω = zeros(length(vertices(g)),length(vertices(g)))
    basal = findall(x->x==0, indegree(g))
    for i in [j for j in 1:length(vertices(g))][Not(basal)]
        for j in inneighbors(g,i)
            Ω[i,j] = 1/indegree(g,i)
        end
    end

    return Ω
end


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

function F_ij(Ω,ω,h,B,g,i,j)
    return (Ω[i,j]*(max(B[j],0) ^ h))/(1 + ω*max(B[i],0) + sum([Ω[i,k]*(max(B[k],0) ^ h) for k in inneighbors(g,i)]))
end


pars = @parameters t

vars = Symbolics.@variables B[1:n_nodes](t), M[1:n_nodes^2](t)

D = Differential(t)

global g = niche_model_graph(n_nodes,1.5)

d_eqs = [Bi for Bi in B]

d_eqs = D.(d_eqs)

dxdr = 0.88 # = vertibrates, 0.314 = invertibrates, 0.597 = both

#y = 4 # = vertibrates, 8 = invertibrates, 6 = both

#x_test = dxdr*(Z.^(get_trophic(g) .- 1)).^-0.25

ω = 0.05

Z = 10

rhs = []

x = dxdr*(Z.^(get_trophic(g) .- 1)).^(-0.25)
y = 4
h = 1.2

Ω = get_Ω(g)
e = get_assim_eff(g,0.45,0.85)

for i in 1:n_nodes
    
    prey = inneighbors(g,i)
    predators = outneighbors(g,i)
    if indegree(g,i) == 0

        append!(rhs, (1-max(B[i],0))*max(B[i],0) - (sum([x[j]*y*max(B[j],0)*F_ij(Ω,ω,h,B,g,j,i)/e[j,i] for j in predators])))
    else
        if predators == []
            append!(rhs, -x[i].*max(B[i],0) + (sum([x[i]*y*max(B[i],0)*F_ij(Ω,ω,h,B,g,i,j) for j in prey])))
        else
            append!(rhs, -x[i].*max(B[i],0) + (sum([x[i]*y*max(B[i],0)*F_ij(Ω,ω,h,B,g,i,j) for j in prey])) - (sum([x[j]*y*max(B[j],0)*F_ij(Ω,ω,h,B,g,j,i)/e[j,i] for j in predators])))
        end
    end
end

eqs = d_eqs .~ rhs

init_B = [get_prop(g,i,:B) for i in vertices(g)]

@named sys = ODESystem(eqs)

# init_params = [Ω => get_Ω(g),e => get_assim_eff(g,0.45,0.85),x => dxdr*(Z.^(get_trophic(g) .-1)).^(-0.25),y => 4,h => 1.2, ω => 0.05]

prob = ODEProblem(sys,init_B,[0,5000])
sol = OrdinaryDiffEq.solve(prob,Vern9(), saveat = 0.1, abstol = 1e-10, reltol = 1e-10)

println(length(findall(x->x>1e-9, sol[:,end])))

# if 10 <= length(findall(x->x>1e-9, sol[:,end])) <= 20

#     new_g = 
#     new_init = sol[:,end]

#     for i in eachindex(new_init)
#         given_init = copy(new_init)
#         given_init[i] = 0

#         prob_extinct = ODEProblem(sys, given_init, [0,5000])
#         sol


# end


# colors = [RGBA(0,0,0,1) for i in 1:5]

# trophic_level = [round(i;digits=2) for i in get_trophic(g)]

# top = findall(x->x == maximum(TL_vec), TL_vec)
# bottom = findall(x->x == minimum(TL_vec), TL_vec)

# colors[top] .= RGBA(1,0,0,1)
# colors[bottom] .= RGBA(0,1,0,1)

# graphplot(g, node_color = colors)



#plot_graph(g)

#stat_test(25,3.85,10000)

