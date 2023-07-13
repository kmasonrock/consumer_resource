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


function get_x(dxdr,Z,m,trophic)
    return dxdr*(Z.^(trophic .- 1)).^m
end

function consumer_resource(n_nodes,β,dxdr,Z,m,y,h,ω,herb,carn)
    pars = @parameters t
    vars = Symbolics.@variables B[1:n_nodes](t)
    graph_failed = true

    while graph_failed
        try
            global g = niche_model_graph(n_nodes,β)
            global x = get_x(dxdr,Z,m,get_trophic(g))
            graph_failed = false
            println("Graph is generated")
        catch
            println("Failed to generate. Trying again...")
            graph_failed = true
        end
    end
    
    Ω = get_Ω(g)
    e = get_assim_eff(g,herb,carn)
    D = Differential(t)

    d_eqs = [Bi for Bi in B]

    d_eqs = D.(d_eqs)
    rhs = []
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

    return g, init_B, sys
end


global n_nodes = 25
global dxdr = 0.88
global ω = 0.05
global Z = 10
global m = -0.25
global y = 4
global h = 1.2
global β = 1.5
global graph_number = 14

while true
    global graph_number
    global g

    g, init_B, sys = consumer_resource(n_nodes,β,dxdr,Z,m,y,h,ω,0.45,0.85)

    prob = ODEProblem(sys,init_B,[0,5000])
    sol = OrdinaryDiffEq.solve(prob,Vern9(), saveat = 0.1, abstol = 1e-10, reltol = 1e-10)

    global living = findall(x -> x>1e-9, sol[:,end])
    println(length(living))
    if 10 <= length(living) <= 20
        println("Graph has enough alive. Checking connectedness...")
        copy_g = copy(SimpleDiGraph(g))
        rem_vertices!(copy_g, vertices(g)[Not(living)])

        if is_connected(copy_g)
            println("Graph is connected! saving as graph_$graph_number")
            new_g = MetaDiGraph(copy_g)
            for (ind,i) in enumerate(living)
                set_props!(new_g, ind, Dict(:B => sol[:,end][i]))
            end
            savegraph("network_control/saved_graphs/graph_$graph_number.lgz",new_g)
            graph_number += 1
        else
            println("Failure. Graph is not connected.")
        end
    end

    if graph_number > 100
        break
    end
end