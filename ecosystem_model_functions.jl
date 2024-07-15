using Distributed
using ModelingToolkit, OrdinaryDiffEq
using JLD2
using Graphs, MetaGraphs
using GLMakie, GraphMakie
using LinearAlgebra
using Symbolics
using Random
using InvertedIndices
using Distributions
using Infiltrator

#include("cr_inline.jl")

function get_assim_eff(g, herb, carn)
    copy_g = SimpleDiGraph(g)
    n = nv(g)
    e = zeros(n,n)
    basal = findall(x-> x==0, indegree(copy_g))

    for edge in edges(copy_g)
        if indexin(edge.src, basal)[1] !== nothing
            e[edge.dst,edge.src] = herb
        else
            e[edge.dst,edge.src] = carn
        end
    end

    return e
end

function get_pred_and_prey(g::SimpleDiGraph)
    prey = Vector{Vector{Int64}}(undef, nv(g))
    pred = Vector{Vector{Int64}}(undef, nv(g))
    for i in 1:nv(g)
        prey[i] = inneighbors(g,i)
        pred[i] = outneighbors(g,i)
    end

    basal = isempty.(prey)

    return pred, prey, basal
end


function get_Ω(g)
    n = nv(g)
    Ω = zeros(n,n)
    basal = findall(x->x==0, indegree(g))
    for i in [j for j in 1:n][Not(basal)]
        for j in inneighbors(g,i)
            Ω[i,j] = 1/indegree(g,i)
        end
    end

    return Ω
end

function get_trophic(g)
        A = Matrix(adjacency_matrix(g))
        d = indegree(g)
        basal = findall(x -> x == 0, d)
        d[basal] .+= 1
        D = diagm(d)

        tl = inv(D - transpose(A))*(D*ones(nv(g)))

    return tl
end 

function F_ij(g,Ω,ω,h,prey,B,i,j)
    #return (Ω[i,j]*(max(B[j],0)^h))/(1 + ω*max(B[i],0) + sum(Ω[i,k]*(max(B[k],0) ^ h) for k in inneighbors(g,i)))
    return @views (Ω[i,j]*(B[j][1]^h))/(0.5^h + ω*B[i]*(0.5^h) + sum(Ω[i,k]*(B[k][1] ^ h) for k in prey[i]; init = 0))
end
 
function log_F_ij(g,Ω,ω,h,prey,B,i,j)
    #return (Ω[i,j]*(max(B[j],0)^h))/(1 + ω*max(B[i],0) + sum(Ω[i,k]*(max(B[k],0) ^ h) for k in inneighbors(g,i)))
    return @views (Ω[i,j]*exp(h*B[j]))/(0.5^h + ω*exp(B[i])*0.5^h + sum(Ω[i,k]*exp(h*B[k]) for k in prey[i]; init = 0))
end 


function get_x(dxdr,Z,m,trophic)
    return dxdr*(Z.^(trophic)).^m
end

function get_node_params(β)
    η = rand(Uniform(0,1))
    b = 0.5
    r = rand(Beta(1,β))*η
    c = rand(Uniform(r/2, η))
    range = [c - r/2, c + r/2]

    return η, b, range
end

function dynamics_gen(g, Ω, e, x, ω, h, y, S)
    #Note -- this has been changed to be in log scale
    n = nv(g)

    @parameters t
    Symbolics.@variables B(t)[1:n], β(t)[1:n]

    rhs = Array{Num}(undef,1,n)

    @views D = Differential(t)
    @views d_eqs = D.(B)
    @views d_log_eqs = D.(β)

    for i in eachindex(vertices(g))
        @views prey = inneighbors(g,i)
        @views predators = outneighbors(g,i)
        #Last Update: December 1st, 2023
        if indegree(g,i) == 0
            if ~isempty(predators)
                #rhs[i] = (1 .- max(B[i],0)).*max(B[i],0) .- (sum(x[j].*y.*max(B[j],0).*F_ij(g,Ω,ω,h,B,j,i)./e[j,i] for j in predators)) + 1e-10
                rhs[i] = (1 .- B[i]).*B[i] .- (sum(x[j].*y.*B[j].*F_ij(g,Ω,ω,h,B,j,i)./e[j,i] for j in predators)) + 1e-10
            else
                #rhs[i] = (1 .- max(B[i],0)).*max(B[i],0)
                rhs[i] = (1 .- B[i]).*B[i]
            end
        else
            if predators == []
                #New Test -- Nov 29, 2023
                #I'm adding this original growth term to the predators.
                #rhs[i] = -x[i].*max(B[i],0) .+ sum(x[i].*y.*max(B[i],0).*F_ij(g,Ω,ω,h,B,i,j) for j in prey)*(max(B[i],0)/(S + max(B[i],0))) + 1e-10
                #*(B[i]/(S + B[i]))
                rhs[i] = -x[i].*B[i] .+ sum(x[i].*y.*B[i].*F_ij(g,Ω,ω,h,prey,B,i,j)*(B[i]/(S + B[i])) for j in prey) + 1e-10
            else
                #rhs[i] = -x[i].*max(B[i],0) .+ sum(x[i].*y.*max(B[i],0).*F_ij(g,Ω,ω,h,B,i,j) for j in prey)*(max(B[i],0)/(S + max(B[i],0))) .- sum(x[j].*y.*max(B[j],0).*F_ij(g,Ω,ω,h,B,j,i)./e[j,i] for j in predators) + 1e-10
                #*(B[i]/(S + B[i]))
                rhs[i] = -x[i].*B[i] .+ sum(x[i].*y.*B[i].*F_ij(g,Ω,ω,h,prey,B,i,j)*(B[i]/(S + B[i])) for j in prey) .- sum(x[j].*y.*B[j].*F_ij(g,Ω,ω,h,B,j,i)./e[j,i] for j in predators) + 1e-10
            end
        end
    end

    eqs = [d_eqs[i] ~ rhs[i] for i in 1:n]

    log_rhs = [exp(-β[i])*substitute(rhs[i], Dict(B => exp.(β))) for i in 1:n]

    log_eqs = [d_log_eqs[i] ~ log_rhs[i] for i in 1:n]

    @named sys = ODESystem(eqs)
    @named log_sys = ODESystem(log_eqs)

    return sys, log_sys,  B, t
end

function log_f(dy,y,p,t)
    dy .= exp.(-y).*p.f(exp.(y),nothing,nothing)
    nothing
end

function ground_up_niche_model(n_nodes, basal_start, basal_lim; β = 1.5, herb = 0.85, carn = 0.85, ω = 0.5, h = 1.5, y = 10.0, dxdr = 0.597, Z = 10.0, m = -0.25, S = 0.01)
    
    local sol
    local g
    #local sys
    #local log_sys
    local B
    local t
    local stable_state

    function condition(u,t,i)
        minimum(u) - 1e-7
    end

    affect!(integrator) = terminate!(integrator)

    cb = ContinuousCallback(condition, affect!)

    local g = MetaDiGraph()


    iter = 0
    node = 1
    while node <= n_nodes

        if node == 1
            add_vertex!(g)
            η, b, range = get_node_params(β)
            set_props!(g,1,Dict(:η => η, :b => b, :range => range))
            node = 2
        end

        add_vertex!(g)
        η, b, range = get_node_params(β)
        set_props!(g,node,Dict(:η => η, :b => b, :range => range))

        for i in 1:node
            for j in 1:node
                if get_prop(g,i,:range)[1] <= get_prop(g,j,:η) <= get_prop(g,i,:range)[2]
                    add_edge!(g,j,i)
                end
            end
        end

        if is_connected(g) == false
            rem_vertex!(g,node)
            continue
        end

        if length([inneighbors(g,i) for i in 1:nv(g) if isempty(inneighbors(g,i))])/n_nodes > basal_lim
            rem_vertex!(g,node)
            continue
        end

        try
            p = cr_params(SimpleDiGraph(g))
            prob = ODEProblem(cr_f, 0.5*ones(nv(g)), [0 5000], p)
            sol = solve(prob, Vern9())
            # Ω = get_Ω(g)
            # e = get_assim_eff(g,herb,carn)
            # x = get_x(dxdr,Z,m,get_trophic(g))
            # sys, _, B, t = dynamics_gen(g,Ω,e,x,ω,h,y,S)
            # func = ODEFunction(sys)
            # prob = ODEProblem(func,[get_prop(g,i,:b) for i in vertices(g)], [0,5000])
            # sol = solve(prob, Vern9())
            #prob = ODEProblem(log_f,[log(get_prop(g,i,:b)) for i in vertices(g)], [0,5000], func, callback = cb)
        catch
            #println("error")
            rem_vertex!(g,node)
            if node <= 6
            g = MetaDiGraph()
            node = 2
                add_vertex!(g)
                η, b, range = get_node_params(β)
                set_props!(g,1,Dict(:η => η, :b => b, :range => range))
            end
            continue
        end

        if all(x -> x > 1e-6, sol[end]) & (sol.t[end] == 5000)
            stable_state = sol[end]
            node += 1
        else
            iter += 1
            rem_vertex!(g,node)
            if iter > 100
                g = MetaDiGraph()
                node = 1
            end
            continue
        end

    end
    return g # B, t, stable_state
end

function param_gen(ω,h)
    
    g = SimpleDiGraph()
    add_vertex!(g)
    add_vertex!(g)
    add_vertex!(g)

    add_edge!(g,1,2)
    add_edge!(g,1,3)
    
    Ω = get_Ω(g)
    e = get_assim_eff(g,0.85,0.85)
    x = get_x(0.597, 10, -0.25, get_trophic(g))

    p = [g,Ω,e,x,ω,h,10]

    return p
end

function new_dynamics_gen(g, Ω, e, x, ω, h, y, S1,S2,S3, m, prey)
    #Note -- this has been changed to be in log scale
    n = nv(g)

    @parameters t
    Symbolics.@variables (B(t))[1:n], (β(t))[1:n]

    rhs = Vector{Num}(undef,n)

    @views D = Differential(t)

    for i in eachindex(vertices(g))
        #@views prey = inneighbors(g,i)
        @views predators = outneighbors(g,i)
        #Last Update: December 1st, 2023
        if indegree(g,i) == 0
            if ~isempty(predators)
                #rhs[i] = (1 .- max(B[i],0)).*max(B[i],0) .- (sum(x[j].*y.*max(B[j],0).*F_ij(g,Ω,ω,h,B,j,i)./e[j,i] for j in predators)) + 1e-10
                #rhs[i] = (1 .- B[i]).*(B[i] - m).*(B[i]/(S + B[i])) .- (sum(x[j].*y.*(B[i] - m).*B[j].*F_ij(g,Ω,ω,h,B,j,i)./(B[i]*e[j,i]) for j in predators))
                rhs[i] = (1 .- B[i]).*(B[i] - m).*(1 - ((S1 + S2)/(S2 + B[i]))^S3) .- (sum(x[j].*y.*(B[i] - m).*B[j].*F_ij(g,Ω,ω,h,prey,B,j,i)./(B[i]*e[j,i]) for j in predators))
            else
                #rhs[i] = (1 .- max(B[i],0)).*max(B[i],0)
                rhs[i] = (1 .- B[i]).(B[i] - m)*(1 - ((S1 + S2)/(S2 + B[i]))^S3)
            end
        else
            if predators == []
                #New Test -- Nov 29, 2023
                #I'm adding this original growth term to the predators.
                #rhs[i] = -x[i].*max(B[i],0) .+ sum(x[i].*y.*max(B[i],0).*F_ij(g,Ω,ω,h,B,i,j) for j in prey)*(max(B[i],0)/(S + max(B[i],0))) + 1e-10
                #*(B[i]/(S + B[i]))
                rhs[i] = -x[i].*(B[i] - m) .+ sum(x[i].*y.*(B[i] - m).*F_ij(g,Ω,ω,h,prey,B,i,j)*(1 - ((S1 + S2)/(S2 + B[i]))^S3) for j in prey[i])
            else
                #rhs[i] = -x[i].*max(B[i],0) .+ sum(x[i].*y.*max(B[i],0).*F_ij(g,Ω,ω,h,B,i,j) for j in prey)*(max(B[i],0)/(S + max(B[i],0))) .- sum(x[j].*y.*max(B[j],0).*F_ij(g,Ω,ω,h,B,j,i)./e[j,i] for j in predators) + 1e-10
                #*(B[i]/(S + B[i]))
                rhs[i] = -x[i].*(B[i] - m) .+ sum(x[i].*y.*(B[i] - m).*F_ij(g,Ω,ω,h,prey,B,i,j)*(1 - ((S1 + S2)/(S2 + B[i]))^S3) for j in prey[i]) .- sum(x[j].*y.*((B[i] - m)/B[i]).*B[j].*F_ij(g,Ω,ω,h,prey,B,j,i)/e[j,i] for j in predators)
            end
        end
    end

    B_ = collect(B)
    β_ = collect(β)
    
    eqs = @. D(B_) ~ rhs 

    log_rhs = [exp(-β[i])*substitute(rhs[i], Dict(B => exp.(β))) for i in 1:n]

    log_eqs = @. D(β_) ~ log_rhs

    @named sys = ODESystem(eqs, t, B_, [])
    @named log_sys = ODESystem(log_eqs, t, β_, [])

    return complete(sys), complete(log_sys),  B, t
end

