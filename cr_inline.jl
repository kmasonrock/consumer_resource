using Distributed
using OrdinaryDiffEq
using ModelingToolkit
using Graphs
using UnPack
include("ecosystem_model_functions.jl")
include("allee_effects.jl")

struct CRParams{T <: Real}
    g::SimpleDiGraph{Int64}
    Ω::Matrix{T}
    e::Matrix{T}
    x::Vector{T}
    w::T
    h::T
    y::T
    S::Union{T,Vector{T}}
    B0::T
    n::Int64
    m::T
    pred::Vector{Vector{Int64}}
    prey::Vector{Vector{Int64}}
    basal::BitVector
    allee_effect::Union{Type{Hill}, Type{Strong}, Type{Absent}}
end
# This is going to be the inline version of the consumer-resource model
# We want to do this because GO FAS



function cr_params(g::SimpleDiGraph, allee_type::Type{Hill}, B0::Float64 = 0.5)
    Ω = get_Ω(g)
    e = get_assim_eff(g,0.85,0.85)
    x = get_x(0.597, 10, -0.25, get_trophic(g))
    w = 0.0
    h = 1.2
    y = 10.0
    S = 0.001
    n = nv(g)
    m = 1e-10
    pred,prey,basal = get_pred_and_prey(g)

    return CRParams(g, Ω, e, x, w, h, y, S, B0, n, m, pred, prey, basal, allee_type)
end

function cr_params(g::SimpleDiGraph, allee_type::Type{Strong}, B0::Float64 = 0.5)
    Ω = get_Ω(g)
    e = get_assim_eff(g,0.85,0.85)
    x = get_x(0.597, 10, -0.25, get_trophic(g))
    w = 0.5
    h = 1.5
    y = 10.0
    S = [1e-8, 0.001, h]
    n = nv(g)
    m = 1e-10
    pred,prey,basal = get_pred_and_prey(g)

    return CRParams(g, Ω, e, x, w, h, y, S, B0, n, m, pred, prey, basal, allee_type)
end

function cr_params(g::SimpleDiGraph, allee_type::Type{Absent}, B0::Float64 = 0.5)
    Ω = get_Ω(g)
    e = get_assim_eff(g,0.85,0.85)
    x = get_x(0.597, 10, -0.25, get_trophic(g))
    w = 0.5
    h = 1.5
    y = 10.0
    S = 0.00001
    n = nv(g)
    m = 1e-10
    pred,prey,basal = get_pred_and_prey(g)

    return CRParams(g, Ω, e, x, w, h, y, S, B0, n, m, pred, prey, basal, allee_type)
end

function cr_f(du,u,p,t)
    #@unpack g,Ω,e,x,w,h,y,S,n,m = p

    for i in 1:p.n
        if in(i,p.basal)
            @views du[i] = (1 - u[i])*u[i] + p.m# .- [sum(x[j]*y*max(u[i],0)*F_ij(g,Ω,w,h,u[1:nv(g)],j,i)/e[j,i] for j in outdegree(g,i))]
        else
            @views du[i] = -p.x[i]*u[i] + p.m
        end

        for j in p.prey[i]
            #*(u[i])/(S + u[i])
            @views du[i] += p.x[i]*p.y*u[i]*F_ij(p.g,p.Ω,p.w,p.h,p.prey,u[1:p.n],i,j)*(u[i]/(p.S + u[i]))
        end

        for j in p.pred[i]
            @views du[i] -= p.x[j]*p.y*u[j]*F_ij(p.g,p.Ω,p.w,p.h,p.prey,u[1:p.n],j,i)/p.e[j,i]
        end

    end
    nothing
end


function cr_log_f(du, u, p, t)
    @unpack g, Ω, e, x, w, h, y, S, B0, n, m, pred, prey, basal,allee = p

    for i in 1:nv(g)
        if indegree(g,i) == 0
            @views du[i] = (1 - exp(u[i]))*exp(u[i])# .- [sum(x[j]*y*max(u[i],0)*F_ij(g,Ω,w,h,u[1:nv(g)],j,i)/e[j,i] for j in outdegree(g,i))]
            for j in outneighbors(g,i)
                @views du[i] -= x[j]*y*exp(u[j])*log_F_ij(g,Ω,w,h,u[1:nv(g)],j,i)/e[j,i]
            end
        else
            @views du[i] = -x[i]*exp(u[i])
            for j in inneighbors(g,i)
                #*(u[i])/(S + u[i])
                @views du[i] += x[i]*y*exp(u[i])*log_F_ij(g,Ω,w,h,u[1:nv(g)],i,j)*(exp(u[i])/(S + exp(u[i])))
            end

            if outdegree(g,i) != 0
                for j in outneighbors(g,i)
                    @views du[i] -= x[j]*y*exp(u[j])*log_F_ij(g,Ω,w,h,u,j,i)/e[j,i]
                end
            end
        end
        @views du[i] = exp(-u[i])*du[i]
    end
end

function ∂F_ij(g,Ω,w,h,B0,n,prey,u,i,j,k)
    @views return (i == k)*(j != k)*F_ij(g,Ω,w,h,B0,prey,u,i,j)*((-w*B0^h - h*Ω[i,i]*u[i]^(h-1))/(B0^h + w*u[i]*B0^h + sum(Ω[i,n]*u[n]^h for n in inneighbors(g,i); init = 0))) + (i != k)*(j == k)*(h/u[j])*F_ij(g,Ω,w,h,B0,prey,u,i,j)*(1 - F_ij(g,Ω,w,h,B0,prey,u,i,j)) - (i != k)*(j != k)*(h/u[k])*F_ij(g,Ω,w,h,B0,prey,u,i,j)*F_ij(g,Ω,w,h,B0,prey,u,i,k) - (i == k)*(j == k)*F_ij(g,Ω,w,h,B0,prey,u,i,k)*(w*B0^h/(B0^h + w*u[i]*B0^h +sum(Ω[i,n]*u[n]^h for n in inneighbors(g,i); init = 0)) + h/u[i] * (F_ij(g,Ω,w,h,B0,prey,u,i,k) - 1))
end
 
function log_∂F_ij(g,Ω,w,h,B0,n,prey,u,i,j,k)
    @views return (i == k)*(j != k)*log_F_ij(g,Ω,w,h,B0,prey,u,i,j)*((-w*exp(u[i])*B0^h - h*Ω[i,i]*exp(h*u[i]))/(B0^h + w*exp(u[i])*B0^h + sum(Ω[i,n]*exp(h*u[n]) for n in inneighbors(g,i); init = 0))) + (i != k)*(j == k)*h*log_F_ij(g,Ω,w,h,B0,prey,u,i,j)*(1 - log_F_ij(g,Ω,w,h,B0,prey,u,i,j)) - (i != k)*(j != k)*h*log_F_ij(g,Ω,w,h,B0,prey,u,i,j)log_F_ij(g,Ω,w,h,B0,prey,u,i,k) + (i == k)*(j == k)*log_F_ij(g,Ω,w,h,B0,prey,u,i,j)*(h - (w* B0^h *exp(u[i]) + h*Ω[i,i]*exp(h*u[i]))/(B0^h+ w*exp(u[i])*B0^h + sum(Ω[i,n]*exp(h*u[n]) for n in inneighbors(g,i))))
end
 

function _time_test_f(g, type::Union{Type{Hill}, Type{Strong}})
    p = cr_params(g, type)
    prob = ODEProblem(cr_f, 0.5.*ones(nv(g)), [0 5000], p)
    sol = solve(prob, Vern9())
    return sol
end

function _time_test_sym(g, type::Union{Type{Hill}, Type{Strong}})
    p = cr_params(g, type)
    _,Ω,e,x,h,w,y = [getfield(p,f) for f in fieldnames(typeof(p))]

    sys,_,_,_ = dynamics_gen(g,Ω,e,x,h,w,y,0.01)
    prob = ODEProblem(sys, 0.5*ones(nv(g)), [0 5000])
    sol = solve(prob, Vern9())

    return sol
end

function cr_jac(J,u,p,t)
    @unpack g,Ω,e,x,w,h,y,S,B0,n,m,allee = p

    for i in eachindex(1:n)
        for j in eachindex(1:n)
            @views J[i,j] = 0
            if indegree(g,i) == 0
                if j == i
                    @views J[i,j] = 1 - 2*u[i]
                    for η in outneighbors(g,i)
                        @views J[i,j] -= x[η]*y/e[η,i] * u[η]*∂F_ij(g,Ω,w,h,B0,n,u,η,i,j)
                    end
                else
                    for η in outneighbors(g,i)
                        @views J[i,j] -= x[η]*y/e[η,i]*((η == j)*F_ij(g,Ω,w,h,u,η,i) + u[η]*∂F_ij(g,Ω,w,h,B0,n,prey,u,η,i,j))
                    end
                end
            else
                if j == i
                    @views J[i,j] = -x[i]
                    for η in inneighbors(g,i)
                        @views J[i,j] += x[i]*y*(F_ij(g,Ω,w,h,u,i,η)*(u[i]/(u[i] + S)) + u[i]*∂F_ij(g,Ω,w,h,B0,n,prey,u,i,η,j)*(u[i]/(S + u[i])) + u[i]*F_ij(g,Ω,w,h,u,i,η)*(S/(S + u[i])^2))
                    end
                    for η in outneighbors(g,i)
                        @views J[i,j] -= y*((x[η]/e[η,i]) *((η == j)*F_ij(g,Ω,w,h,u[1:n],η,i) + u[η] * ∂F_ij(g,Ω,w,h,B0,n,prey,u,η,i,j)))
                    end

                    #@views J[i,j] = -x[i] + x[i]*y*sum(F_ij(g,Ω,w,h,u[1:n],i,η) + u[i]*∂F_ij(g,Ω,w,h,n,u[1:n],i,η,j) for η in inneighbors(g,i); init = 0) - y*sum((x[η]/e[η,i]) *((η == j)*F_ij(g,Ω,w,h,u[1:n],η,i) + u[η] * ∂F_ij(g,Ω,w,h,B0,n,u[1:n],η,i,j)) for η in outneighbors(g,i); init = 0)
                    #@views J[i,j] = -x[i] + x[i]*y*sum(F_ij(g,Ω,w,h,u[1:n],i,η)*(u[i]/(u[i] + S)) + u[i]*∂F_ij(g,Ω,w,h,n,u[1:n],i,η,j)*(u[i]/(S + u[i])) + u[i]*F_ij(g,Ω,w,h,u[1:n],i,η)*(S/(S + u[i])^2) for η in inneighbors(g,i); init = 0) - y*sum((x[η]/e[η,i]) *((η == j)*F_ij(g,Ω,w,h,u[1:n],η,i) + u[η] * ∂F_ij(g,Ω,w,h,B0,n,u[1:n],η,i,j)) for η in outneighbors(g,i); init = 0)
                    
                else
                    for η in inneighbors(g,i)
                        @views J[i,j] += x[i]*y*u[i]*∂F_ij(g,Ω,w,h,n,u[1:n],i,η,j)*(u[i]/(S + u[i]))
                    end
                    for η in outneighbors(g,i)
                        @views J[i,j] -= (x[η]*y/e[η,i])*((η == j)*F_ij(g,Ω,w,h,u[1:n],η,i) + u[η]*∂F_ij(g,Ω,w,h,B0,n,u[1:n],η,i,j))
                    end
                    #@views J[i,j] = sum(x[i]*y*u[i]*∂F_ij(g,Ω,w,h,n,u[1:n],i,η,j)*(u[i]/(S + u[i])) for η in inneighbors(g,i); init = 0) - sum((x[η]*y/e[η,i])*((η == j)*F_ij(g,Ω,w,h,u[1:n],η,i) + u[η]*∂F_ij(g,Ω,w,h,B0,n,u[1:n],η,i,j)) for η in outneighbors(g,i); init = 0)
                end
            end
        end
    end
    nothing
end

function log_cr_jac(J, u, p, t)

    for i in 1:nv(p.g)
        for j in 1:nv(p.g)
            @views J[i,j] = 0
            if indegree(p.g,i) == 0
                if i == j
                    @views J[i,j] = -exp(u[i])
                    for η in outneighbors(p.g,i)
                        @views  J[i,j] -= p.x[η]*p.y/p.e[η,i] * exp(u[η])*((exp(-u[i])*log_∂F_ij(p.g,p.Ω,p.w,p.h,nv(p.g),u[1:nv(p.g)],η,i,j)) - exp(-u[i])*log_F_ij(p.g,p.Ω,p.w,p.h,u[1:nv(p.g)],η,i))
                    end
                else
                    for η in outneighbors(p.g,i)
                        @views J[i,j] -= p.x[η]*p.y/p.e[η,i] * exp(-u[i]) * (log_∂F_ij(p.g,p.Ω,p.w,p.h,nv(p.g),u[1:nv(p.g)],η,i,j)*exp(u[η]) + (η == j)*exp(u[η])*log_F_ij(p.g,p.Ω, p.w, p.h, u[1:nv(p.g)], η, i))
                    end
                end
            else
                if i == j
                    for η in inneighbors(p.g,i)
                        @views J[i,j] += p.x[i]*p.y*(log_∂F_ij(p.g,p.Ω,p.w,p.h,nv(p.g),u[1:nv(p.g)],i,η,j)*(exp(u[i])/(p.S + exp(u[i]))) + log_F_ij(p.g,p.Ω,p.w,p.h,u[1:nv(p.g)],i,η)*(p.S*exp(u[i])/(p.S + exp(u[i]))^2))
                    end
                    for η in outneighbors(p.g,i)
                        @views J[i,j] -= p.x[η]*p.y/p.e[η,i] * ((exp(u[η])*(exp(-u[i])*log_∂F_ij(p.g,p.Ω,p.w,p.h,nv(p.g),u[1:nv(p.g)],η,i,j) - (η != j)*log_F_ij(p.g,p.Ω,p.w,p.h,u[1:nv(p.g)],η,i)*exp(-u[i]))))
                    end
                else
                    for η in inneighbors(p.g,i)
                        @views J[i,j] += p.x[i]*p.y*(exp(u[i])/(p.S + exp(u[i])))*log_∂F_ij(p.g,p.Ω,p.w,p.h,nv(p.g),u[1:nv(p.g)],i,η,j)
                    end
                    for η in outneighbors(p.g,i)
                        @views J[i,j] -= (p.x[η]*p.y/p.e[η,i]) * exp(-u[i])*((η == j)*exp(u[η])*log_F_ij(p.g,p.Ω,p.w,p.h,u[1:nv(p.g)],η,i) + exp(u[η])*log_∂F_ij(p.g,p.Ω,p.w,p.h,nv(p.g),u[1:nv(p.g)],η,i,j))
                    end
                end
            end
        end
    end
end


function sean_cr(du,u,p,t)
    @unpack g, Ω, e, x, w, h, y, S, B0, n, m, pred, prey, basal, allee = p

    for i in 1:nv(g)
        if indegree(g,i) == 0
            du[i] = (1 - u[i])*(u[i] - m)*(u[i]/(S+u[i]))
            for j in outneighbors(g,i)
                @views du[i] -= x[j]*y*((u[i] - m)/u[i])*u[j]*F_ij(g,Ω,w,h,u[1:nv(g)],j,i)/e[j,i]
            end
        else
            du[i] = -x[i]*(u[i] - 1e-10)
            for j in inneighbors(g,i)
                #*(u[i])/(S + u[i])
                @views du[i] += x[i]*y*(u[i] - m)*F_ij(g,Ω,w,h,u[1:nv(g)],i,j)*(u[i]/(S + u[i]))
            end

            if outdegree(g,i) != 0
                for j in outneighbors(g,i)
                    @views du[i] -= x[j]*y*((u[i] - m)/u[i])*u[j]*F_ij(g,Ω,w,h,u[1:nv(g)],j,i)/e[j,i]
                end
            end
        end
    end


end

function sean_cr_jac(J,u,p,t)
    @unpack g,Ω,e,x,w,h,y,S,B0,n,m,pred,prey,basal,allee = p

    n = Int(n)

    for i in eachindex(1:n)
        for j in eachindex(1:n)
            @views J[i,j] = 0
            if indegree(g,i) == 0
                if j == i
                    @views J[i,j] = (1 - u[i])*(u[i] - 1e-10)*∂C(u[i],S) + (1 - u[i])*C(u[i]) + (u[i] - 1e-10)*C(u[i])
                    for η in outneighbors(g,i)
                        @views J[i,j] -= (x[η]*y*u[η]/(u[i]*e[j,i]))*((u[i] - 1e-10)*∂F_ij(g,Ω,w,h,n,u[1:n],η,j,i) - (u[i] - m)*F_ij(g,Ω,w,h,n,u[1:n],η,i,j)/u[i] + F_ij(g,Ω,w,h,n,u[1:n],η,i,j))
                    end
                else
                    for η in outneighbors(g,i)
                        @views J[i,j] -= x[η]*y/e[η,i]*((η == j)*F_ij(g,Ω,w,h,u[1:n],η,i) + u[η]*∂F_ij(g,Ω,w,h,B0,n,u[1:n],η,i,j))
                    end
                end
            else
                if j == i
                    @views J[i,j] = -x[i]
                    for η in inneighbors(g,i)
                        @views J[i,j] += x[i]*y*(F_ij(g,Ω,w,h,u[1:n],i,η)*(u[i]/(u[i] + S)) + u[i]*∂F_ij(g,Ω,w,h,n,u[1:n],i,η,j)*(u[i]/(S + u[i])) + u[i]*F_ij(g,Ω,w,h,u[1:n],i,η)*(S/(S + u[i])^2))
                    end
                    for η in outneighbors(g,i)
                        @views J[i,j] -= y*((x[η]/e[η,i]) *((η == j)*F_ij(g,Ω,w,h,u[1:n],η,i) + u[η] * ∂F_ij(g,Ω,w,h,B0,n,u[1:n],η,i,j)))
                    end

                    #@views J[i,j] = -x[i] + x[i]*y*sum(F_ij(g,Ω,w,h,u[1:n],i,η) + u[i]*∂F_ij(g,Ω,w,h,n,u[1:n],i,η,j) for η in inneighbors(g,i); init = 0) - y*sum((x[η]/e[η,i]) *((η == j)*F_ij(g,Ω,w,h,u[1:n],η,i) + u[η] * ∂F_ij(g,Ω,w,h,B0,n,u[1:n],η,i,j)) for η in outneighbors(g,i); init = 0)
                    #@views J[i,j] = -x[i] + x[i]*y*sum(F_ij(g,Ω,w,h,u[1:n],i,η)*(u[i]/(u[i] + S)) + u[i]*∂F_ij(g,Ω,w,h,n,u[1:n],i,η,j)*(u[i]/(S + u[i])) + u[i]*F_ij(g,Ω,w,h,u[1:n],i,η)*(S/(S + u[i])^2) for η in inneighbors(g,i); init = 0) - y*sum((x[η]/e[η,i]) *((η == j)*F_ij(g,Ω,w,h,u[1:n],η,i) + u[η] * ∂F_ij(g,Ω,w,h,B0,n,u[1:n],η,i,j)) for η in outneighbors(g,i); init = 0)
                    
                else
                    for η in inneighbors(g,i)
                        @views J[i,j] += x[i]*y*u[i]*∂F_ij(g,Ω,w,h,n,u[1:n],i,η,j)*(u[i]/(S + u[i]))
                    end
                    for η in outneighbors(g,i)
                        @views J[i,j] -= (x[η]*y/e[η,i])*((η == j)*F_ij(g,Ω,w,h,u[1:n],η,i) + u[η]*∂F_ij(g,Ω,w,h,B0,n,u[1:n],η,i,j))
                    end
                    #@views J[i,j] = sum(x[i]*y*u[i]*∂F_ij(g,Ω,w,h,n,u[1:n],i,η,j)*(u[i]/(S + u[i])) for η in inneighbors(g,i); init = 0) - sum((x[η]*y/e[η,i])*((η == j)*F_ij(g,Ω,w,h,u[1:n],η,i) + u[η]*∂F_ij(g,Ω,w,h,B0,n,u[1:n],η,i,j)) for η in outneighbors(g,i); init = 0)
                end
            end
        end
    end
    nothing
end


function new_sean_cr!(du,u,p,t)
    @unpack g,Ω,e,x,w,h,y,S, B0,n,m, pred, prey, basal, allee_effect = p
    for i in 1:nv(g)
        if basal[i]
            du[i] = (1 - u[i])*(u[i] - m)*allee(u[i],S,allee_effect)
        else
            du[i] = -x[i]*(u[i] - m)
        end

        for j in prey[i]
            du[i] += x[i]*y*(u[i] - m)*allee(u[i],S,allee_effect)*F_ij(g,Ω,w,h,B0,prey,u,i,j)
        end

        for j in pred[i]
            du[i] -= x[j]*y*((u[i] - m)/u[i])*u[j]*F_ij(g,Ω,w,h,B0,prey,u,j,i)/e[j,i]
        end
    end
    nothing
end

function new_log_sean_cr!(du,u,p,t)
    @unpack g,Ω,e,x,w,h,y,S, B0,n,m, pred, prey, basal, allee_effect = p
    for i in 1:n
        # if basal[i]
        #     @views du[i] = (1 - exp(u[i]))*(exp(u[i]) - m)*log_allee(exp(u[i]),S,allee_effect)
        # else
        #     @views du[i] = -x[i]*(exp(u[i]) - m)
        # end
        @views du[i] = basal[i]*(1 - exp(u[i]))*(exp(u[i]) - m)*log_allee(exp(u[i]),S,allee_effect) + !basal[i]*(-x[i]*(exp(u[i]) - m))

        for j in prey[i]
            @views du[i] += x[i]*y*(exp(u[i]) - m)*log_allee(exp(u[i]),S,allee_effect)*log_F_ij(g,Ω,w,h,B0,prey,u,i,j)
        end

        for j in pred[i]
            @views du[i] -= x[j]*y*((exp(u[i]) - m)/exp(u[i]))*exp(u[j])*log_F_ij(g,Ω,w,h,B0,prey,u,j,i)/e[j,i]
        end

        @views du[i] = exp(-u[i])*du[i]
    end
    nothing
end

# function new_sean_jac(J, u, p, t)
#     @unpack g,Ω,e,x,w,h,y,S,n,m = p
#     n = Int(n)

#     for i in 1:nv(g)
#         for j in 1:nv(g)
#             if isempty(inneighbors(g,i))
#                 J[i,j] = (j == i)*((1 - u[i])*(u[i] - m)*∂allee(u[i],S,allee) + (1 - u[i])*allee(u[i],S,allee) - (u[i] - m)*allee(u[i],S,allee))
#             else
#                 J[i,j] = (j == i)*(-x[i])
#             end

#             for η in inneighbors(g,i)
#                 J[i,j] += y*x[i]*((∂F_ij(g,Ω,w,h,B0,n,u[1:n],η,i,j) + ∂F_ij(g,Ω,w,h,B0,n,u[1:n],η,i,j))*allee(u[i],S,allee) + (j == i)*(u[i] - m)*F_ij(g,Ω,w,h,u[1:n],i,η)*∂allee(u[i],S,allee) + (i == j)*F_ij(g,Ω,w,h,u[1:n],i,η)*allee(u[i],S,allee))
#             end

#             for η in outneighbors(g,i)
#                 J[i,j] -= (y*x[η]/(u[i]*e[η,i]))*((u[i] - m)*((i == j)*∂F_ij(g,Ω,w,h,n,u[1:n],η,j,i) + (j == η)*∂F_ij(g,Ω,w,h,n,u[1:n],η,j,i))*u[η] - (i == j)*(F_ij(g,Ω,w,h,u[1:n],η,i)*(u[i] - m)*u[η])/u[i] + (j == η)*(u[i] - m)*F_ij(g,Ω,w,h,u[1:n],η,i) + (i == j)*F_ij(g,Ω,w,h,u[1:n],η,i)*u[η])
#             end

#         end
#     end
# end


function new_sean_jac!(J,u,p,t)
    @unpack g,Ω,e,x,w,h,y,S,B0,n,m,pred,prey,basal,allee_effect = p

    n = Int(n)

    for i in eachindex(1:nv(g))
        for j in eachindex(1:nv(g))
            J[i,j] = 0
            if basal[i]
                if j == i
                    J[i,j] = (1 - u[i])*(u[i] - m)*∂allee(u[i],S,allee_effect) + (1 - u[i])*allee(u[i],S,allee_effect) - (u[i] - m)*allee(u[i],S,allee_effect)
                    for η in pred[i]
                        J[i,j] -= x[η]*y*u[η]/(u[i]*e[η,i]) * ((u[i] - m)*(∂F_ij(g,Ω,w,h,B0,n,prey,u[1:n],η,i,j) - F_ij(g,Ω,w,h,B0,prey,u[1:n],η,i)/u[i]) + F_ij(g,Ω,w,h,B0,prey,u[1:n],η,i))
                    end
                else
                    for η in pred[i]
                        J[i,j] -= (y*x[η]*(u[i] - m)/(u[i]*e[η,i])) * (∂F_ij(g,Ω,w,h,B0,n,prey,u[1:n],η,i,j)*u[η] + (j == η)*F_ij(g,Ω,w,h,B0,prey,u[1:n],η,i))
                    end
                end
            else
                if j == i
                    J[i,j] = -x[i]
                    for η in prey[i]
                        J[i,j] += y*x[i]*((u[i] - m)*(∂F_ij(g,Ω,w,h,B0,n,prey,u[1:n],i,η,j)*allee(u[i],S,allee_effect) + F_ij(g,Ω,w,h,B0,prey,u[1:n],i,η)*∂allee(u[i],S,allee_effect)) + F_ij(g,Ω,w,h,B0,prey,u[1:n],i,η)*allee(u[i],S,allee_effect))
                    end
                    for η in pred[i]
                        J[i,j] -=  (x[η]*y/(u[i]*e[η,i]))*(u[η]*((u[i] - m)*(∂F_ij(g,Ω,w,h,B0,n,prey,u[1:n],η,i,j) - F_ij(g,Ω,w,h,B0,prey,u[1:n],η,i)/u[i]) + F_ij(g,Ω,w,h,B0,prey,u[1:n],η,i)) + (j == η)*(u[i] - m)*F_ij(g,Ω,w,h,B0,prey,u[1:n],η,i))
                    end

                    #@views J[i,j] = -x[i] + x[i]*y*sum(F_ij(g,Ω,w,h,u[1:n],i,η) + u[i]*∂F_ij(g,Ω,w,h,n,u[1:n],i,η,j) for η in inneighbors(g,i); init = 0) - y*sum((x[η]/e[η,i]) *((η == j)*F_ij(g,Ω,w,h,u[1:n],η,i) + u[η] * ∂F_ij(g,Ω,w,h,B0,n,u[1:n],η,i,j)) for η in outneighbors(g,i); init = 0)
                    #@views J[i,j] = -x[i] + x[i]*y*sum(F_ij(g,Ω,w,h,u[1:n],i,η)*(u[i]/(u[i] + S)) + u[i]*∂F_ij(g,Ω,w,h,n,u[1:n],i,η,j)*(u[i]/(S + u[i])) + u[i]*F_ij(g,Ω,w,h,u[1:n],i,η)*(S/(S + u[i])^2) for η in inneighbors(g,i); init = 0) - y*sum((x[η]/e[η,i]) *((η == j)*F_ij(g,Ω,w,h,u[1:n],η,i) + u[η] * ∂F_ij(g,Ω,w,h,B0,n,u[1:n],η,i,j)) for η in outneighbors(g,i); init = 0)
                    
                else
                    for η in prey[i]
                        J[i,j] +=  y*x[i]*((u[i] - m)*∂F_ij(g,Ω,w,h,B0,n,prey,u[1:n],i,η,j)*allee(u[i],S,allee_effect) + (i == η)*((u[i] - m)*F_ij(g,Ω,w,h,B0,prey,u[1:n],i,η)*∂allee(u[i],S,allee_effect) + F_ij(g,Ω,w,h,B0,prey,u[1:n],i,η)*allee(u[i],S,allee_effect)))
                    end
                    for η in pred[i]
                        J[i,j] -= (x[η]*y/(u[i]*e[η,i]))*(u[η]*(∂F_ij(g,Ω,w,h,B0,n,prey,u[1:n],η,i,j)*(u[i] - m)) + (i == η)*(F_ij(g,Ω,w,h,B0,prey,u[1:n],η,i)*u[η] - (u[i] - m)*F_ij(g,Ω,w,h,B0,prey,u[1:n],η,i)/u[i]) + (j == η)*(u[i] - m)*F_ij(g,Ω,w,h,B0,prey,u[1:n],η,i))
                    end
                    #@views J[i,j] = sum(x[i]*y*u[i]*∂F_ij(g,Ω,w,h,n,u[1:n],i,η,j)*(u[i]/(S + u[i])) for η in inneighbors(g,i); init = 0) - sum((x[η]*y/e[η,i])*((η == j)*F_ij(g,Ω,w,h,u[1:n],η,i) + u[η]*∂F_ij(g,Ω,w,h,B0,n,u[1:n],η,i,j)) for η in outneighbors(g,i); init = 0)
                end
            end
        end
    end
    nothing
end

function new_log_sean_jac!(J, u, p, t)
    @unpack g,Ω,e,x,w,h,y,S,B0,n,m, pred, prey, basal, allee_effect = p

    for i in eachindex(1:nv(g))
        for j in eachindex(1:nv(g))
            @views J[i,j] = 0
            if basal[i]
                if j == i
                    @views J[i,j] -= (1 - exp(u[i]))*(exp(u[i]) - m)*log_allee(u[i],S,allee_effect) - sum(y*(exp(u[i]) - m)*log_F_ij(g,Ω,w,h,B0,prey,u[1:nv(g)],η,i)*exp(-u[i])*exp(u[η])*x[η]/e[η,i] for η in outneighbors(g,i))
                    @views J[i,j] += (1 - exp(u[i]))*(exp(u[i]) - m)*∂log_allee(u[i],S,allee_effect) + (1 - exp(u[i]))*log_allee(u[i],S,allee_effect)*exp(u[i]) - (exp(u[i]) - m)*log_allee(u[i],S,allee_effect)*exp(u[i])
                    for η in pred[i]
                        @views J[i,j] -= (y*(exp(u[i]) - m)*log_∂F_ij(g,Ω,w,h,B0,n,prey,u,i,j,k)*exp(-u[i])*exp(u[η])*x[η]/e[η,i]) - (y*(exp(u[i]) - m)*log_F_ij(g,Ω,w,h,B0,prey,u[1:nv(g)],η,i)*exp(-u[i])*exp(u[η])*x[η]/e[η,i]) + (y*log_F_ij(g,Ω,w,h,B0,prey,u[1:nv(g)],η,i)*exp(u[η])*x[η]/e[η,i])
                    end
                else
                    for η in pred[i]
                        @views J[i,j] -= (y*(exp(u[i]) - m)*log_∂F_ij(g,Ω,w,h,B0,n,prey,u,i,j,k)*exp(-u[i])*exp(u[η])*x[η]/e[η,i]) + (j == η)*(y*(exp(u[i]) - m)*log_F_ij(g,Ω,w,h,B0,prey,u[1:nv(g)],η,i)*exp(-u[i])*exp(u[j])*x[η]/e[η,i])
                    end
                end
            else
                if j == i
                    @views J[i,j] -= (-x[i]*(exp(u[i]) - m)) + sum(y*(exp(u[i]) - m)*log_allee(u[i],S,allee_effect)*log_F_ij(g,Ω,w,h,B0,prey,u[1:nv(g)],i,η)*x[i] for η in inneighbors(g,i); init = 0) - sum(y*(exp(u[i]) - m)*log_F_ij(g,Ω,w,h,B0,prey,u[1:nv(g)],η,i)*exp(-u[i])*exp(u[η])*x[η]/e[η,i] for η in outneighbors(g,i); init  = 0)
                    @views J[i,j] += (-x[i]*exp(u[i]))
                    for η in prey[i]
                        @views J[i,j] += y*(exp(u[i]) - m)*log_∂F_ij(g,Ω,w,h,B0,n,prey,u[1:nv(g)],i,η,j)*log_allee(u[i],S,allee_effect)*x[i] + y*(exp(u[i]) - m)*log_F_ij(g,Ω,w,h,B0,prey,u[1:nv(g)],i,η)*∂log_allee(u[i],S,allee_effect)*x[i] + y*log_allee(u[i],S,allee_effect)*log_F_ij(g,Ω,w,h,B0,prey,u[1:nv(g)],i,η)*exp(u[i])*x[i]
                    end
                    for η in pred[i]
                        @views J[i,j] -= (y*(exp(u[i]) - m)*log_∂F_ij(g,Ω,w,h,B0,n,prey,u,i,j,k)*exp(-u[i])*exp(u[η])*x[η]/e[η,i]) - (y*(exp(u[i]) - m)*log_F_ij(g,Ω,w,h,B0,prey,u[1:nv(g)],η,i)*exp(-u[i])*exp(u[η])*x[η]/e[η,i]) + (y*log_F_ij(g,Ω,w,h,B0,prey,u[1:nv(g)],η,i)*exp(u[η])*x[η]/e[η,i] + (η == j)*(y*(exp(u[i]) - m)*log_F_ij(g,Ω,w,h,B0,prey,u[1:nv(g)],η,i)*exp(-u[i])*exp(u[η])*x[η]/e[η,i]))
                    end

                    #@views J[i,j] = -x[i] + x[i]*y*sum(F_ij(g,Ω,w,h,u[1:n],i,η) + u[i]*∂F_ij(g,Ω,w,h,n,u[1:n],i,η,j) for η in inneighbors(g,i); init = 0) - y*sum((x[η]/e[η,i]) *((η == j)*F_ij(g,Ω,w,h,u[1:n],η,i) + u[η] * ∂F_ij(g,Ω,w,h,B0,n,u[1:n],η,i,j)) for η in outneighbors(g,i); init = 0)
                    #@views J[i,j] = -x[i] + x[i]*y*sum(F_ij(g,Ω,w,h,u[1:n],i,η)*(u[i]/(u[i] + S)) + u[i]*∂F_ij(g,Ω,w,h,n,u[1:n],i,η,j)*(u[i]/(S + u[i])) + u[i]*F_ij(g,Ω,w,h,u[1:n],i,η)*(S/(S + u[i])^2) for η in inneighbors(g,i); init = 0) - y*sum((x[η]/e[η,i]) *((η == j)*F_ij(g,Ω,w,h,u[1:n],η,i) + u[η] * ∂F_ij(g,Ω,w,h,B0,n,u[1:n],η,i,j)) for η in outneighbors(g,i); init = 0)
                    
                else
                    for η in prey[i]
                        @views J[i,j] += y*(exp(u[i]) - m)*log_∂F_ij(g,Ω,w,h,B0,n,prey,u[1:nv(g)],i,η,j)*log_allee(u[i],S,allee_effect)*x[i]
                    end
                    for η in pred[i]
                        @views J[i,j] -= (y*(exp(u[i]) - m)*log_∂F_ij(g,Ω,w,h,B0,n,prey,u,i,j,k)*exp(-u[i])*exp(u[η])*x[η]/e[η,i]) + (j == η)*(y*(exp(u[i]) - m)*log_F_ij(g,Ω,w,h,B0,prey, u[1:nv(g)],η,i)*exp(-u[i])*exp(u[j])*x[η]/e[η,i])
                    end
                    #@views J[i,j] = sum(x[i]*y*u[i]*∂F_ij(g,Ω,w,h,n,u[1:n],i,η,j)*(u[i]/(S + u[i])) for η in inneighbors(g,i); init = 0) - sum((x[η]*y/e[η,i])*((η == j)*F_ij(g,Ω,w,h,u[1:n],η,i) + u[η]*∂F_ij(g,Ω,w,h,B0,n,u[1:n],η,i,j)) for η in outneighbors(g,i); init = 0)
                end
            end
        @views J[i,j] = J[i,j]*exp(-u[i])
        end
    end
    nothing
end