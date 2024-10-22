

using TOML
using Graphs
using Glob
using Distributions
using PreallocationTools
using Infiltrator
using LinearAlgebra
using OrdinaryDiffEq
using UnPack
using InvertedIndices
using DiffEqCallbacks
using ForwardDiff
using CairoMakie
using GraphMakie
using JLD2
using SymEngine_jll
using SymEngine
using SymEngine: CMapBasicBasic, CDenseMatrix
using RuntimeGeneratedFunctions
using IterTools
using MacroTools: @capture, prewalk, MacroTools
using JuMP
import Ipopt, HSL_jll
using Static
using Distributed
using ProgressBars


RuntimeGeneratedFunctions.init(@__MODULE__)

### CHAPTER 1: STRUCTURES AND TYPES ###
    ## TYPES ##
        # TOML TYPES -- for knowing what part of the data goes where
        struct NetworkGen{x}
        end

        struct Dynamics{x}
        end

        struct Optimization{x}
        end

        struct Universal{x}
        end

        #ALLEE TYPES -- What kind of allee effect
        struct Hill{x}
        end

        struct Absent{x}
        end

        struct Strong{x}
        end

    ## STRUCTURES ##

        #Structure for all of the Network Gen Parameters
        mutable struct NetworkGenParams{T <: Real}
            β::T
            ex_thresh::T
            fp_thresh::T
            abstol::T
            reltol::T
            integrate_time::Vector{T}
            starting_B::T
            init_B::T
            basal_limit::T
            iter_limit::T
        end

        #Structure for the Dynamics Parameters
        mutable struct DynamicsParams{T <: Real}
            e_herb::T
            e_carn::T
            y::T
            dxdr::T
            Z::T
            m_trophic::T
            w::T
            B0::T
            S::T
            h::T
            m_migrate::T
            dyn_abstol::T
            dyn_reltol::T
            allee_type::Any
        end

        #Structure for the Optimization Algorithm
        #TODO: This does not need to be a mutable
        mutable struct OptimizationParams{T <: Real}
            iter_limit::T
            basal_lb::T
            nonbasal_lb::T
            closest_time::Vector{T}
            integrate_time::Vector{T}
            targets::Vector{Vector{T}}
            invaders::Vector{T}
            ϵ_max::T
            ϵ_min::T
            extinction::T
            time_increment::T
        end


        mutable struct CRParams{T <: Real}
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
            dummy_vals::Vector{DiffCache{Vector{T}, Vector{T}}}
        end

        #Structure for all of the general parameters
        mutable struct UniversalParams{T <: Real}
            log_space::Bool
            net_gen::NetworkGenParams{T}
            dyn::DynamicsParams{T}
            opt::OptimizationParams{T}
        end

        struct ModelParams
            func::ODEFunction
            J::Matrix{Float64}
            n::Int64
            crp::Union{Vector{Any}, CRParams}
        end

        struct SaveObject
            up::UniversalParams{Float64}
            crp::Vector{Any}
            stable::Vector{Float64}
        end
        

include("consumer_resource.jl")
include("FastMTK.jl")
include("perturbations.jl")
#--------------------------------------------------------------#

### CHAPTER 2: INITIALIZERS ###

#The following functions load all relevant variables for each given task
#in NetworkControl, determined by the type of initialization being done

#For Network Generation

function get_allee_type(allee::String)
    if allee == "Hill"
        return Hill
    elseif allee == "Absent"
        return Absent
    elseif allee == "Strong"
        return Strong
    else
        throw("The provided allee effect type is undefined! Valid types that have been included in NetworkControl are Hill, Absent, and Strong. Details of each type can be found in the documentation. If you want to establish a new type of allee effect, you must generate your own allee type and function to be associated with it.")
    end
end

function allee(u,S, ::Type{Hill})
    return u/(S + u)
end

function allee(u,S::Vector{Float64}, ::Type{Strong})
    return 1 - ((S[1]+ S[2])/(S[2] + u))^S[3]
end

function allee(u,S, ::Type{Absent})
    return 1
end

function initialize_params(file, ::Type{NetworkGen}, _β, _ex_thresh, _fp_thresh, _ngen_abstol, _ngen_reltol, 
    _netgen_integrate_time, _init_B, _starting_B, _basal_limit, _netgen_iter_limit)
    
    data = file["NetworkGenVars"]

    β = isnothing(_β) ? data["_beta"] : _β
    ex_thresh = isnothing(_ex_thresh) ? data["_extinction_threshold"] : _ex_thresh
    fp_thresh = isnothing(_fp_thresh) ? data["_fixedpoint_threshold"] : _fp_thresh
    ngen_abstol = isnothing(_ngen_abstol) ? data["_ngen_abstol"] : _ngen_abstol
    ngen_reltol = isnothing(_ngen_reltol) ? data["_ngen_reltol"] : _ngen_reltol
    netgen_integrate_time = isnothing(_netgen_integrate_time) ? data["_netgen_integrate_time"] : _netgen_integrate_time
    starting_B = isnothing(_starting_B) ? data["_starting_B"] : _starting_B
    init_B = isnothing(_init_B) ? data["_init_B"] : _init_B
    basal_limit = isnothing(_basal_limit) ? data["_basal_limit"] : _basal_limit
    netgen_iter_limit = isnothing(_netgen_iter_limit) ? data["_netgen_iter_limit"] : _netgen_iter_limit

    return NetworkGenParams(β, ex_thresh, fp_thresh, ngen_abstol, ngen_reltol, netgen_integrate_time, starting_B, init_B, basal_limit, netgen_iter_limit)
end

#For the dynamics
function initialize_params(file, ::Type{Dynamics}, _e_herb, _e_carn, _y, _dxdr, _Z, _m_trophic, _w, _B0,
    _S, _h, _m_migrate, _dyn_abstol, _dyn_reltol, _allee_type)


    data = file["DynamicsVars"]

    e_herb = isnothing(_e_herb) ? data["_e_herb"] : _e_herb
    e_carn = isnothing(_e_carn) ? data["_e_carn"] : _e_carn
    y = isnothing(_y) ? data["_y"] : _y
    dxdr = isnothing(_dxdr) ? data["_dxdr"] : _dxdr
    Z = isnothing(_Z) ? data["_Z"] : _Z
    m_trophic = isnothing(_m_trophic) ? data["_m_trophic"] : _m_trophic
    w = isnothing(_w) ? data["_w"] : _w
    B0 = isnothing(_B0) ? data["_B0"] : _B0
    S = isnothing(_S) ? data["_S"] : _S
    h = isnothing(_h) ? data["_h"] : _h
    m_migrate = isnothing(_m_migrate) ? data["_m_migrate"] : _m_migrate
    dyn_abstol = isnothing(_dyn_abstol) ? data["_dyn_abstol"] : _dyn_abstol
    dyn_reltol = isnothing(_dyn_reltol) ? data["_dyn_reltol"] : _dyn_reltol
    allee_type = isnothing(_allee_type) ? get_allee_type(data["_allee_type"]) : get_allee_type(_allee_type)


    return DynamicsParams(e_herb, e_carn, y, dxdr, Z, m_trophic, w, B0, S, h, m_migrate, dyn_abstol, dyn_reltol, allee_type)
end

#For the optimization
function initialize_params(file, ::Type{Optimization}, _opt_iter_limit, _basal_lb, _nonbasal_lb, _closest_time, _opt_integrate_time,
    _targets, _invaders,_max_epsilon, _min_epsilon, _extinction, _time_increment)
    data = file["OptimizationVars"]

    opt_iter_limit = isnothing(_opt_iter_limit) ? data["_opt_iter_limit"] : _opt_iter_limit
    basal_lb = isnothing(_basal_lb) ? data["_basal_lb"] : _basal_lb
    nonbasal_lb = isnothing(_nonbasal_lb) ? data["_nonbasal_lb"] : _nonbasal_lb
    closest_time = isnothing(_closest_time) ? data["_closest_time"] : _closest_time
    opt_integrate_time = isnothing(_opt_integrate_time) ? data["_opt_integrate_time"] : _opt_integrate_time
    targets = isnothing(_targets) ? data["_targets"] : _targets
    invaders = isnothing(_invaders) ? data["_invaders"] : _invaders
    max_epsilon = isnothing(_max_epsilon) ? data["_max_epsilon"] : _max_epsilon
    min_epsilon = isnothing(_min_epsilon) ? data["_min_epsilon"] : _min_epsilon
    extinction = isnothing(_extinction) ? data["_extinction"] : _extinction
    time_increment = isnothing(_time_increment) ? data["_time_increment"] : _time_increment

    return OptimizationParams(opt_iter_limit, basal_lb, nonbasal_lb, closest_time, opt_integrate_time, targets, invaders, max_epsilon, min_epsilon, extinction, time_increment)
end

function cr_params(g::SimpleDiGraph{Int64}, dyn_p::DynamicsParams{Float64}, allee_type::Type{Hill})
    Ω = get_Ω(g)
    e = get_assim_eff(g, dyn_p.e_herb, dyn_p.e_carn)
    x = get_x(dyn_p.dxdr, dyn_p.Z, dyn_p.m_trophic, get_trophic(g))
    n = nv(g)
    pred,prey,basal = get_pred_and_prey(g)

    dummy_vals = Vector{DiffCache{Vector{Float64}, Vector{Float64}}}(undef,  4)
    for i in 1:4
        dummy_vals[i] = DiffCache(zeros(nv(g)))
    end

    return CRParams(g, Ω, e, x, dyn_p.w, dyn_p.h, dyn_p.y, dyn_p.S, dyn_p.B0, n, dyn_p.m_migrate, pred, prey, basal, allee_type, dummy_vals)
end

function cr_params(g::SimpleDiGraph, dyn_p::DynamicsParams{Float64}, allee_type::Type{Strong})
    Ω = get_Ω(g)
    e = get_assim_eff(g, dyn_p.e_herb, dyn_p.e_carn)
    x = get_x(dyn_p.dxdr, dyn_p.Z, dyn_p.m_trophic, get_trophic(g))
    n = nv(g)
    pred,prey,basal = get_pred_and_prey(g)

    dummy_vals = Vector{DiffCache{Vector{Float64}, Vector{Float64}}}(undef,  4)
    for i in 1:4
        dummy_vals[i] = DiffCache(zeros(nv(g)))
    end

    return CRParams(g, Ω, e, x, dyn_p.w, dyn_p.h, dyn_p.y, dyn_p.S, dyn_p.B0, n, dyn_p.m_migrate, pred, prey, basal, allee_type, dummy_vals)
end

function cr_params(g::SimpleDiGraph, dyn_p::DynamicsParams{Float64}, allee_type::Type{Absent})
    Ω = get_Ω(g)
    e = get_assim_eff(g, dyn_p.e_herb, dyn_p.e_carn)
    x = get_x(dyn_p.dxdr, dyn_p.Z, dyn_p.m_trophic, get_trophic(g))
    n = nv(g)
    pred,prey,basal = get_pred_and_prey(g)

    dummy_vals = Vector{DiffCache{Vector{Float64}, Vector{Float64}}}(undef,  4)
    for i in 1:4
        dummy_vals[i] = DiffCache(zeros(nv(g)))
    end

    return CRParams(g, Ω, e, x, dyn_p.w, dyn_p.h, dyn_p.y, dyn_p.S, dyn_p.B0, n, dyn_p.m_migrate, pred, prey, basal, allee_type, dummy_vals)
end

### CHAPTER 3: FUNCTIONS FOR GRAPHS ###

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

function get_x(dxdr,Z,m,trophic)
    return dxdr*(Z.^(trophic .- 1)).^m
end

function initialize(;_β = nothing, _ex_thresh = nothing, _fp_thresh = nothing, _ngen_abstol = nothing, _ngen_reltol = nothing, _netgen_integrate_time = nothing,
    _init_B = nothing, _starting_B = nothing,  _basal_limit = nothing,  _netgen_iter_limit = nothing, _e_herb = nothing, _e_carn = nothing, _y = nothing,
    _dxdr = nothing, _Z = nothing, _m_trophic = nothing, _w = nothing, _B0 = nothing,_S = nothing, _h = nothing, _m_migrate = nothing, _dyn_abstol = nothing,
    _dyn_reltol = nothing, _allee_type = nothing, _opt_iter_limit = nothing, _basal_lb = nothing, _nonbasal_lb = nothing, _closest_time = nothing, 
    _opt_time_integrate = nothing, _targets = nothing, _invaders = nothing, _max_epsilon = nothing, _min_epsilon = nothing, _time_increment = nothing,
    _extinction = nothing, _log_space = nothing)

    to_transform(::NetworkGenParams{Float64}) = (:init_B, :starting_B, :ex_thresh)
    #to_transform(::DynamicsParams{Float64}) = [:m_migrate]
    to_transform(::OptimizationParams{Float64}) = (:basal_lb, :nonbasal_lb, :extinction)

    file = TOML.parsefile("Config.toml")
    log_space = isnothing(_log_space) ? file["UniversalVars"]["_log_space"] : _log_space

    net_gen = initialize_params(file, NetworkGen, _β, _ex_thresh, _fp_thresh, _ngen_abstol, _ngen_reltol, _netgen_integrate_time, _init_B, _starting_B, _basal_limit, _netgen_iter_limit)
    dyn = initialize_params(file, Dynamics, _e_herb, _e_carn, _y, _dxdr, _Z, _m_trophic, _w, _B0, _S, _h, _m_migrate, _dyn_abstol, _dyn_reltol, _allee_type)
    opt = initialize_params(file, Optimization, _opt_iter_limit, _basal_lb, _nonbasal_lb, _closest_time, _opt_time_integrate, _targets, _invaders, _max_epsilon, _min_epsilon, _extinction, _time_increment)
    if log_space
        for param in [net_gen, opt]
            for s in to_transform(param)
                setfield!(param, s, log(getfield(param, s)))
            end
        end
    end

    return UniversalParams(log_space, net_gen, dyn, opt)
end

