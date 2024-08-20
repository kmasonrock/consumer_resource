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



function cr_params(g::SimpleDiGraph, allee_type::Type{Hill}, B0::Float64 = 0.5, h::Float64 = 1.0)
    Ω = get_Ω(g)
    e = get_assim_eff(g,0.45,0.85)
    x = get_x(0.597, 10, -0.25, get_trophic(g))
    w = 0.0
    y = 6.0
    S = 1e-6
    n = nv(g)
    m = 1e-10
    pred,prey,basal = get_pred_and_prey(g)

    return CRParams(g, Ω, e, x, w, h, y, S, B0, n, m, pred, prey, basal, allee_type)
end

function cr_params(g::SimpleDiGraph, allee_type::Type{Strong},  B0::Float64 = 0.5, h::Float64 = 1.0)
    Ω = get_Ω(g)
    e = get_assim_eff(g,0.85,0.85)
    x = get_x(0.597, 10, -0.25, get_trophic(g))
    w = 0.5
    y = 6.0
    S = [1e-8, 0.001, h]
    n = nv(g)
    m = 1e-10
    pred,prey,basal = get_pred_and_prey(g)

    return CRParams(g, Ω, e, x, w, h, y, S, B0, n, m, pred, prey, basal, allee_type)
end

function cr_params(g::SimpleDiGraph, allee_type::Type{Absent},  B0::Float64 = 0.5, h::Float64 = 1.0)
    Ω = get_Ω(g)
    e = get_assim_eff(g,0.85,0.85)
    x = get_x(0.597, 10, -0.25, get_trophic(g))
    w = 0.0
    y = 10.0
    S = 1e-6
    n = nv(g)
    m = 1e-10
    pred,prey,basal = get_pred_and_prey(g)

    return CRParams(g, Ω, e, x, w, h, y, S, B0, n, m, pred, prey, basal, allee_type)
end
