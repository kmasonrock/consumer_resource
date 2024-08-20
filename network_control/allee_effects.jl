
using UnPack
#------ Types of Allee ------#
struct Hill{x}
end

struct Strong{x}
end

struct Absent{x}
end


#------ Allee Effects ------#

function allee(u,S, ::Type{Hill})
    return u/(S + u)
end

function allee(u,S::Vector{Float64}, ::Type{Strong})
    return 1 - ((S[1]+ S[2])/(S[2] + u))^S[3]
end

function allee(u,S, ::Type{Absent})
    return 1
end

#------ Partial Allee Effects ------#

function ∂allee(u,S, ::Type{Hill})
    return S/(S + u)^2
end

function ∂allee(u,S::Vector{Float64}, ::Type{Strong})
    S1,S2,S3 = p.S
    return (S[3]*((S[1] + S[2])/(S[2] + u))^S[3])/(S[2] + u)
end

function ∂allee(u,p, ::Type{Absent})
    return 0
end

#------ Log Allee Effects ------#

function log_allee(u,S, ::Type{Hill})
    return exp(u)/(S + exp(u))
end

function log_allee(u,S, ::Type{Strong})
    return 1 - ((S[1] + S[2])/(S[2] + exp(u)))^S[3]
end

function log_allee(u, S, ::Type{Absent})
    return 1
end

#------ Log Partial Allee Effects ------#

function ∂log_allee(u,S, ::Type{Hill})
    return (exp(u)/(S + exp(u))) - (exp(2*u)/((exp(u) + S)^2))
end

function ∂log_allee(u,S::Vector{Float64}, ::Type{Strong})
    return (exp(u)*S[3]*((S[1] + S[2])/(S[2] + exp(u)))^S[3])/(S[2] + exp(u))
end

function ∂log_allee(u, S, ::Type{Absent})
    return 0
end