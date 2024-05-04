
#------ Types of Allee ------#
struct Hill{x}
end

struct Strong{x}
end


#------ Allee Effects ------#

function allee(u,p, type::Type{Hill})
    return u/(p.S + u)
end

function allee(u,p, type::Type{Strong})
    S1,S2,S3 = p.S
    return 1 - ((S1+ S2)/(S2 + u))^S3
end

#------ Partial Allee Effects ------#

function ∂allee(u,p,type::Type{Hill})
    return p.S/(p.S + u)^2
end

function ∂allee(u,p,type::Type{Strong})
    S1,S2,S3 = p.S
    return (S3*((S1 + S2)/(S2 + u))^S3)/(S2 + u)
end

#------ Log Allee Effects ------#

function log_allee(u,p, type::Type{Hill})
    return exp(u)/(p.S + exp(u))
end

function log_allee(u,p, type::Type{Strong})
    S1,S2,S3 = p.S
    return 1 - ((S1 + S2)/(S2 + exp(u)))^S3
end

#------ Log Partial Allee Effects ------#

function ∂log_allee(u,p,type::Type{Hill})
    return (exp(u)/(p.S + exp(u))) - (exp(2*u)/((exp(u) + p.S)^2))
end

function ∂log_allee(u,p,type::Type{Strong})
    S1,S2,S3 = p.S
    return (exp(u)*S3*((S1 + S2)/(S2 + exp(u)))^S3)/(S2 + exp(u))
end