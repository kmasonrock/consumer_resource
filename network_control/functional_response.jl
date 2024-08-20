#=

    ____   ____     _   __ ____  ______   ______ __  __ ___     _   __ ______ ______
   / __ \ / __ \   / | / // __ \/_  __/  / ____// / / //   |   / | / // ____// ____/
  / / / // / / /  /  |/ // / / / / /    / /    / /_/ // /| |  /  |/ // / __ / __/   
 / /_/ // /_/ /  / /|  // /_/ / / /    / /___ / __  // ___ | / /|  // /_/ // /___   
/_____/ \____/  /_/ |_/ \____/ /_/     \____//_/ /_//_/  |_|/_/ |_/ \____//_____/   
                                                                                    
  ______ __  __ ______ _____  ______   _    __ ___     __    __  __ ______ _____    
 /_  __// / / // ____// ___/ / ____/  | |  / //   |   / /   / / / // ____// ___/    
  / /  / /_/ // __/   \__ \ / __/     | | / // /| |  / /   / / / // __/   \__ \     
 / /  / __  // /___  ___/ // /___     | |/ // ___ | / /___/ /_/ // /___  ___/ /     
/_/  /_/ /_//_____/ /____//_____/     |___//_/  |_|/_____/\____//_____/ /____/      
                                                                                    
                                                                            
you have been warned.

=#
function _u(u,p,i,j) # Numerator for the F_ij function
    return p.Ω[i,j] * u[j]^p.h
end

function _v(u,p,i,j) # Denominator for the F_ij function
    return p.B0^p.h + p.w * u[i] * p.B0^p.h + sum(p.Ω[i,k] * u[k]^p.h for k in p.prey[i]; init = 0)
end

function _F_ij(u, p, i, j)
    return _u(u,p,i,j)/_v(u,p,i,j)
end

function _∂F_ij(u, p, i, j, k)
    if (i == k) & (j != k)
        return -_F_ij(u, p, i, j)*(p.w * p.B0^p.h - p.h* p.Ω[i,i] * u[i]^(p.h - 1))/_v(u,p,i,j)

    elseif (i != k) & (j == k)
        #return (p.h/u[j])*F_ij(u,p,i,j) * (1 - F_ij(u,p,i,j) - (i == j)*(p.B0 * p.w * u[i])/_v(u,p,i,j))
        return (p.h / u[j]) * _F_ij(u,p,i,j) * (1 - _F_ij(u,p,i,j))
    elseif (i != k) & (j != k)
        #return p.h * F_ij(u,p,i,j) * ((j == k)/u[j] - ((i == k) * p.B0 * p.w * u[i]^(p.h - 1) + sum((z == k) * p.Ω[i,z]*u[z]^(h-1) for z in prey[i]; init = 0)))/_v(u,p,i,j)
        return -(p.h / u[k]) * _F_ij(u,p,i,j) * _F_ij(u,p,i,k) 
    else
        return (p.h/u[i])*_F_ij(u,p,i,j)*(1 + _F_ij(u,p,i,j)) - (_F_ij(u,p,i,j) * p.w * p.B0 ^ p.h)/_v(u,p,i,j)
    end

end

function _log_∂F_ij(u, p, i, j, k)
    if (i == k) & (j != k)
        return -_F_ij(exp.(u),p,i,j) * (p.B0 * p.w * exp(h*u[i]) - p.h * p.Ω[i,i] * exp(p.h * u[i]))/_v(exp.(u), p, i, j)
    elseif (i != k) & (j == k)
        return p.h * _F_ij(exp.(u),p,i,j) * (1 - _F_ij(exp.(u),p,i,j))
    elseif (i != k) & (j != k)
        return -p.h * _F_ij(exp.(u),p,i,j) * _F_ij(exp.(u),p,i,k)
    else
        return p.h * _F_ij(exp.(u0),p,i,j) - _F_ij(exp.(u), p, i, j)*((p.B0^p.h * p.w * exp.(u[i]) - p.h * p.Ω[i,j] * exp(p.h * u[i]))/_v(exp.(u), p, i, j))
    end
end

function get_F_ij(u,p, rows::Union{Vector{Float64}, Vector{Int64}}, cols::Union{Vector{Float64}, Vector{Int64}})
    F_ij = Matrix{Float64}(undef, p.n, p.n)
    
    for i in rows
        for j in cols
            F_ij[i,j] = _F_ij(u,p,i,j)
        end
    end

    return F_ij
end

