function _v(u,p) # Denominator for the F_ij function
    v = get_tmp(p.dummy_vals[3], u)
    for i in 1:p.n
        sum_val = 0
        for j in p.prey[i]
            sum_val += p.Ω[i,j]*u[j]^p.h
        end
        v[i] = sum_val
    end
    @. v += p.B0 ^ p.h + p.w * p.B0.^p.h * u
    return v
end

function F_ij(g,Ω,ω,h,B0,prey,B,i,j)
    #return (Ω[i,j]*(max(B[j],0)^h))/(1 + ω*max(B[i],0) + sum(Ω[i,k]*(max(B[k],0) ^ h) for k in inneighbors(g,i)))
    return (Ω[i,j]*(B[j][1]^h))/(B0^h + ω*B[i]*(B0^h) + sum(Ω[i,k]*(B[k][1] ^ h) for k in prey[i]; init = 0))
end

function cr_f!(du,u,p,t)

    @unpack g,Ω,e,x,w,h,y,S,B0,n,m, pred, prey, basal, allee_effect, dummy_vals = p
    #M = get_tmp(DiffCache(zeros(n)), u)
    #X = get_tmp(DiffCache(zeros(n)), u)

    M = get_tmp(dummy_vals[1], u)
    X = get_tmp(dummy_vals[2], u)
    #Might be misusing this
    #When I create the caches I should be creating the cache outside the function

    #I'm just allocating a new array right now
    #If I do this instead:


    @inbounds @. M = (u - m)
    @inbounds @. X = x*y

    V = _v(u,p)
    
    @inbounds for i in vertices(g)
        #gives a type instability
        #Type of prey term should match the element type of what we expect
        #That is -- zero(eltype(du))
        prey_term = zero(eltype(du))
        pred_term = zero(eltype(du))

        if basal[i]
            du[i] = M[i]*(1-u[i])*allee(u[i], S, allee_effect)
        else
            du[i] = -x[i]*M[i]
        end

        @inbounds @simd for j in prey[i]
            prey_term += Ω[i,j]*u[j]^h
        end

        prey_term *= X[i]*M[i]*allee(u[i],S,allee_effect)/V[i]

        @inbounds @simd for j in pred[i]
            pred_term += Ω[j,i]*u[j]*X[j]/(e[j,i] * V[j])
        end

        pred_term *= -u[i]^(h-1)*M[i]

        du[i] += prey_term + pred_term
    end
    nothing
end

function log_cr_f!(du, u , p, t)
    # want to do something like 
    # x = p.x
    # @. x = exp(u)
    # this allocates
    e_u0 = get_tmp(p.dummy_vals[4], u)
    @. e_u0 = exp(u)
    cr_f!(du, e_u0, p, t)
    @. du *= exp(-u)
    nothing
end
