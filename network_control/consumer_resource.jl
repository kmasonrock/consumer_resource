using OrdinaryDiffEq


function cr_f!(du,u,p,t)
    @unpack g,Ω,e,x,w,h,y,S, B0,n,m, dummy,pred, prey, basal, allee_effect = p
    m_term = dummy
    prey_term = dummy
    pred_term = dummy
    @. m_term = (u - m)
    for i in 1:nv(g)
        if basal[i]
            du[i] = (1 - u[i])*(u[i] - m)#*allee(u[i],S,allee_effect)
        else
            du[i] = -x[i]*(u[i] - m)
        end

        prey_term = zero(eltype(du))
        for j in prey[i]
            prey_term += _F_ij(u,p,i,j)
        end
        prey_term *= x[i]*y*(u[i] - m)*allee(u[i],S,allee_effect)

        for j in pred[i]
            pred_term[i] -= x[j]*y*((u[i] - m)/u[i])*u[j]*_F_ij(u,p,j,i)/e[j,i]
        end
    end

    du .= prey_term .+ pred_term
    nothing
end

function cr_jac!(J,u,p,t)
    @unpack g,Ω,e,x,w,h,y,S,B0,n,m,pred,prey,basal,allee_effect = p

    n = Int(n)

    for i in eachindex(1:nv(g))
        for j in eachindex(1:nv(g))
            J[i,j] = 0
            if basal[i]
                if j == i
                    #(1 - u[i])*(u[i] - m)*∂allee(u[i],S,allee_effect) + (1 - u[i])*allee(u[i],S,allee_effect) - (u[i] - m)*allee(u[i],S,allee_effect)
                    J[i,j] = (1 - u[i]) - (u[i] - m)
                    for η in pred[i]
                        J[i,j] -= x[η]*y*u[η]/(u[i]*e[η,i]) * ((u[i] - m)*(_∂F_ij(u,p,η,i,j) - _F_ij(u,p,η,i)/u[i]) + _F_ij(u,p,η,i))
                    end
                else
                    for η in pred[i]
                        J[i,j] -= (y*x[η]*(u[i] - m)/(u[i]*e[η,i])) * (_∂F_ij(u,p,η,i,j)*u[η] + (j == η)*_F_ij(u,p,η,i))
                    end
                end
            else
                if j == i
                    J[i,j] = -x[i]
                    for η in prey[i]
                        J[i,j] += y*x[i]*((u[i] - m)*(_∂F_ij(u,p,i,η,j)*allee(u[i],S,allee_effect) + _F_ij(u,p,i,η)*∂allee(u[i],S,allee_effect)) + _F_ij(u,p,i,η)*allee(u[i],S,allee_effect))
                    end
                    for η in pred[i]
                        J[i,j] -=  (x[η]*y/(u[i]*e[η,i]))*(u[η]*((u[i] - m)*(_∂F_ij(u,p,η,i,j) - _F_ij(u,p,η,i)/u[i]) + _F_ij(u,p,η,i)) + (j == η)*(u[i] - m)*_F_ij(u,p,η,i))
                    end

                    #@views J[i,j] = -x[i] + x[i]*y*sum(F_ij(g,Ω,w,h,u[1:n],i,η) + u[i]*∂F_ij(g,Ω,w,h,n,u[1:n],i,η,j) for η in inneighbors(g,i); init = 0) - y*sum((x[η]/e[η,i]) *((η == j)*F_ij(g,Ω,w,h,u[1:n],η,i) + u[η] * ∂F_ij(g,Ω,w,h,B0,n,u[1:n],η,i,j)) for η in outneighbors(g,i); init = 0)
                    #@views J[i,j] = -x[i] + x[i]*y*sum(F_ij(g,Ω,w,h,u[1:n],i,η)*(u[i]/(u[i] + S)) + u[i]*∂F_ij(g,Ω,w,h,n,u[1:n],i,η,j)*(u[i]/(S + u[i])) + u[i]*F_ij(g,Ω,w,h,u[1:n],i,η)*(S/(S + u[i])^2) for η in inneighbors(g,i); init = 0) - y*sum((x[η]/e[η,i]) *((η == j)*F_ij(g,Ω,w,h,u[1:n],η,i) + u[η] * ∂F_ij(g,Ω,w,h,B0,n,u[1:n],η,i,j)) for η in outneighbors(g,i); init = 0)
                    
                else
                    for η in prey[i]
                        J[i,j] +=  y*x[i]*((u[i] - m)*_∂F_ij(u,p,i,η,j)*allee(u[i],S,allee_effect) + (i == η)*((u[i] - m)*_F_ij(u,p,i,η)*∂allee(u[i],S,allee_effect) + _F_ij(u,p,i,η)*allee(u[i],S,allee_effect)))
                    end
                    for η in pred[i]
                        J[i,j] -= (x[η]*y/(u[i]*e[η,i]))*(u[η]*(_∂F_ij(u,p,η,i,j)*(u[i] - m)) + (i == η)*(_F_ij(u,p,η,i)*u[η] - (u[i] - m)*_F_ij(u,p,η,i)/u[i]) + (j == η)*(u[i] - m)*_F_ij(u,p,η,i))
                    end
                    #@views J[i,j] = sum(x[i]*y*u[i]*∂F_ij(g,Ω,w,h,n,u[1:n],i,η,j)*(u[i]/(S + u[i])) for η in inneighbors(g,i); init = 0) - sum((x[η]*y/e[η,i])*((η == j)*F_ij(g,Ω,w,h,u[1:n],η,i) + u[η]*∂F_ij(g,Ω,w,h,B0,n,u[1:n],η,i,j)) for η in outneighbors(g,i); init = 0)
                end
            end
        end
    end

    nothing
end

function log_cr_f!(du, u , p, t)
    # want to do something like 
    # x = p.x
    # @. x = exp(u)
    # this allocates
    cr_f!(du, exp.(u), p, t)
    @. du *= exp(-u)
    nothing
end

function log_cr_jac!(J, u, p, t)
    # the exp.(u) allocates
    cr_jac!(J, exp.(u), p, t)
    for i in 1:p.n
        for j in 1:p.n
            J[i,j] = exp(-u[i]) * exp(u[j]) * J[i,j]
        end
    end
end

