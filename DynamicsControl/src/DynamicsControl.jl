module DynamicsControl

using DifferentialEquations, ModelingToolkit
using OrdinaryDiffEq
using DynamicalSystems
using Symbolics
using PyPlot
using LinearAlgebra
#using BenchmarkTools

using JuMP
import Ipopt

function jac_get(sys)
    return calculate_jacobian(sys)
end

function get_system(sys, pars, vars)
    pars
    vars

    M = reshape(vars[length(equations(sys)) + 1 : end], length(equations(sys)),length(equations(sys)))
    println(M)
    jac = jac_get(sys)

    dfM = jac*M


    D = Differential(t)
    new_arr = [equations(sys)[i] for i in 1:length(equations(sys))]

    for i in eachindex(dfM)
        append!(new_arr, [D(vars[2+i]) ~ dfM[i]])
    end

    @named new_sys = ODESystem(new_arr)

end

function system_evolve(sys, u0, par, time_closest,time_integrate, time_inc, target)
    global new_u0 = convert(Vector{Float64},u0)
    global points = Array{Float64}([])
    global closeness = 1
    global iters = 0
    #Solves the ODE
    while closeness >= 1e-2
        prob_closest = ODEProblem(sys, new_u0, time_closest, par)
        global sol_closest = solve(prob_closest, Vern9(lazy = false), saveat = time_inc, reltol = 1e-10, abstol = 1e-10)

        #Distance of closest approach
        dist = sum([(sol_closest[i,:] .- target[i]).^2 for i in length(target)]) #all distances
        index = findall(x -> x==minimum(dist), dist) #index of point at closest distances

        function get_M(target) #Gets the M matrix
            M = zeros(length(target),length(target))

            for i in 1:length(target)^2
                M[Int(ceil(i/length(target))), Int(mod(i,length(target))+1)] = sol_closest[i + length(target), index][1] #Selects M values at the time of closest approach
            end
            return M
        end

        M = get_M(target)

        #Set up the constraint optimization
        model = Model(Ipopt.Optimizer)

        #Variable definition
        JuMP.@variables(model, begin
            dx[1:length(target)] <= 0
        end)

        #Expression that we need to minimize
        expr = JuMP.@NLexpression(model,

            sum((target[i] - (sol_closest[i,index][1] + sum(M[i,j]*dx[j] for j = 1:length(target))))^2 for i = 1:length(target))
            # for i in 1:length(target)^2
            #     (target[i] - (sol[i,index][1] + sum([M[i,]*dx[j] for j in 1:length(target)])))^2
            # end
        )

        #Constraints of the model
        JuMP.@NLconstraint(model,
        0.001^2 <= sum(dx[i]^2 for i = 1:length(target)) <= 0.01^2
        )

        JuMP.@NLconstraint(model,
        dx[1] <= 0)

        JuMP.@NLconstraint(model,
        dx[2] <= 0)
        
        #Objective of the model (minimize = Min)
        JuMP.@NLobjective(model, Min, expr)

        #Returns our δx₀
        optimize!(model)

        for i in 1:length(target)
            new_u0[i] = new_u0[i] + value(dx[i])
        end
        println(new_u0)
        append!(points,new_u0[1:length(target)])

        prob_integrate = ODEProblem(sys, new_u0, time_integrate, par)
        global sol_integrate = solve(prob_integrate, Vern9(lazy = false), saveat = time_inc, reltol = 1e-10, abstol = 1e-10)
        closeness = norm(target - sol_integrate[1:length(target),end])
        iters += 1

        

        if iters > 800
            println("max iterations")
            return sol_integrate, points
        end
    end
    return sol_integrate,points
end

end # module DynamicsControl
