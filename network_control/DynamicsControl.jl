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

# function get_basin(f, init_point, par, x_range, y_range)
#     ds = ContinuousDynamicalSystem(f, init_point, par)

#     mapper = AttractorsViaRecurrences(ds, (x_range, y_range); sparse = false)

#     basins, attractors  = basins_of_attraction(mapper;show_progress = false)

#     return basins, attractors
# end

# function potential_well(dx, x, p, t)
#     dx[1] = x[2]
#     dx[2] = 2*p.γ*x[1]*exp(-p.γ*x[1]^2)*(p.b*x[1]^2 + p.c*x[1]^3 + p.d*x[1]^4) - exp(-p.γ*x[1]^2)*(2*p.b*x[1] + 3*p.c*x[1]^2 + 4*p.d*x[1]^3) - p.η*x[2]
# end

# mutable struct Params
#     γ:: Float64
#     b:: Float64
#     c:: Float64
#     d:: Float64
#     η:: Float64
# end

# function params(γ,b,c,d,η)
#     Params(γ,b,c,d,η)
# end

# PyPlot.close_figs()
# using PyCall
# @pyimport matplotlib.colors as matcolors
# custom_cmap = matcolors.ListedColormap([(1,1,1),(151/255,185/255,254/255),(1,1,153/255)],"A")

# # par = params(1.0,-1.0,-0.1,0.5,0.1)

# ds = ContinuousDynamicalSystem(potential_well, rand(2), par)


# xg=range(-3,3,length=600)
# yg=range(-1,1,length=300)

# mapper = AttractorsViaRecurrences(ds, (xg, yg); sparse = false)

# basins, attractors  = basins_of_attraction(mapper;show_progress = false)

# basins[findall(x -> x == 1, basins)] .= 0
# basins[findall(x -> x == 2, basins)] .= 1

# pcolormesh(xg, yg, basins'; cmap = custom_cmap)

# pars = @parameters t, γ, b, c, d, η
# vars = Symbolics.@variables x₁(t), x₂(t), M₁₁(t), M₂₁(t), M₁₂(t), M₂₂(t)

# D = Differential(t)
# eqs = [D(x₁) ~ x₂,
#     D(x₂) ~ 2*γ*x₁*exp(-γ*x₁^2)*(b*x₁^2 + c*x₁^3 + d*x₁^4) - exp(-γ*x₁^2)*(2*b*x₁ + 3*c*x₁^2 + 4*d*x₁^3) - η*x₂]
# @named sys = ODESystem(eqs)

# new_sys = get_system(sys,pars,vars)

# prob = ODEProblem(new_sys, [-1,0,1.0,0.0,0.0,1.0],[0.0,10.0], [0.5,1.0,0.1,-1.0,-0.1])
# sol = solve(prob, Vern9(lazy = false), saveat = 0.01)
# plot(sol[1,:], sol[2,:], ls = ":", color = "k")

# b = @benchmarkable system_evolve(new_sys, [-1,0,1,0,0,1], [0.5, 1, 0.1, -1, -0.1], [0.0,100.0], 0.01,[-0.73262, 0.0] )

# tune!(b);[-0.73262, 0.0]
# run(b)

# new_sol,points = system_evolve(new_sys, [-1,0,1,0,0,1], [0.5, 1, 0.1, -1, -0.1], [0.0,10.0], [0.0,1000.0], 0.1, [0.79711, 0.0])
# points = reshape(points, (2,Int(length(points)/2)))

# scatter(points[1,:], points[2,:], color = "red", s = 1)

# plot(new_sol[1,:],new_sol[2,:], ls = "-", color = "k")
# xlim(-3,3)
# ylim(-1,1)
# gcf()
#savefig("trajectory2.pdf")




