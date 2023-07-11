using DifferentialEquations
using OrdinaryDiffEq
using DynamicalSystems
using PyPlot
using LinearAlgebra

using JuMP
import Ipopt

mutable struct Params
    γ:: Float64
    b:: Float64
    c:: Float64
    d:: Float64
    η:: Float64
end

function potential_well(dx, x, p, t)
    dx[1] = x[2]
    dx[2] = 2*p.γ*x[1]*exp(-p.γ*x[1]^2)*(p.b*x[1]^2 + p.c*x[1]^3 + p.d*x[1]^4) - exp(-p.γ*x[1]^2)*(2*p.b*x[1] + 3*p.c*x[1]^2 + 4*p.d*x[1]^3) - p.η*x[2]
end

function big_f(du, u, p, t)
    du[1] = u[2]
    du[2] = 2*p.γ*u[1]*exp(-p.γ*u[1]^2)*(p.b*u[1]^2 + p.c*u[1]^3 + p.d*u[1]^4) - exp(-p.γ*u[1]^2)*(2*p.b*u[1] + 3*p.c*u[1]^2 + 4*p.d*u[1]^3) - p.η*u[2]
    du[3] = u[5]
    du[4] = u[6]
    du[5] = ((+)((+)((+)((+)((*)((+)((+)((*)(-2, p.b), (*)((*)(-12, p.d), (^)(u[1], 2))), (*)((*)(-6, p.c), u[1])), (exp)((*)((*)(-1, p.γ), (^)(u[1], 2)))), (*)((*)((*)(2, p.γ), (+)((+)((*)(p.b, (^)(u[1], 2)), (*)(p.c, (^)(u[1], 3))), (*)(p.d, (^)(u[1], 4)))), (exp)((*)((*)(-1, p.γ), (^)(u[1], 2))))), (*)((*)((*)((*)(-4, (+)((+)((*)(p.b, (^)(u[1], 2)), (*)(p.c, (^)(u[1], 3))), (*)(p.d, (^)(u[1], 4)))), (^)(p.γ, 2)), (^)(u[1], 2)), (exp)((*)((*)(-1, p.γ), (^)(u[1], 2))))), (*)((*)((*)((*)(-2, p.γ), (+)((+)((*)((*)(-3, p.c), (^)(u[1], 2)), (*)((*)(-4, p.d), (^)(u[1], 3))), (*)((*)(-2, p.b), u[1]))), u[1]), (exp)((*)((*)(-1, p.γ), (^)(u[1], 2))))), (*)((*)((*)((*)(2, p.γ), (+)((+)((*)((*)(3, p.c), (^)(u[1], 2)), (*)((*)(4, p.d), (^)(u[1], 3))), (*)((*)(2, p.b), u[1]))), u[1]), (exp)((*)((*)(-1, p.γ), (^)(u[1], 2))))))*u[3] - p.η*u[5]
    du[6] = ((+)((+)((+)((+)((*)((+)((+)((*)(-2, p.b), (*)((*)(-12, p.d), (^)(u[1], 2))), (*)((*)(-6, p.c), u[1])), (exp)((*)((*)(-1, p.γ), (^)(u[1], 2)))), (*)((*)((*)(2, p.γ), (+)((+)((*)(p.b, (^)(u[1], 2)), (*)(p.c, (^)(u[1], 3))), (*)(p.d, (^)(u[1], 4)))), (exp)((*)((*)(-1, p.γ), (^)(u[1], 2))))), (*)((*)((*)((*)(-4, (+)((+)((*)(p.b, (^)(u[1], 2)), (*)(p.c, (^)(u[1], 3))), (*)(p.d, (^)(u[1], 4)))), (^)(p.γ, 2)), (^)(u[1], 2)), (exp)((*)((*)(-1, p.γ), (^)(u[1], 2))))), (*)((*)((*)((*)(-2, p.γ), (+)((+)((*)((*)(-3, p.c), (^)(u[1], 2)), (*)((*)(-4, p.d), (^)(u[1], 3))), (*)((*)(-2, p.b), u[1]))), u[1]), (exp)((*)((*)(-1, p.γ), (^)(u[1], 2))))), (*)((*)((*)((*)(2, p.γ), (+)((+)((*)((*)(3, p.c), (^)(u[1], 2)), (*)((*)(4, p.d), (^)(u[1], 3))), (*)((*)(2, p.b), u[1]))), u[1]), (exp)((*)((*)(-1, p.γ), (^)(u[1], 2))))))*u[4] - p.η*u[6]
end

#(2*p.γ*exp(-p.γ*u[1])*(p.b*u[1]^2 + p.c*u[1]^3 + p.d*u[1]^4) - 4*p.γ*(u[1]^2)*exp(-p.γ*u[1]^2)*(p.b*u[1]^2 + p.c*u[1]^3 + p.d*u[1]^4) + 2*p.γ*exp(-γ*u[1]^2)*(2*p.b*u[1] + 3*p.c*u[1]^2 + 4*p.d*u[1]^3) + 2*p.γ*exp(-p.γ*u[1]^2)*(2*p.b*u[1] + 3*p.c*u[1]^2 + 4*p.d*u[1]^3) - exp(-p.γ*u[1]^2)*(2*p.b + 6*p.c*u[1] + 12*p.d*u[1]))*u[3] 
#(2*p.γ*exp(-p.γ*u[1])*(p.b*u[1]^2 + p.c*u[1]^3 + p.d*u[1]^4) - 4*p.γ*(u[1]^2)*exp(-p.γ*u[1]^2)*(p.b*u[1]^2 + p.c*u[1]^3 + p.d*u[1]^4) + 2*p.γ*exp(-γ*u[1]^2)*(2*p.b*u[1] + 3*p.c*u[1]^2 + 4*p.d*u[1]^3) + 2*p.γ*exp(-p.γ*u[1]^2)*(2*p.b*u[1] + 3*p.c*u[1]^2 + 4*p.d*u[1]^3) - exp(-p.γ*u[1]^2)*(2*p.b + 6*p.c*u[1] + 12*p.d*u[1]))*u[4]

function params(γ,b,c,d,η)
    Params(γ,b,c,d,η)
end
PyPlot.close_figs()


par = params(1.0,-1.0,-0.1,0.5,0.1)

function system_evolve(big_f, par)
    u0 = [2.5, -0.15]

    for i in 1:10000
        target_a = [0.79711, 0.0]
        target_b = [-0.73262, 0.0]

        #ds = ContinuousDynamicalSystem(potential_well, rand(2), par)

        prob = ODEProblem(big_f, [u0[1],u0[2],1,0,0,1], [0.0,100], par)
        global sol = solve(prob, Vern9(lazy = false), saveat = 0.1, abstol = 1e-10, reltol = 1e-10)

        println(sol[:,1])

        dist = (sol[1,:] .- target_b[1]).^2 + (sol[2,:] .- target_b[2]).^2
        index = findall(x->x==minimum(dist),dist)

        time_val = sol[index].t

        M_c = zeros(2,2)

        M_init = zeros(2,2)

        M_c[1,:] = [sol[3,index][1], sol[4,index][1]]
        M_c[2,:] = [sol[5,index][1], sol[6,index][1]]

        print(M_c)

        M_init[1,:] = [sol[3,1][1], sol[4,1][1]]
        M_init[2,:] = [sol[5,1][1], sol[6,1][1]]


        xy = target_b.-sol[1:2,index]

        θ = atan(xy[2]/xy[1])

        δx₂ =  0.01*sin(θ)
        δx₁ = 0.01*cos(θ)

        δxₜ = [δx₁; δx₂]

        δx₀ = inv(M_c)δxₜ

        model = Model(Ipopt.Optimizer)



        @variables(model, begin
            dx[1:2] <= 0
        end
        )

        expr = @NLexpression(model,
            #(target_b[1]^2 + target_b[2]^2) - (sol[1,index][1] + M_c[1,1][1]*dx[1] + M_c[1,2][1]*dx[2])^2 +  (sol[2,index][1] + M_c[2,1][1]*dx[1] + M_c[2,2][1]*dx[2])^2
            (target_b[1] - (sol[1,index][1] + M_c[1,1][1]*dx[1] + M_c[1,2]*dx[2]))^2 + (target_b[2] - (sol[2,index][1] + M_c[2,1]*dx[1] + M_c[2,2]*dx[2]))^2
            )

        @NLconstraint(
            model,
            0.001^2 <= dx[1]^2 + dx[2]^2 <= 0.01^2
        )

        @NLconstraint(
            model,
            dx[1] <= 0
        )

        @NLconstraint(
            model,
            dx[2] <= 0
        )

        @NLobjective(model, Min, expr)



        #@NLobjective(model, Min, expr[1])
        #@NLobjective(model, Min, expr[2])



        optimize!(model)
        # println("""
        # termination_status = $(termination_status(model))
        # primal_status      = $(primal_status(model))
        # objective_value    = $(objective_value(model))
        # """)
        # return

        println(value(dx[1]))
        println(value(dx[2]))

        if i == 1
            plot(sol[1,:], sol[2,:], color = "yellow")
        end

        scatter(sol[1,1][1] + value(dx[1]),sol[2,1][1] + value(dx[2]), s = 0.1, color = "blue")
        scatter(target_b[1], target_b[2])

        if abs((sol[1,end] - target_b[1])/target_b[1]) <= 1e-2
            break
        else
            u0 = [u0[1] + value(dx[1]), u0[2] + value(dx[2])]
        end

    end
    plot(sol[1,:], sol[2,:], ls = "--", color = "black")
    xlim(-3,3)
    ylim(-1,1)
    gcf()
end


ds = ContinuousDynamicalSystem(potential_well, rand(2), par)


xg=range(-3,3,length=600)
yg=range(-1,1,length=300)

mapper = AttractorsViaRecurrences(ds, (xg, yg); sparse = false)

basins, attractors  = basins_of_attraction(mapper;show_progress = false)
basins

init_cons = [[-3,0.6], [-3,0.95],[-3,0.5],[3,-0.7], [3, -0.925]]

pcolormesh(xg, yg, basins'; cmap = "Accent")

system_evolve(big_f,par)


