using Distributed
addprocs(4)
@everywhere begin
    using Graphs, MetaGraphs
    using JLD2, Glob
    using Infiltrator
    using UnPack
    using JuMP
    import Ipopt, HSL_jll
    using OrdinaryDiffEq
    using ModelingToolkit
    using InvertedIndices
    using Symbolics
    using LinearAlgebra
    using Static
    include("cr_inline.jl")
    include("ecosystem_model_functions.jl")
end

@everywhere struct ModelParams
    func::ODEFunction
    J::Matrix{Float64}
    n::Int64
    crp::CRParams
end

@everywhere function big_f(du,u,p,t)
    @unpack func, J, n, crp = p
    @views func.f(du[1:n], u[1:n], crp, t)
    @views func.jac(J, u[1:n], crp, t)
    @views M = reshape(u[n+1 : end], n, n)
    @views dM = reshape(du[n+1 : end], n, n)
    @views mul!(dM,J,M)

    @views du[n+1 : end] = reshape(dM,n,n)

    nothing
end

@everywhere function system_evolve(func::ODEFunction{true}, init_vals::Matrix{Float64}, p::ModelParams, time_closest::Matrix{Int64},time_integrate::Matrix{Int64}, time_inc::Float64, target::Vector{Float64}, invader::Int64, max_iters::Int64)
    
    
    n = p.n
    local sol_integrate
    new_u0 = copy(init_vals)
    closeness = 1
    times_thru = 0
    points = Matrix{Float64}(undef,n,0)
    prev_dx = zeros(n)
    perturb = Matrix{Float64}(undef,n,0)
    term_status = []
    tc_array = []
    delta = (ones(n)/norm(ones(n)))*1e-2
    num_err = 0
    for iters in 1:max_iters
        init_prob = ODEProblem(big_f,new_u0, time_closest, p)
        init_sol = OrdinaryDiffEq.solve(init_prob, Vern9(), abstol = 1e-10, reltol = 1e-10, saveat = 0.1)

        dist = [sum((init_sol[i][1:n] .- target).^2, dims = 1) for i in eachindex(init_sol)]
        smallest_ind = findall(x -> x == minimum(dist), dist)[1]
        
        Mt = init_sol[smallest_ind][n+1 : end]
        append!(tc_array, smallest_ind*time_inc)
        Mt = reshape(Mt,n,n)

        model = Model(Ipopt.Optimizer)
        JuMP.@variable(model,
            dx[i=1:n],
            start = delta[i]
            )

        expr = JuMP.@expression(model,
            #sum((target .- (init_sol[smallest_ind][1:n] .+ Mt*dx)).^2) #make objective more greedy
            (target[invader] .- (init_sol[smallest_ind][invader] .+ (Mt*dx)[invader]))^2
            #we want the closest approach time value
            )

        JuMP.@constraint(model,
            dx[invader] == 0
            )

        JuMP.@constraint(model,
            dx .+ new_u0[1:n] .<= init_vals[1:n]
            )

        JuMP.@constraint(model,
            dx .+ new_u0[1:n] .>=  -20#last sqrt with 7 successes was at 1e-8
            )

        JuMP.@constraint(model,
            dot(dx,prev_dx) >= 0
        )

        JuMP.@NLconstraint(model,
         0.001 <= sqrt(sum(dx[j]^2 for j = 1:n)) <= 0.1
        )

        JuMP.@objective(model, Min, expr)
        set_attribute(model, "hsllib", HSL_jll.libhsl_path)
        #Returns our δx₀
        set_attribute(model, "linear_solver", "ma97")
        optimize!(model)

        delta = value.(dx)
        prev_dx = delta
        new_u0[1:n] = new_u0[1:n] .+ delta

        prob_integrate = ODEProblem{true}(new_log_sean_cr!, new_u0[1:n], time_integrate, p.crp)
        sol_integrate = OrdinaryDiffEq.solve(prob_integrate, Vern9(), abstol = 1e-10, reltol = 1e-10, save_everystep = false)

        #u = mapreduce(permutedims,vcat, sol_integrate.u)

        points::Matrix{Float64} = hcat(new_u0[1:n],points)
        perturb::Matrix{Float64} = hcat(delta, perturb)
        
        println(termination_status(model))

        term_status = vcat(term_status, termination_status(model))

        if termination_status(model) == NUMERICAL_ERROR
            num_err += 1
        end

        times_thru += 1

        #success_vec = [(u[k,invader] < -20) && all(x -> x > -20, u[k, Not(invader)]) for k in 1:length(sol_integrate)]
        if (sol_integrate[end][invader] < -20) && (all(x -> x >= -20, sol_integrate[end][Not(invader)]))#any(success_vec)
            return sol_integrate[end], perturb, term_status, tc_array, points, "Success", num_err, times_thru
        end
    end
    println("max iters")
    return sol_integrate[end], perturb, term_status, tc_array, points, "MaxIters", num_err, times_thru
end

@everywhere function get_init_sys!(ret::Vector{Float64}, func::T, p::CRParams{Float64}) where T
    u0 = log.(0.5*ones(p.n))
    prob = ODEProblem{true}(func, u0, [0 5000], p)
    sol = solve(prob, Vern7(), abstol = 1e-10, reltol = 1e-10, save_everystep = false)
    ret .= sol[end]
    nothing
end


function perturb_dynamics(dir, max_iters, succ_save_dir, fail_save_dir, allee_type::Union{Type{Absent},Type{Hill},Type{Strong}})
    files = glob(string(dir, "/*.jld2"))

    Infiltrator.clear_disabled!()
    #st = sol_type()
    if !isdir(succ_save_dir)
        mkdir(succ_save_dir)
    end

    if !isdir(fail_save_dir)
        mkdir(fail_save_dir)
    end
    # num_err_dict = Dict()
    func = ODEFunction{true}(new_log_sean_cr!; jac = new_log_sean_jac!)
    for file in files
        f = jldopen(file)
        g::SimpleDiGraph{Int64} = f["g"]
        stable = f["stable"][1:20]
        nodes = f["target_ind"]
        targets = [i[1:20] for i in f["target_states"]]
        close(f)

        p::CRParams{Float64} = cr_params(g, allee_type)
        # init_sys = Vector{Float64}(undef,p.n)
        # invasive = findall(x -> length(x) != 0, p.prey)
        # n = p.n
        # prob = ODEProblem(func, log.(0.5*ones(p.n)), [0 5000], p)
        # sol = solve(prob, solver, abstol = 1e-10, reltol = 1e-10)
        # get_init_sys!(init_sys, func, p)

        init_M = Matrix(1.0I, p.n, p.n)
        init_M = reshape(init_M, p.n^2, 1)
        init_vals = vcat(stable[1:p.n], init_M)
        model_params = ModelParams(func,zeros(p.n,p.n),p.n, p)
        @sync @distributed for ind in eachindex(nodes)
            node = nodes[ind]
            init_goal = deepcopy(stable[1:p.n])
            target = targets[ind]
            best_sol, perturb::Matrix{Float64}, term_status::Vector{Any}, tc_array::Vector{Any}, init_cons::Matrix{Float64}, end_state::String, num_err::Int64, iters::Int64 = system_evolve(func, init_vals, model_params, [0 10], [0 5000], 0.1, target[1:p.n], node, max_iters)
            if end_state == "Success"
                jldsave(string("$succ_save_dir/",replace(file,string(dir,"/")=>"")[1:end-5],"_$node",".jld2"), g = g, best_sol = best_sol, perturb = perturb, term_status = term_status, tc_array = tc_array, init_cons = init_cons, end_state = end_state)
            else
                jldsave(string("$fail_save_dir/",replace(file,string(dir,"/")=>"")[1:end-5],"_$node",".jld2"), g = g, best_sol = best_sol, perturb = perturb, term_status = term_status, tc_array = tc_array, init_cons = init_cons, end_state = end_state)
            end
        end
    end
    #     @sync @distributed for species in invasive
	#     println(species)
	#     println()
    #         sleep(0.1)
    #         init_goal = copy(init_sys)
    #         init_goal[species] = -20
    #         #You must check if the stable state to ensure that it goes to a correct dynamics
    #         target_prob = ODEProblem{true}(func, init_goal, [0 5000], p)
    #         target_sol = OrdinaryDiffEq.solve(target_prob, Vern9(lazy = false), save_everystep = false)
    #         if length(findall(x -> x < -20, target_sol[end])) == 1
    #             target::Vector{Float64} = target_sol[end]
    #             best_sol, perturb::Matrix{Float64}, term_status::Vector{Any}, tc_array::Vector{Any}, init_cons::Matrix{Float64}, end_state::String, num_err::Int64, iters::Int64 = system_evolve(func, init_vals, model_params, [0 10], [0 5000], 0.1, target, species, max_iters)
    #             # num_err_dict[string(replace(file,string(dir,"/")=>"")[1:end-5],"_$species","_num_err")] = num_err
    #             # num_err_dict[string(replace(file,string(dir,"/")=>"")[1:end-5],"_$species","_percent_err")] = (num_err/iters)*100
    #             if end_state == "Success"
    #                 jldsave(string("$succ_save_dir/",replace(file,string(dir,"/")=>"")[1:end-5],"_$species",".jld2"),best_sol = best_sol, perturb = perturb, term_status = term_status, tc_array = tc_array, init_cons = init_cons, end_state = end_state)
    #             else
    #                 jldsave(string("$fail_save_dir/",replace(file,string(dir,"/")=>"")[1:end-5],"_$species",".jld2"),best_sol = best_sol, perturb = perturb, term_status = term_status, tc_array = tc_array, init_cons = init_cons, end_state = end_state)
    #             end
    #         else
    #             println("Incompatible. Continuing...")
    #             continue
    #         end
    #     end
    # end
    
    # return nothing
end