using Distributed
#addprocs(5)
@everywhere begin
    using ProgressBars
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
    function get_distances(u, t)
        return [sqrt(sum((i .- t).^2)) for i in u]
    end

    n = p.n
    local sol_integrate
    new_u0 = copy(init_vals[1:20])
    big_u0 = copy(init_vals)
    closeness = 1
    times_thru = 0
    points = Matrix{Float64}(undef,n,0)
    prev_dx = zeros(n)
    perturb = Matrix{Float64}(undef,n,0)
    term_status = []
    tc_array = []
    delta = (ones(n)/norm(ones(n)))*1e-2
    num_err = 0

    time_tc = []
    time_vm = []
    time_op = []
    time_cs = []
    for iters in 1:max_iters
        #Seperate finding tc and the variational Matrix
        #What is the times for each of these?
        #println("Get tc")
        tc = @elapsed begin
            @views tc_prob = ODEProblem(new_log_sean_cr!, new_u0, time_closest, p.crp)
            tc_sol = solve(tc_prob, Tsit5(), abstol = 1e-4, reltol = 1e-4, saveat = 0.1)
            smallest_ind = argmin(get_distances(tc_sol.u, target))
            tc = tc_sol.t[smallest_ind]
            #init_prob = ODEProblem(big_f, new_u0, time_closest, p)
            #init_sol = OrdinaryDiffEq.solve(init_prob, Vern9(), abstol = 1e-10, reltol = 1e-10, saveat = 0.1)
        end
        #println("$init_solve seconds")
        append!(time_tc, tc)

        #println("Variational Matrix")
        vm = @elapsed begin
            init_prob = ODEProblem(big_f, big_u0, [0 tc], p)
            init_sol = solve(init_prob, AutoTsit5(Rosenbrock23()), abstol = 1e-4, reltol = 1e-4, save_everystep = false, save_start = false)
        end
        #println("$small_time seconds")
        @views Mt = init_sol.u[1][n+1 : end]
        append!(time_vm, vm)
        Mt = reshape(Mt,n,n)

        model = Model(Ipopt.Optimizer)
        set_silent(model)
        #println("Optimizer")
        op = @elapsed begin
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
        end
        append!(time_op, op)
        #println("$control_alg seconds")

        delta = value.(dx)
        prev_dx = delta
        new_u0 = new_u0 .+ delta
        big_u0[1:n] = big_u0[1:n] .+ delta


        #println("Check Perturbation")
        cs = @elapsed begin
        prob_integrate = ODEProblem(new_log_sean_cr!, new_u0, time_integrate, p.crp)
        sol_integrate = solve(prob_integrate, AutoTsit5(Rosenbrock23()), abstol = 1e-4, reltol = 1e-4, save_everystep = false, save_start = false)
        #Tsit5() or KenCarp47()? Depends on what we need, and how accurate it is.
        end
        append!(time_cs, cs)

        #println("$check_perturb seconds")
        #u = mapreduce(permutedims,vcat, sol_integrate.u)

        points::Matrix{Float64} = hcat(new_u0[1:n],points)
        perturb::Matrix{Float64} = hcat(delta, perturb)
        
        #println(termination_status(model))

        term_status = vcat(term_status, termination_status(model))

        if termination_status(model) == NUMERICAL_ERROR
            num_err += 1
        end

        times_thru += 1

        #success_vec = [(u[k,invader] < -20) && all(x -> x > -20, u[k, Not(invader)]) for k in 1:length(sol_integrate)]
        if (sol_integrate[end][invader] < -20) && (all(x -> x >= -20, sol_integrate[end][Not(invader)]))#any(success_vec)
            return time_tc, time_vm, time_op, time_cs#sol_integrate[end], perturb, term_status, tc_array, points, "Success", num_err, times_thru
        end
    end
    #println("max iters")
    return time_tc, time_vm, time_op, time_cs#sol_integrate[end], perturb, term_status, tc_array, points, "MaxIters", num_err, times_thru
end

@everywhere function get_init_sys!(ret::Vector{Float64}, func::T, p::CRParams{Float64}) where T
    u0 = log.(0.5*ones(p.n))
    prob = ODEProblem{true}(func, u0, [0 5000], p)
    sol = solve(prob, Vern9(), abstol = 1e-10, reltol = 1e-10, save_everystep = false)
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
    time_tc = []
    time_vm = []
    time_op = []
    time_cs = []
    # num_err_dict = Dict()
    func = ODEFunction{true}(new_log_sean_cr!; jac = new_log_sean_jac!)
    for file in ProgressBar(files)
        f = jldopen(file)
        g::SimpleDiGraph{Int64} = f["g"]
        p = f["p"]
        stable = f["stable"][1:p.n]
        nodes = f["target_ind"]
        targets = [i[1:20] for i in f["target_states"]]
        close(f)

        init_M = Matrix(1.0I, p.n, p.n)
        init_M = reshape(init_M, p.n^2, 1)
        init_vals = vcat(stable, init_M)
        model_params = ModelParams(func,zeros(p.n,p.n),p.n, p)

        @sync @distributed for ind in eachindex(nodes)
            init_goal = deepcopy(stable[1:p.n])
            target = targets[ind]
            node = nodes[ind]
            #best_sol, perturb::Matrix{Float64}, term_status::Vector{Any}, tc_array::Vector{Any}, init_cons::Matrix{Float64}, end_state::String, num_err::Int64, iters::Int64 = system_evolve(func, init_vals, model_params, [0 10], [0 5000], 0.1, target[1:p.n], node, max_iters)    
            try
                tc, vm, op, cs = system_evolve(func, init_vals, model_params, [0 10], [0 5000], 0.1, target[1:p.n], node, max_iters)
                append!(time_tc, tc)
                append!(time_vm, vm)
                append!(time_op, op)
                append!(time_cs, cs)
            catch e
                if e == TaskFailedException
                    continue
                end
            end
            # num_err_dict[string(replace(file,string(dir,"/")=>"")[1:end-5],"_$species","_num_err")] = num_err
            # num_err_dict[string(replace(file,string(dir,"/")=>"")[1:end-5],"_$species","_percent_err")] = (num_err/iters)*100
            #if end_state == "Success"
                #jldsave(string("$succ_save_dir/",replace(file,string(dir,"/")=>"")[1:end-5],"_$species",".jld2"),best_sol = best_sol, perturb = perturb, term_status = term_status, tc_array = tc_array, init_cons = init_cons, end_state = end_state)
            #else
                #jldsave(string("$fail_save_dir/",replace(file,string(dir,"/")=>"")[1:end-5],"_$species",".jld2"),best_sol = best_sol, perturb = perturb, term_status = term_status, tc_array = tc_array, init_cons = init_cons, end_state = end_state)
            #end
        end
    end
    
    return time_tc, time_vm, time_op, time_cs
end
