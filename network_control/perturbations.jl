function big_f(du, u, p, t)
    func = p[1]
    J = p[2]
    n = p[3]

    @views func.f(du[1:n], u[1:n], nothing, t)
    @views func.jac(J, u[1:n], nothing, t)
    @views M = reshape(u[n+1 : end], n, n)
    @views dM = reshape(du[n+1 : end], n, n)
    @views mul!(dM, J, M)

    @views du[n+1 : end] = reshape(dM, n, n)

end

function optimization(func::ODEFunction, ic, target_state, target_node::Int64, opt::OptimizationParams, dyn::DynamicsParams, basal, n)
    function get_distances(u,t)
        return [norm(i .- t) for i in u]
    end

    local sol

    historical_delta  = []
    historical_ic = []
    historical_sol = []

    J = zeros(n,n)

    func_u0 = ic[1:n]
    big_u0 = deepcopy(ic)
    delta = (ones(n)/norm(ones(n))) * 1e-2 #opt.ϵ_min

    for iter in 1:opt.iter_limit
        
        time_closest_prob = ODEProblem(func, func_u0, opt.closest_time)
        time_closest_sol = solve(time_closest_prob, Vern9(), abstol = dyn.dyn_abstol,
                                    reltol = dyn.dyn_reltol, saveat = opt.time_increment)

        smallest_ind = argmin(get_distances(time_closest_sol.u, target_state))
        tc = time_closest_sol.t[smallest_ind]


        big_prob = ODEProblem(big_f, big_u0, [0 tc], [func, J, n])
        big_sol = solve(big_prob, Vern9(), abstol = dyn.dyn_abstol, reltol = dyn.dyn_reltol,
                        save_everystep = false, save_start = false)

        @views M_tc = big_sol.u[1][n+1 : end]
        M_tc = reshape(M_tc, n, n)
        model = Model(Ipopt.Optimizer)
        set_silent(model)

        JuMP.@variable(model,
            dx[i = 1:n],
            start = delta[i]
        )

        expr = JuMP.@expression(model,
            (target_state[Int(target_node)] .- (big_sol.u[1][target_node] .+ (M_tc*dx)[target_node]))^2
        )

        JuMP.@constraint(model,
            dx[target_node] == 0
        )

        JuMP.@constraint(model,
            dx .+ func_u0[1:n] .<= ic[1:n]
        )

        JuMP.@constraint(model,
            dx[findall(x -> x == 1, basal)] .+ func_u0[1:n][findall(x -> x == 1, basal)] .>= opt.basal_lb
        )

        JuMP.@constraint(model,
            dx[Not(findall(x -> x == 1, basal))] .+ func_u0[1:n][Not(findall(x -> x == 1, basal))] .>= opt.nonbasal_lb
        )

        JuMP.@constraint(model,
            dot(dx, delta) >= 0
        )

        JuMP.@NLconstraint(model,
            opt.ϵ_min <= sqrt(sum(dx[j]^2 for j = 1:n)) <= opt.ϵ_max
        )

        JuMP.@objective(model, Min, expr)
        set_attribute(model, "hsllib", HSL_jll.libhsl_path)
        set_attribute(model, "linear_solver", "ma97")
        optimize!(model)

        push!(historical_delta, delta)
        push!(historical_ic, func_u0)

        delta = value.(dx)

        func_u0 = func_u0 .+ delta
        @views big_u0[1:n] = big_u0[1:n] .+ delta

        prob = ODEProblem(func, func_u0, opt.integrate_time)
        sol = solve(prob, Vern9(), abstol = dyn.dyn_abstol, reltol = dyn.dyn_reltol)
        push!(historical_sol, sol.u[end])

        if (sol[end][Int(target_node)] < opt.extinction) & (all(x -> x > opt.extinction, sol[end][Not(Int(target_node))]))
            return sol.u[end], "Success", historical_delta, historical_ic, historical_sol
        end
    end

    num_extinct = length(findall(x -> x > opt.extinction, sol.u[end]))

    if num_extinct < n - 1
        if sol[end][Int(target_node)] < opt.extinction
            retcode = "MultiExtinct_WithTarget"
        else
            retcode = "MultiExtinct_NotTarget"
        end
    elseif num_extinct == n - 1
        retcode = "OneExtinct_NotTarget"
    else
        retcode = "NoExtinctions"
    end

    return sol.u[end], retcode, historical_delta, historical_ic, historical_sol
end


function perturb_function(file_loc::String, save_loc::String)

    files = glob(file_loc*"/*.jld2")

    if !isdir(save_loc)
        mkpath(save_loc)
    end

    Threads.@threads for file in files
        f = jldopen(file, "r")
        println(file)
        
        up = f["network_params"].up; crp = f["network_params"].crp; stable = f["network_params"].stable;
        n = length(stable)
        local func = fmtk_cr(crp; log_space = up.log_space)
        local ref_ic = vcat(stable, reshape(Matrix(1.0I, n, n), n^2, 1))
        
        for (target_state, target_node) in zip(up.opt.targets, up.opt.invaders)
            #target_state = up.opt.targets[ind]
            #target_node = Int(up.opt.invaders[ind])
            ic = deepcopy(ref_ic)
            sol, retcode, historical_delta, historical_ic, historical_sol = optimization(func, ic, target_state, Int(target_node), up.opt, up.dyn, crp[end-1], n)
            jldsave(save_loc*split(file,file_loc)[end][1:end-5]*"_$(Int(target_node)).jld2"; end_state = sol, retcode = retcode, deltas = historical_delta, ics = historical_ic, sols = historical_sol, up = up)
        end

        drop_expr(func.f)
        drop_expr(func.jac)
        GC.gc() #REDUCE MEMORY SIZE PLS
    end
    println("$file_loc complete.")

    nothing
end

function perturb_all(folders_starter, saveloc_starter)

    folders = glob(folders_starter*"-B0_*-S_*")

    for folder in folders
        println(folder)
        perturb_function(folder, saveloc_starter*split(folder, folders_starter)[end])
    end

    nothing

end