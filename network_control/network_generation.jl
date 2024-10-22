include("initialize.jl")

function generate_node_params()
    η = rand(Uniform(0,1))
    r = rand(Beta(1,1.5))*η
    centre = rand(Uniform(r/2, η))
    c = (centre - r/2, centre + r/2)
    return η, r, c
end

function test_du(func, vals, du, p, thresh)
    func(du,vals,p,nothing)
    return all(x -> abs(x) < thresh, du)
end

function test_eig(func, vals, du, p)
    #TODO Why is ForwardDiff returning NaN or Inf when we use the log versions?
    J = ForwardDiff.jacobian((_du, _u) -> func(_du, _u, p, nothing), du, vals)
    return maximum(real(eigvals(J))) < 0 ? true : false
end

function max_basal(g, basal_limit, n_nodes)
    try
        tl = get_trophic(g)
        if (sum(tl .<= 1)/n_nodes < basal_limit)
            return false
        else
            return true
        end
    catch e
        println(e)
        println("You made an invalid graph!")
        jldsave("LinAlgFails/graph_$(length(readdir("LinAlgFails")) + 1)"; g)
        return true
    end
end


function test_dynamics(g, func, u0, dyn_p, net_gen_p; testing_targets = false)
    @unpack e_herb, e_carn, y, dxdr, Z, m_trophic, w, B0, S, h, m_migrate, dyn_abstol, dyn_reltol, allee_type = dyn_p
    @unpack β, ex_thresh, fp_thresh, abstol, reltol, integrate_time, starting_B, basal_limit, iter_limit = net_gen_p

    local p

    init_conditions = deepcopy(u0)
    if !testing_targets
        append!(init_conditions, starting_B)
    end
    
    du = zeros(length(init_conditions))
    J = zeros(length(init_conditions), length(init_conditions))

    try
        p = cr_params(g, dyn_p, allee_type)
        prob = ODEProblem(func, init_conditions, integrate_time, p)
        sol = solve(prob, Vern9(), save_everystep = false, abstol = fp_thresh, reltol = fp_thresh, callback = TerminateSteadyState(dyn_reltol, dyn_abstol), verbose = false)
        
        if sol.t[end] == integrate_time[end]
            return false, u0, nothing
        end

        if all(x -> x > ex_thresh, sol[end]) & (all(x -> isapprox(starting_B, x ) == false, sol[end])) & test_eig(func, sol[end], du, p)
            return true, sol[end], p
        elseif testing_targets
            if (count(x -> x > ex_thresh, sol[end]) == (nv(g) - 1)) & test_eig(func, sol[end], du, p)
                return true, sol[end], p
            else
                return false, u0, p
            end
        else
            return false, u0, p
        end

    catch e
        return false, u0, cr_params(SimpleDiGraph(), dyn_p, allee_type)
    end
end

function initiate_niche(starting_B)
    η = rand(Uniform(0,1))
    r = 0.0
    centre = rand(Uniform(r/2, η))
    c = (centre,centre)
    g = SimpleDiGraph(1)
    return [starting_B], [η], [r], [c], g
end

function get_resources(η_vec, c_vec)
    return findall(x -> (c_vec[end][1] <= x) & (c_vec[end][2] >= x), η_vec)
end

function get_consumers(η_vec, c_vec)
    return findall(x -> (x[1] <= η_vec[end]) & (x[2] >= η_vec[end]), c_vec)
end

function has_cycles(g)
    return isempty(simplecycles(g)) ? false : length(argmax(length,simplecycles(g))) > 2
end

function add_species(u0, η0, r0, c0, g)
    η, r, c = generate_node_params()
end

function get_targets(g, func, u0, dyn_p, net_gen_p)
    target_nodes = []
    target_states = Vector{Vector{Float64}}(undef, 0)
    for i in vertices(g)
        init_vals = deepcopy(u0)
        init_vals[i] = net_gen_p.ex_thresh
        works, ts, _ = test_dynamics(g, func, init_vals, dyn_p, net_gen_p; testing_targets = true)
        if works
            if (count(x -> x > net_gen_p.ex_thresh, ts) == (nv(g) - 1)) & (ts[i] < net_gen_p.ex_thresh)
                append!(target_nodes, i)
                push!(target_states, ts)
            end
        end
    end
    return target_nodes, target_states
end

function network_generation(n_nodes, func, universal_params; saveloc = nothing)

    local net_gen, dyn, p
    
    try
        net_gen = universal_params.net_gen
        dyn = universal_params.dyn
        opt = universal_params.opt
    catch
        throw("NetworkGeneration Parameters are not in the universal_params variable! Please make sure that you're providing the universal parameters, and ensure that you generate them using the initialize() function.")
    end

    u0, η_vec, r_vec, c_vec, g = initiate_niche(net_gen.init_B)

    iterations = 1

    while (nv(g) < n_nodes)
        if iterations >= net_gen.iter_limit
            u0, η_vec, r_vec, c_vec, g = initiate_niche(net_gen.init_B)
            iterations = 1
        end

        add_vertex!(g)
        η, r, c = generate_node_params()

        for (i,_n) in enumerate(η_vec)
            if (c[1] <= _n) & (c[2] >= _n)
                add_edge!(g, i, nv(g))
            end
        end

        for (i,v) in enumerate(c_vec)
            if (v[1] <= η) & (v[2] >= η)
                add_edge!(g,nv(g), i)
            end
        end

        if indegree(g,nv(g)) < 1
            r = 0
            c = (c[1],c[1])
        end

        if !is_connected(g)
            rem_vertex!(g,nv(g))
            iterations += 1
            continue
        end

        if has_cycles(g)
            rem_vertex!(g, nv(g))
            iterations += 1
            continue
        end

        if max_basal(g, net_gen.basal_limit, n_nodes)
            rem_vertex!(g,nv(g))
            iterations += 1
            continue
        end

        working_du, u0, p = test_dynamics(g, func, u0, dyn, net_gen)
        if !working_du
            rem_vertex!(g,nv(g))
            iterations += 1
            continue
        end
        append!(η_vec, η); append!(r_vec, r); push!(c_vec, c);
    end

    target_nodes, target_states = get_targets(g, func, u0, dyn, net_gen)

    if iterations < net_gen.iter_limit
        func = fmtk_cr(vectorize_params(p))
        l_func = fmtk_cr(vectorize_params(p); log_space = true)
    else
        func = l_func = nothing
    end

    #println("Complete!")
    
    universal_params.opt.targets = target_states
    universal_params.opt.invaders = Float64.(target_nodes)

    save_struct = SaveObject(universal_params, vectorize_params(p), u0)

    if !isnothing(saveloc)
        jldsave(saveloc; network_params = save_struct)
        return nothing
    else
        return save_struct, iterations
    end
end

function gen_networks(n_nodes, n_networks, func, universal_params::UniversalParams, savefolder::String)
    
    mkpath(savefolder)

    Threads.@threads for i in ProgressBar(1:n_networks)
        network_generation(n_nodes, func, universal_params; saveloc = savefolder*"/graph_$(i).jld2")
    end
end


function generate_many_graph(n_nodes, n_networks, func, savefolder_starter::String; _B0_vec::Union{Nothing, Vector{Float64}} = nothing, _S_vec::Union{Nothing, Vector{Float64}} = nothing, _log_space = nothing)

    if isnothing(_B0_vec) & isnothing(_S_vec)
        throw("What are you doing? You can just initialize with the toml!")
        return
    end

    B0_vec = isnothing(_B0_vec) ? [nothing] : _B0_vec
    S_vec = isnothing(_S_vec) ? [nothing] : _S_vec

    for B0 in B0_vec
        for S in S_vec
            up = initialize(; _B0 = B0, _S = S, _log_space = _log_space)
            println("B0: $(up.dyn.B0)")
            println(" S: $(up.dyn.S)")
            savefolder = savefolder_starter*"-B0_$(up.dyn.B0)-S_$(up.dyn.S)"
            gen_networks(n_nodes, n_networks, func, up, savefolder)
        end
    end

end