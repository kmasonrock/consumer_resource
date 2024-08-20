using Graphs
using Distributions
using Random
using DiffEqCallbacks
include("ecosystem_model_functions.jl")
include("functional_response.jl")
include("allee_effects.jl")
include("consumer_resource.jl")
include("cr_params.jl")


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

function test_eig(jac, vals, J, p)
    jac(J, vals, p, nothing)
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


function test_dynamics(g, func, jac, u0, B0, h, allee_effect::Union{Type{Hill}, Type{Absent}}; start_val = log(1e-3), integrate_time = [0 50000], abstol = 1e-5, reltol = 1e-5, fp_thresh = 1e-4, ex_thresh = log(1e-6), testing_targets = false)
    init_conditions = deepcopy(u0)
    if !testing_targets
        append!(init_conditions, start_val)
    end
    
    du = zeros(length(init_conditions))
    J = zeros(length(init_conditions), length(init_conditions))

    try
        p = cr_params(g, allee_effect, B0, h, init_conditions)
        prob = ODEProblem(func, init_conditions, integrate_time, p)
        sol = solve(prob, AutoTsit5(Rosenbrock23()), save_everystep = false, abstol = abstol, reltol = reltol, callback = TerminateSteadyState(abstol, reltol))
        if sol.t[end] == integrate_time[end]
            #println("Cannot reach stable state")
            return false, u0, nothing
        end
        if all(x -> x > ex_thresh, sol[end]) & test_eig(jac, sol[end], J, p) & test_du(func, sol[end], du, p, fp_thresh)
            return true, sol[end], p
        elseif testing_targets
            if (count(x -> x > ex_thresh, sol[end]) == (nv(g) - 1)) & test_eig(jac, sol[end], J, p) & test_du(func, sol[end], du, p, fp_thresh)
                return true, sol[end], p
            else
                return false, u0, p
            end
        else
            return false, u0, p
        end

    catch e
        @infiltrate
        if e == SingularException
            return false, u0, cr_params(SimpleDiGraph(), allee_effect, B0, h)
        end
    end
end

function initiate_niche()
    u0 = 0.0
    η = rand(Uniform(0,1))
    r = 0.0
    centre = rand(Uniform(r/2, η))
    c = (centre,centre)
    g = SimpleDiGraph(1)
    return [u0], [η], [r], [c], g
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

function get_targets(g, func, jac, u0, B0, h, allee_effect::Union{Type{Hill}, Type{Absent}})
    target_nodes = []
    target_states = Vector{Vector{Float64}}(undef, 0)
    for i in 1:nv(g)
        init_vals = deepcopy(u0)
        init_vals[i] = log(1e-8)

        works, ts, _ = test_dynamics(g, func, jac, init_vals, B0, h, allee_effect; testing_targets = true)
        if works
            if (count(x -> x > log(1e-7), ts) == (nv(g) - 1)) & (ts[i] < log(1e-8))
                append!(target_nodes, i)
                push!(target_states, ts)
            end
        end
    end
    return target_nodes, target_states
end

function niche_model_graph(n_nodes, func, jac, allee_effect::Union{Type{Hill}, Type{Absent}}; basal_limit = 0.32, B0 = 1.0, h = 2.0, saveloc::Union{String,Nothing} = nothing, iter_limit = 1000)
    local p
    u0, η_vec, r_vec, c_vec, g = initiate_niche()
    iterations = 0
    while nv(g) < n_nodes
        if iterations >= iter_limit
            u0, η_vec, r_vec, c_vec, g = initiate_niche()
            iterations = 0
        end

        add_vertex!(g)
        η, r, c = generate_node_params()

        for (i,n) in enumerate(η_vec)
            if (c[1] <= n) & (c[2] >= n)
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

        if max_basal(g, basal_limit, n_nodes)
            rem_vertex!(g,nv(g))
            iterations += 1
            continue
        end

        working_du, u0, p = test_dynamics(g, func, jac, u0, B0, h, allee_effect)
        if !working_du
            rem_vertex!(g,nv(g))
            iterations += 1
            continue
        end
        append!(η_vec, η); append!(r_vec, r); push!(c_vec, c);
        println("$(nv(g))/$(n_nodes)")
    end
    target_nodes, target_states = get_targets(g, func, jac, u0, B0, h, allee_effect)
    println("Complete!")
    if !isnothing(saveloc)
        jldsave(saveloc; g = g, p = p, stable = u0, target_ind = target_nodes, target_states = target_states)
    end
    return g, p, u0, target_nodes, target_states
end


    


 
