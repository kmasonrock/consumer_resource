include("cr_inline.jl")
include("ecosystem_model_functions.jl")

struct CRParameter
    g::SimpleGraph
end

function kmp(n = 20, e = 120)
    return CRParameter(SimpleGraph(n,e))
end

function kuramoto_model(du, u, p, t)
    for i in vertices(p.g)
        du[i] = sum(sin(u[j] - u[i]) for j in all_neighbors(p.g,i))
    end
end

function func_test(n = 10_000)
    file = jldopen("new_function_graphs/graph_1.jld2")
    g = file["g"]
    p = cr_params(g)
    du = Vector{Float64}(undef,20)
    u = log.(0.5*ones(20))
    t = 0
    for i in 1:n
        println(i)
        new_log_sean_cr!(du,u,p,t)
    end
    return du
end

function ode_func_test(n = 10_000)
    file = jldopen("new_function_graphs/graph_1.jld2")
    g = file["g"]
    p = cr_params(g)
    du = Vector{Float64}(undef,20)
    J = zeros(20,20)
    u = log.(0.5*ones(20))

    func = ODEFunction(new_log_sean_cr!; jac = new_log_sean_jac!)
    println("Function test...")
    sleep(1)
    for i in 1:n
        println(i)
        func.f(du, u, p, nothing)
    end
    sleep(5)
    println("Jacobian test...")
    sleep(1)
    for i in 1:n
        println(i)
        func.jac(J, u, p, nothing)
    end

    return du, J
end


function ode_sol_test(n = 10_000)
    file = jldopen("new_function_graphs/graph_1.jld2")
    g = file["g"]
    p = cr_params(g)
    du = Vector{Float64}(undef,20)
    J = zeros(20,20)
    u = log.(0.5*ones(20))

    func = ODEFunction(new_log_sean_cr!; jac = new_log_sean_jac!)
    prob = ODEProblem(func, u, [0 5000], p)
    println("solution test...")

    for i in 1:n
        println(i)
        sol = solve(prob, Vern9(), abstol = 1e-10, reltol = 1e-10)
    end
    return sol
end

function test_km_model(n = 10_000)
    p = kmp()

    du = Vector{Float64}(undef,20)
    J = zeros(20,20)
    u = rand(20)

    func = ODEFunction(kuramoto_model)
    prob = ODEProblem(func, u, [0 5000], p)

    for i in 1:n
        println(i)
        sol = solve(prob, Vern9(), abstol = 1e-10, reltol = 1e-10)
    end

    return solve
end