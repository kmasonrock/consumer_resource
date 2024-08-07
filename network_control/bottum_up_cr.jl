using ModelingToolkit, OrdinaryDiffEq
using JLD2
using Graphs, MetaGraphs
using GLMakie, GraphMakie
using LinearAlgebra
using Symbolics
using Random
using InvertedIndices
using Distributions
using Infiltrator

# Contains the functions for all parameters and dynamics
# of the consumer-resource model. Uses Symbolics and ModelingToolkit
# to generate -- useful for calculating the Jacobian when we get to
# the algorithm we use to perturb the system.
include("ecosystem_model_functions.jl")
include("cr_inline.jl")

function make_loc(loc)
    mkpath(loc)
end

mutable struct NicheStruct
    # Holds data for all nodes in the graph.
    # If used correctly, we can easily save this data
    # and even switch programming langauges if necessary!
    # (Python might be prettier than Julia for visualization)

    r::Vector{Float64} # Range of prediation
    η::Vector{Float64} # Niche value parameter, taken from uniform dist
    c::Vector{Float64} # Centre of the range for the niche model_params
    G::SimpleDiGraph   # The graph generated from these parameters.
end

function test_dynamics(ns::NicheStruct,u0, B0, h, allee_type::Union{Type{Hill}, Type{Strong}, Type{Absent}})
    # The parameters used here are chosen based on
    # parameters used in my PhD advisor Sean Cornelius'
    # function, in attempt to acheive his success rate
    # for viable graphs.

    # Parameters from Sagar's paper (referenced throughout
    # this code) are as follows:
    # herb = 0.45, carn = 0.85, dxdr = 0.597, Z = 10, m = -0.25,
    # ω = 0.05, y = 6, and S = 0.

    # Ω = get_Ω(ns.G)
    # e = get_assim_eff(ns.G, 0.85, 0.85)
    # x = get_x(0.597, 10, -0.25, get_trophic(ns.G))
    # sys, log_sys, B, t = dynamics_gen(ns.G, Ω, e, x, 0.5, 1.5, 10, 0.01)

    try
        p = cr_params(ns.G, allee_type, B0, h)
        prob = ODEProblem(new_log_sean_cr!, u0, [0 5000], p)
        sol = solve(prob, Vern9(); verbose = false)
    
        if any(x -> x < log(1e-4), sol[end])
            return false, u0
        end
    
        return true, sol[end] 
    catch e
        if isa(e, LAPACKException)
            return false, u0
        end
    end
    
 
end

function bottum_up_niche(graph_size, basal_start, basal_limit, B0, h, graph_num, allee_type::Union{Type{Hill}, Type{Strong}, Type{Absent}}, dyn_tol = 1000, enforce_size = true)
    #Inspirtaiton for this code comes from Sagar Sahasraudhe (2011)
    function allee_string(allee_type::Type{Hill})
        make_loc("hill_w0_h$(h)_B0_$(B0)")
        return "hill_w0_h$(h)_B0_$(B0)"
    end

    function allee_string(allee_type::Type{Strong})
        make_loc("strong_2")
        return "strong"
    end

    function allee_string(allee_type::Type{Absent})
        make_loc("absent_2")
        return "absent"
    end


    
    function graph_init(basal_start) 
        # This function initializes the graph based on
        # the number of basal species allowed to exist.
        # Returns a stucture unique to the graph including
        # all parameters for generating the graph via the
        # niche model.
        G = SimpleDiGraph()
        init_r = zeros(basal_start)
        init_η = zeros(basal_start)
        init_c = zeros(basal_start)
        for s in 1:basal_start
            init_η[s] = rand(Uniform(0,1))
            add_vertex!(G)
        end
        return NicheStruct(init_r, init_η, init_c, G)
    end
    
    function add_prey(ns::NicheStruct, prey)
        #Creates a link between prey and a given node
        for s in prey
            add_edge!(ns.G,s,nv(ns.G))
        end
    end

    function add_pred(ns::NicheStruct, pred)
        #Creates a link between a given node and its predator(s)
        for s in pred
            add_edge!(ns.G,nv(ns.G),s)
        end
    end

    function rem_disconnected(ns::NicheStruct)
        # If the graph is not weakly connected, this function
        # removes the unconnected nodes and the parameters
        # associated with them in the NicheStruct    

        disconnected = findall(x -> x == 0, degree(ns.G)) # Find all disconnected nodes

        rem_vertices!(ns.G, disconnected, keep_order = true) # Remove these vertices, preserving order
        
        ns.r = ns.r[Not(disconnected)]
        ns.c = ns.c[Not(disconnected)]
        ns.η = ns.η[Not(disconnected)]

        while length(connected_components(ns.G)) != 1 # While there are multiple connected components
            cc = connected_components(ns.G) # Returns connected components
            _,lcc = findmax(length.(cc)) #Gives us the index of the largest connected component
            rem = cc[Not(lcc)][1] # What is the first component that is not largest?

            rem_vertices!(ns.G, rem, keep_order = true) #Remove these nodes, preserving order
            ns.r = ns.r[Not(rem)]
            ns.c = ns.c[Not(rem)]
            ns.η = ns.η[Not(rem)]
        end

    end

    function grow(ns::NicheStruct, basal_limit, n, dyn_tol, u0, allee_type::Union{Type{Hill}, Type{Strong}, Type{Absent}}, β = 1.5)

        # Function adds a node, determines niche parameters, 
        # subsequently builds the network -- connecting the
        # new node to its respective prey and predators.
        # Note: The direction of links matches the direction
        # where biomass travels. That is, a predator obtains
        # biomass from its prey, and therefore directions go
        # prey --> predator.

        # We have a tolerance here (number of iterations before we say it failed)
        # because it is possible that the graph we construct is so inhibitive that
        # we can never have a steady state. In these cases, we want to start from scratch.
        i = 1
        while i <= dyn_tol
            new_u0 = append!(deepcopy(u0), -4.0)
            # if mod(i, round(dyn_tol/10)) == 0
            #        println("$(round(i/dyn_tol; digits = 2)*100)% of tolerance...")
            # end

            η = rand(Uniform(0,1)) #Niche value
            r = rand(Beta(1,β))*η #Cal
            c = rand(Uniform(r/2, η))

            add_vertex!(ns.G)

            centre = ns.c
            range = ns.r

            # Predator Loop
            # Connects current species to all its predators
            pred = []
            for s in 1:(nv(ns.G) - 1)
                if centre[s] - range[s]/2 < η < centre[s] + range[s]/2
                    append!(pred,s)
                end
            end

            # Prey Loop
            # Connects current species to all its prey
            prey= []
            for s in 1:(nv(ns.G) - 1)
                if c - r/2 < ns.η[s] < c + r/2
                    append!(prey, s)
                end
            end

            # If the node doesn't have any prey, then this makes it a basal species.
            # It also ensures that there is no excess of basal species, given that 
            # any environment should have 32% ± 8% basal species in the network 
            # (Sahasraudhe, 2011, Supplimentary Information)
            if (isempty(prey))
                r = 0; c = 0
                if ((length(findall(x -> x == 0, indegree(ns.G)))/n) >= basal_limit)
                    rem_vertex!(ns.G,nv(ns.G))
                    new_u0 = new_u0[1:end-1]
                    i += 1
                    continue
                end
            end

            add_prey(ns,prey) #Add the links, as they're needed for the dynamics!
            add_pred(ns,pred)

            if c - r/2 < η < c + r/2
                add_edge!(ns.G, nv(ns.G), nv(ns.G))
            end 
            stable, new_u0 = test_dynamics(ns, new_u0, B0, h, allee_type)
            if  stable #If stable dynamics exist for the system
                # Then we're successful, and we add the links for the prey and predators
                # to node s,
                #and add these values to the NicheStructure
                append!(ns.η, η)
                append!(ns.r, r)
                append!(ns.c, c)# We return true here because if we're successful, we leave the loop
                return true, true, new_u0
            else
                # If the dyanmics are not stable, we remove the node and try again.
                rem_vertex!(ns.G, nv(ns.G))
                new_u0 = new_u0[1:end-1]

                i += 1
                if i == dyn_tol
                            # If this fails, we return false so we can better control the while loop
                            # that calls the grow function.
                    return true, false, u0
                end
            end
        end
        return false, false, u0
    end

    function get_targets(graph_num::Union{Int64, Float64}, g::SimpleDiGraph{Int64}, u0::Vector{Float64}, B0, h, allee_type::Union{Type{Hill}, Type{Strong}, Type{Absent}})
        targets = Vector{Vector{Float64}}(undef, nv(g))
        p = cr_params(g, allee_type, B0, h)
        for node in 1:nv(g)
            new_u0 = deepcopy(u0)
            new_u0[node] = -18

            prob = ODEProblem(new_log_sean_cr!, new_u0, [0 5000], p)
            sol = solve(prob, Vern9(), save_everystep = false; verbose = false)

            targets[node] = sol[end]
        end

        ind = findall(x -> sum(@. x > log(1e-4)) == length(x)-1, targets)
        targets = targets[ind]
        loc = allee_string(allee_type)
        #mkpath(loc)
        jldsave(loc*"/graph_"*string(Int64(graph_num))*".jld2"; g = g, stable = u0, target_ind = ind, target_states = targets, p = p)
        nothing
    end

    function generate_test_graph()
        g = SimpleDiGraph(4)
        add_edge!(g,1,3)
        add_edge!(g,2,3)
        add_edge!(g,4,4)
        
        η = [0.0, 0.0, 0.15, 0.5]
        r = [0.0, 0.0, 0.1, 0.2]
        c = [0.0, 0.0, 0.1, 0.4]

        ns = NicheStruct(r,η,c,g)

        return ns
    end

    ns = graph_init(basal_start) # Initializes the graph based
                                #  on the initial number of basal
                                #  species specified
    u0 = Vector{Float64}(undef,1)
    u0[1] = 0.0


    s = basal_start
    while s < graph_size        # This loop grows the graph to a given size
        basal, dynamics, u0 = grow(ns, basal_limit, graph_size, dyn_tol,u0, allee_type)
        if basal & dynamics          # If false, we know to start again for node "s"
            s += 1
            #println("$s/$graph_size")
        elseif basal & ~dynamics
            ns = graph_init(basal_start)
            s = basal_start
        end

        if s == graph_size
            if ~is_weakly_connected(ns.G) #If the graph is not connected
                rem_disconnected(ns) # Remove the unconnected species.
                if enforce_size
                    s = nv(ns.G)
                end
            end
            get_targets(graph_num, ns.G, u0, B0, h, allee_type)
        end

    end
    return ns
end