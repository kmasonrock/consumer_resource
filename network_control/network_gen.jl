using Graphs, MetaGraphs
using Distributions
using GLMakie, GraphMakie
using NetworkLayout


function niche_model_graph(n_nodes, β)
    
    global connected = false

    

    while connected == false
        g = SimpleDiGraph(n_nodes)
        global mg = MetaDiGraph(g)
        for i in 1:n_nodes
            set_props!(mg,i,Dict(:η=>rand(Uniform(0,1))))
            set_props!(mg,i,Dict(:B=>rand(Uniform(0,1))))
            set_props!(mg,i,Dict(:G=> (1-get_prop(mg,i,:B))))
        end

        global lowest_η = 1
        for i in 1:n_nodes
            new_η = get_prop(mg,i,:η)
            if new_η < lowest_η
                lowest_η = new_η
                global smallest = i
            end
        end

        #set_prop!(mg,smallest,:η,0)

        for i in 1:n_nodes
            if i == smallest
                r = 0
            else
                r = rand(Beta(1,β))*get_prop(mg,i,:η)
            end
            c = rand(Uniform(r/2,get_prop(mg,i,:η)))
            r_range = [c - r/2, c + r/2]

            for j in 1:n_nodes
                if r_range[1] <= get_prop(mg,j,:η) <= r_range[2]
                    add_edge!(mg,j,i)
                end
            end
        end
        connected = is_connected(mg)
        if connected == false
            g = 0
            mg = 0
        end
    end
    return mg
end

function get_beta(n_nodes,target_C,init_β)
    global inc = 0.05
    global β = init_β
    global error = 1

    global iterations = 0
    while error > 1e-3
        S² = n_nodes^2
        L_sm = []
        L_lg = []
        for i in 1:100
            g_sm = niche_model_graph(n_nodes,β[1])
            g_lg = niche_model_graph(n_nodes,β[2])
            append!(L_sm,length(collect(edges(g_sm))))
            append!(L_lg,length(collect(edges(g_lg))))
        end

        avg_L_sm = mean(L_sm)
        avg_L_lg = mean(L_lg)

        calc_C_sm = avg_L_sm/S²
        calc_C_lg = avg_L_lg/S²

        

        global m = (calc_C_sm + calc_C_lg)/2

        error = abs(m - target_C)/target_C
        if m > target_C
            β[1] = β[1] + inc
        elseif m < target_C
            β[2] = β[2] - inc
        end

        inc = inc/2

        iterations += 1

        println(β)

        if iterations >= 100
            println("Max Iterations")
            break
        end
    end
    return m, iterations
end




function plot_graph(g)
    return graphplot(g, self_edge_size = 0.2, layout = Spring(), curve_distance = 0.1, node_size = 20)
end


function avg_connectedness(n_nodes,β)
    S² = []
    L = []
    for i in 1:1000
        g = niche_model_graph(n_nodes,β)
        append!(S²,n_nodes^2)
        append!(L,length(collect(edges(g))))
    end

    return mean(L)/mean(S²)
end

function stat_test(n_nodes, β, ensemble_size)

    L_ens = []
    C_ens = []
    for i in 1:ensemble_size
        g = niche_model_graph(n_nodes,β)
        append!(L_ens,length(collect(edges(g))))
        append!(C_ens, length(collect(edges(g)))/n_nodes^2)
    end


    return mean(L_ens), mean(C_ens)
end

# g = niche_model_graph(25,1.5)
# println(length(collect(edges(g)))/25^2)
# graphplot(g)

#initialpos  = Dict(
#     1=>Point2f(-1,1),
#     2=>Point2f(-1,-1),
#     3=>Point2f(0,-1.73),
#     4=>Point2f(1,-1),
#     5=>Point2f(1,1),
#     6=>Point2f(0,1.73),
# )

# pin  = Dict(
#     1=>true,
#     2=>true,
#     3=>true,
#     4=>true,
#     5=>true,
#     6=>true
# )

# #β for 25 nodes seems to be ~ 2.162

# g = niche_model_graph(10,2.162)

# L,C = stat_test(30,2.2225,100000)

#graphplot(g,self_edge_size = 0.2, layout = Shell(), curve_distance = 0.2, node_size = 20)

#plot_graph()
