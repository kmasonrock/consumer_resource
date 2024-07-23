using Distributed
addprocs(8)
@everywhere begin
    using JLD2
    using ProgressBars
    include("cr_inline.jl")
    include("ecosystem_model_functions.jl")
    include("bottum_up_cr.jl")
end

@everywhere function generate_different_h_and_b(n_nodes, n_graphs, h_range, b0_range)

    for h in h_range
        @sync @distributed for b0 in b0_range
            for i in 1:n_graphs
                bottum_up_niche(n_nodes, 1, 0.2, b0, h, i, Hill)
            end
        end
    end
end