using ModelingToolkit, OrdinaryDiffEq, Symbolics
using Glob, JLD2
using PyPlot
using LinearAlgebra, StatsBase
using JuMP
import Ipopt, HSL_jll
using Infiltrator

function model_stats(dir)
    files = glob(string(dir,"/*.jld2"))
    
    local termination = []
    local end_state = []
    local no_extinct = 0
    local ne_file = []
    local multi_extinct = 0
    local me_file = []
    local one_extinct = 0
    local oe_file = []
    for file in files
        f = jldopen(file)
        termination = vcat(termination,f["term_status"])
        end_state = vcat(end_state,f["end_state"])

        sol = f["best_sol"].u

        num_alive = sol[end] .>= -8

        if sum(num_alive) == length(sol[end])
            no_extinct += 1
            ne_file = vcat(ne_file, [file])

        elseif sum(num_alive) == (length(sol[end]) - 1)
            one_extinct += 1
            oe_file = vcat(oe_file, [file])
        else
            multi_extinct += 1
            me_file = vcat(me_file, [file])
        end
    end
    @infiltrate
end


#Greedy Algorithm 3.0
# Success = 20
# No Extinct = 30
# Not Target = 18
# Multiple = 28


    