using ModelingToolkit, OrdinaryDiffEq, Symbolics
using Glob, JLD2
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
    local multi_extinct_nt = 0
    local multi_extinct_tg = 0
    local me_nt_file = []
    local me_tg_file = []
    local one_extinct = 0
    local oe_file = []
    for file in files
        f = jldopen(file)
        termination = vcat(termination,f["term_status"])
        end_state = vcat(end_state,f["end_state"])
        target = parse(Int64, string(split(split(file,"_")[end],".")[1]))

        sol = f["best_sol"]
        num_alive = sol .>= -8

        if sum(num_alive) == length(sol)
            no_extinct += 1
            ne_file = vcat(ne_file, [file])

        elseif sum(num_alive) == (length(sol) - 1)
            one_extinct += 1
            oe_file = vcat(oe_file, [file])
        else
            mask = findall(x -> x == 1, num_alive)
            if target in mask
                multi_extinct_tg += 1
                me_tg_file = vcat(me_tg_file, [file])
            else
                multi_extinct_nt += 1
                me_nt_file = vcat(me_nt_file, [file])
            end
        end
    end

    jldsave(dir*"_extinction_stat_files.jld2"; me_nt = me_nt_file, me_tg = me_tg_file, oe = oe_file, ne = ne_file, counts = [multi_extinct_nt, multi_extinct_tg, one_extinct, no_extinct])
    @infiltrate
end
#hill_w0_h1_fail:
#   me = 84 (88.4%)
#   oe = 7
#   ne = 4

#hill_w0_h2_fail:
#   me = 90 (60.4%)
#   oe = 4
#   ne = 55

#w0-05_h1-2_fail
#   me = 92 (98.9%)
#   oe = 0
#   ne = 1


#Greedy Algorithm 3.0
# Success = 20
# No Extinct = 30
# Not Target = 18
# Multiple = 28


    