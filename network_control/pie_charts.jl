include("initialize.jl")


folders = glob("PerturbedGraphs_Compare/*")

titles = transpose(reshape(collect(0.6:0.05:1.0),3,3))

fig = Figure(size = (2400,400))

colors = [:red, :green, :yellow, :blue, :purple]
points = ["Success", "Multiple with Target", "Multiple excluding target", "Not Target", "No Extinctions"]

for (ind,folder) in enumerate(folders)
    files = glob(folder*"/*.jld2")
    println(folder)
    success = 0
    me_target = 0
    me_notarget = 0
    not_target = 0
    no_extinct = 0

    x = Int(ceil(ind/3))
    y = Int(mod(ind-1,3)) + 1

    println(titles[x,y])
    ax = Axis(fig[x,y], title = string(titles[x,y]), aspect = DataAspect())

    for file in files
        f = jldopen(file)

        if f["retcode"] == "Success"
            success += 1
        elseif f["retcode"] == "MultiExtinct_WithTarget"
            me_target += 1
        elseif f["retcode"] == "MultiExtinct_NotTarget"
            me_notarget += 1
        elseif f["retcode"] == "OneExtinct_NotTarget"
            not_target += 1
        elseif f["retcode"] == "NoExtinctions"
            no_extinct += 1
        else
            @infiltrate
            throw("No Retcode was returned!")
        end
    end

    pie!(ax, [success, me_target, me_notarget, not_target, no_extinct], color = colors)

end

marker_elems= Vector{MarkerElement}(undef, 5)
for i in eachindex(marker_elems)
    marker_elems[i] = MarkerElement(color = colors[i], marker = :circle, markersize = 20)
end

Legend(fig[1,4], marker_elems, points)
save("pie_charts_compare.png", fig)


