using CairoMakie

function hill(x,S2)
    return x/(S2 + x)
end

function strong(x,S1,S2,S3)
    return 1 - ((S1 + S2)/(S2 + x))^S3
end

function compare_effects(S1,S2,S3)
    x = exp.(collect(range(-23,0,250)))

    hill_data = Point2f.(x, hill.(x,S2))
    strong_data = Point2f.(x, strong.(x,S1,S2,S3))

    f = Figure()
    ax = Axis(f[1,1], xlabel = "Biomass", ylabel = "Strength", title = "S = [$S1, $S2, $S3]", xscale = log)

    lines!(hill_data, label = "Hill")
    lines!(strong_data, label = "Strong")

    f[1, 2] = Legend(f, ax, "Allee Type", framevisible = false)

    return f
end