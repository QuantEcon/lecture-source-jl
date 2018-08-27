using StatPlots     # needed for box plot support

function tmp()
    n = 500
    x = randn(n)        # N(0, 1)
    x = exp.(x)         # Map x to lognormal
    y = randn(n) .+ 2.0  # N(2, 1)
    z = randn(n) .+ 4.0  # N(4, 1)
    data = vcat(x, y, z)
    l = [LaTeXString("\$X\$") LaTeXString("\$Y\$")  LaTeXString("\$Z\$") ]
    xlabels = reshape(repeat(l, n), n*3, 1)

    boxplot(xlabels, data, label="", ylims=(-2, 14))
end
tmp()
