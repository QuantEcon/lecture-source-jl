using Pkg
Pkg.activate(@__DIR__); #Activate project local to notebook
pkg"instantiate; precompile"
pkg"st"
