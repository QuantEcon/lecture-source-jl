# PackageCompiler sysimg compiling. 
using Pkg, InstantiateFromURL  
github_project("QuantEcon/quantecon-notebooks-julia") # for versioning
pkg"add GR Plots Plots StatsPlots DataFrames StatsPlots CSV PlotUtils GeometryTypes Tables PackageCompiler#sd-notomls CategoricalArrays IteratorInterfaceExtensions PooledArrays WeakRefStrings"
pkg"add Images DualNumbers Unitful Compat LaTeXStrings UnicodePlots DataValues IterativeSolvers VisualRegressionTests"

# new sysimg 
using PackageCompiler
syso, sysold = PackageCompiler.compile_incremental(:Plots, :DataFrames, :CSV, :StatsPlots, install = true)
cp(syso, sysold, force = true)
