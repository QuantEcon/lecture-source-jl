# necessary setup
using Pkg 
Pkg.activate("../Project.toml")
using GR, Plots
GR.__init__()

# make sure the lecture runs
plot = Plots.plot
contour = Plots.contour

# include the McCall code 
using NBInclude
@nbinclude("mccall_model.ipynb")