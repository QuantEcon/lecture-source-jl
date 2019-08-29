using GR, Plots
GR.__init__()

# Methods from McCall
x = rand(100)
Plots.plot(x) 

using Distributions, Expectations
d = Normal()
E = expectation(d)