import REPL
Base.atreplinit() do repl
    # make the ; shell mode sticky
    !isdefined(repl, :interface) && (repl.interface = REPL.setup_interface(repl))
    repl.interface.modes[2].sticky=true
end

try
    @eval using Revise
    # Turn on Revise's automatic-evaluation behavior
    Revise.async_steal_repl_backend()
catch err
    @warn "Could not load Revise."
end

