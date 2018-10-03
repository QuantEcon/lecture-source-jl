# Goal of this file is to test unicode support across platforms we'll be using (Atom, Terminal, Jupyter) and across OSs.

# Glyphs
    L̃ = 2.0 # L-tilde
    u₀ = 3.0 # u-naught
    â = 4.0 # a-hat
    z̲ = 5.0 # z-underbar

# Ordinary unicode characters
    println("π, α, ζ, ξ, δ, ρ, γ, β, χ, ω, θ, υ, ν, Γ, Δ")

# Git
    "⋅" # cdot
    "⊗" # otimes
    "∂" # partial
    "ℒ" # scrL
    "𝒟" # scrD
    "∈" # in
    "∉" # notin
    "⊆" # subseteq
    "≠" # ne
    "≈" # approx

# Ones to avoid!
# Combining characters don't show on Windows Atom/vscode/jupyter
    γ̃  = 1 # gamma-tilde
    α̂ = 1 # alpha-hat
    π̲ = 1 # pi-underbar
    π̄ = 1 # pi-bar
