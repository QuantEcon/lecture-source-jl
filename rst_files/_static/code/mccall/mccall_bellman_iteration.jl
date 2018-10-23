using Distributions, LinearAlgebra, Compat, Expectations

# A default utility function

function u(c, σ)
    if c > 0
        return (c^(1 - σ) - 1) / (1 - σ)
    else
        return -10e6
    end
end

# default wage vector with probabilities

const n = 60                                           # n possible outcomes for wage
const default_w_vec = range(10, 20, length = n) # wages between 10 and 20
const a, b = 600, 400                                  # shape parameters
const dist = BetaBinomial(n-1, a, b)

mutable struct McCallModel{TF <: AbstractFloat,
                           TAV <: AbstractVector{TF}}
    α::TF         # Job separation rate
    β::TF         # Discount rate
    γ::TF         # Job offer rate
    c::TF         # Unemployment compensation
    σ::TF         # Utility parameter
    w_vec::TAV    # Possible wage values

    McCallModel(α::TF = 0.2,
                β::TF = 0.98,
                γ::TF = 0.7,
                c::TF = 6.0,
                σ::TF = 2.0,
                w_vec::TAV = default_w_vec,
                ) where {TF, TAV} =
        new{TF, TAV}(α, β, γ, c, σ, w_vec)
end

function update_bellman!(mcm, V, V_new, U, E)
    # Simplify notation
    α, β, σ, c, γ = mcm.α, mcm.β, mcm.σ, mcm.c, mcm.γ

    for (w_idx, w) in enumerate(mcm.w_vec)
        # w_idx indexes the vector of possible wages
        V_new[w_idx] = u(w, σ) + β * ((1 - α) * V[w_idx] + α * U)
    end

    U_new = u(c, σ) + β * (1 - γ) * U +
            β * γ * E*max.(U, V)
    return U_new
end

function solve_mccall_model(mcm; tol = 1e-5, max_iter = 2000)

    V = ones(length(mcm.w_vec))    # Initial guess of V
    V_new = similar(V)             # To store updates to V
    U = 1.0                        # Initial guess of U
    i = 0
    error = tol + 1
    E = expectation(dist, nodes = mcm.w_vec)

    while error > tol && i < max_iter
        U_new = update_bellman!(mcm, V, V_new, U, E)
        error_1 = maximum(abs, V_new - V)
        error_2 = abs(U_new - U)
        error = max(error_1, error_2)
        V[:] = V_new
        U = U_new
        i += 1
    end

    return V, U
end
