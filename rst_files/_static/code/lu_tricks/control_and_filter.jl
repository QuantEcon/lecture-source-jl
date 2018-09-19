#=

Author: Shunsuke Hori

=#

using Polynomials, LinearAlgebra

struct LQFilter{TF_ <: Union{Vector{Float64},Nothing}, TI_ <: Union{Int,Nothing}}
    d::Vector{Float64}
    h::Float64
    y_m::Vector{Float64}
    m::Int
    ϕ::Vector{Float64}
    β::Float64
    ϕ_r::TF_
    k::TI_
end

function LQFilter(d, h, y_m;
                  r = nothing,
                  β = nothing,
                  h_eps = nothing)

    m = length(d) - 1

    m == length(y_m) ||
        throw(ArgumentError("y_m and d must be of same length = $m"))

    #---------------------------------------------
    # Define the coefficients of ϕ up front
    #---------------------------------------------

    ϕ = zeros(2m + 1)
    for i in -m:m
        ϕ[m-i+1] = sum(diag(d*d', -i))
    end
    ϕ[m+1] = ϕ[m+1] + h

    #-----------------------------------------------------
    # If r is given calculate the vector ϕ_r
    #-----------------------------------------------------

    if r === nothing
        k = nothing
        ϕ_r = nothing
    else
        k = size(r, 1) - 1
        ϕ_r = zeros(2k + 1)

        for i = -k:k
            ϕ_r[k-i+1] = sum(diag(r*r', -i))
        end

        if h_eps !== nothing
            ϕ_r[k+1] = ϕ_r[k+1] + h_eps
        end
    end

    #-----------------------------------------------------
    # If β is given, define the transformed variables
    #-----------------------------------------------------
    if β === nothing
        β = 1.0
    else
        d = β.^(collect(0:m)/2) * d
        y_m = y_m * β.^(- collect(1:m)/2)
    end

    return LQFilter(d, h, y_m, m, ϕ, β, ϕ_r, k)
end

function construct_W_and_Wm(lqf, N)

    d, m = lqf.d, lqf.m

    W = zeros(N + 1, N + 1)
    W_m = zeros(N + 1, m)

    #---------------------------------------
    # Terminal conditions
    #---------------------------------------

    D_m1 = zeros(m + 1, m + 1)
    M = zeros(m + 1, m)

    # (1) Constuct the D_{m+1} matrix using the formula

    for j in 1:(m+1)
        for k in j:(m+1)
            D_m1[j, k] = dot(d[1:j, 1], d[k-j+1:k, 1])
        end
    end

    # Make the matrix symmetric
    D_m1 = D_m1 + D_m1' - Diagonal(diag(D_m1))

    # (2) Construct the M matrix using the entries of D_m1

    for j in 1:m
        for i in (j + 1):(m + 1)
            M[i, j] = D_m1[i-j, m+1]
        end
    end
    M

    #----------------------------------------------
    # Euler equations for t = 0, 1, ..., N-(m+1)
    #----------------------------------------------
    ϕ, h = lqf.ϕ, lqf.h

    W[1:(m + 1), 1:(m + 1)] = D_m1 + h * I
    W[1:(m + 1), (m + 2):(2m + 1)] = M

    for (i, row) in enumerate((m + 2):(N + 1 - m))
        W[row, (i + 1):(2m + 1 + i)] = ϕ'
    end

    for i in 1:m
        W[N - m + i + 1 , end-(2m + 1 - i)+1:end] = ϕ[1:end-i]
    end

    for i in 1:m
        W_m[N - i + 2, 1:(m - i)+1] = ϕ[(m + 1 + i):end]
    end

    return W, W_m
end

function roots_of_characteristic(lqf)
    m, ϕ = lqf.m, lqf.ϕ

    # Calculate the roots of the 2m-polynomial
    ϕ_poly = Poly(ϕ[end:-1:1])
    proots = roots(ϕ_poly)
    # sort the roots according to their length (in descending order)
    roots_sorted = sort(proots, by=abs)[end:-1:1]
    z_0 = sum(ϕ) / polyval(poly(proots), 1.0)
    z_1_to_m = roots_sorted[1:m]     # we need only those outside the unit circle
    λ = 1 ./ z_1_to_m
    return z_1_to_m, z_0, λ
end

function coeffs_of_c(lqf)
    m = lqf.m
    z_1_to_m, z_0, λ = roots_of_characteristic(lqf)
    c_0 = (z_0 * prod(z_1_to_m) * (-1.0)^m)^(0.5)
    c_coeffs = coeffs(poly(z_1_to_m)) * z_0 / c_0
    return c_coeffs
end

function solution(lqf)
    z_1_to_m, z_0, λ = roots_of_characteristic(lqf)
    c_0 = coeffs_of_c(lqf)[end]
    A = zeros(lqf.m)
    for j in 1:m
        denom = 1 - λ/λ[j]
        A[j] = c_0^(-2) / prod(denom[1:m .!= j])
    end
    return λ, A
end

function construct_V(lqf; N = nothing)
    if N === nothing
        error("N must be provided!!")
    end
    if !(N isa Integer)
        throw(ArgumentError("N must be Integer!"))
    end

    ϕ_r, k = lqf.ϕ_r, lqf.k
    V = zeros(N, N)
    for i in 1:N
        for j in 1:N
            if abs(i-j) ≤ k
                V[i, j] = ϕ_r[k + abs(i-j)+1]
            end
        end
    end
    return V
end

function simulate_a(lqf, N)
    V = construct_V(N + 1)
    d = MVNSampler(zeros(N + 1), V)
    return rand(d)
end

function predict(lqf, a_hist, t)
    N = length(a_hist) - 1
    V = construct_V(N + 1)

    aux_matrix = zeros(N + 1, N + 1)
    aux_matrix[1:t+1 , 1:t+1 ] = Matrix(I, t + 1, t + 1)
    L = cholesky(V).U'
    Ea_hist = inv(L) * aux_matrix * L * a_hist

    return Ea_hist
end

function optimal_y(lqf, a_hist, t = nothing)
    β, y_m, m = lqf.β, lqf.y_m, lqf.m

    N = length(a_hist) - 1
    W, W_m = construct_W_and_Wm(lqf, N)

    F = lu(W, Val(true))

    L, U = F
    D = diagm(0 => 1.0 ./ diag(U))
    U = D * U
    L = L * diagm(0 => 1.0 ./ diag(D))

    J = reverse(Matrix(I, N + 1, N + 1), dims = 2)

    if t === nothing                      # if the problem is deterministic
        a_hist = J * a_hist

        #--------------------------------------------
        # Transform the a sequence if β is given
        #--------------------------------------------
        if β != 1
            a_hist =  reshape(a_hist * (β^(collect(N:0)/ 2)), N + 1, 1)
        end

        a_bar = a_hist - W_m * y_m        # a_bar from the lecutre
        Uy = \(L, a_bar)                  # U @ y_bar = L^{-1}a_bar from the lecture
        y_bar = \(U, Uy)                  # y_bar = U^{-1}L^{-1}a_bar
        # Reverse the order of y_bar with the matrix J
        J = reverse(Matrix(I, N + m + 1, N + m + 1), dims = 2)
        y_hist = J * vcat(y_bar, y_m)     # y_hist : concatenated y_m and y_bar
        #--------------------------------------------
        # Transform the optimal sequence back if β is given
        #--------------------------------------------
        if β != 1
            y_hist = y_hist .* β.^(- collect(-m:N)/2)
        end

    else                                  # if the problem is stochastic and we look at it
        Ea_hist = reshape(predict(a_hist, t), N + 1, 1)
        Ea_hist = J * Ea_hist

        a_bar = Ea_hist - W_m * y_m       # a_bar from the lecutre
        Uy = \(L, a_bar)                  # U @ y_bar = L^{-1}a_bar from the lecture
        y_bar = \(U, Uy)                  # y_bar = U^{-1}L^{-1}a_bar

        # Reverse the order of y_bar with the matrix J
        J = reverse(Matrix(I, N + m + 1, N + m + 1), dims = 2)
        y_hist = J * vcat(y_bar, y_m)     # y_hist : concatenated y_m and y_bar
    end
    return y_hist, L, U, y_bar
end
