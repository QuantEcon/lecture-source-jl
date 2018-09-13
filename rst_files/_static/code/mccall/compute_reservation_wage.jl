function compute_reservation_wage(mcm; return_values = false)
    V, U = solve_mccall_model(mcm)
    w_idx = searchsortedfirst(V .- U, 0)

    if w_idx == length(V)
        w_bar = Inf
    else
        w_bar = mcm.w_vec[w_idx]
    end

    if return_values == false
        return w_bar
    else
        return w_bar, V, U
    end
end
