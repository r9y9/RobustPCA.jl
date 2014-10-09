# Robust Principal Component Analysis (RPCA) using the inexact augumented
# Lagrange multiplier.
# Given a observation matrix D, find row-rank matrix A and sparse matrix E
# so that D = A + E.
function rpca_inexact_alm(D::AbstractMatrix;
                          max_iter::Int=1000, error_tol::Float64=1.0e-7,
                          ρ::Float64=1.5, verbose::Bool=false)
    const M, N = size(D)
    const λ = 1 / sqrt(M)

    A⁰, E⁰ = zeros(M, N), zeros(M, N)
    
    # initialize
    Y⁰ = copy(D)
    const norm² = svdvals(Y⁰)[1] # can be tuned
    const norm∞ = maximum(abs(Y⁰)) / λ
    const dual_norm = max(norm², norm∞)
    const d_norm = norm(D)
    Y⁰ /= dual_norm

    μ⁰ = 1.25 / norm²
    const μ̄ = μ⁰ * 1.0e+7

    converged = false
    epoch = 0
    sv⁰ = 10

    Yᵏ, Aᵏ, Eᵏ, μᵏ, svᵏ = Y⁰, A⁰, E⁰, μ⁰, sv⁰
    while !converged
        # update sparse matrix E
        temp_T = D - Aᵏ + μᵏ^-1 * Yᵏ
        Eᵏ = max(temp_T - λ / μᵏ, 0) + min(temp_T + λ / μᵏ, 0)

        # force non-negative
        Eᵏ = max(Eᵏ, 0) # heuristic
        U, S, V = svd(D - Eᵏ + μᵏ^-1 * Yᵏ)

        # trancate dimention
        svpᵏ = int(sum(S .> 1 / μᵏ))
        if svpᵏ < svᵏ
            svᵏ = min(svpᵏ + 1, N);
        else
            svᵏ = min(svpᵏ + round(0.05 * N), N)
        end

        # update A
        Aᵏ = U[:,1:svpᵏ] * diagm(S[1:svpᵏ] - μᵏ^-1) * V[:,1:svpᵏ]'

        # force non-negative
        Aᵏ = max(Aᵏ, 0)

        Z = D - Aᵏ - Eᵏ;
        Yᵏ = Yᵏ + μᵏ * Z;
        μᵏ = min(μᵏ * ρ, μ̄)

        objective = norm(Z) / d_norm;
        if verbose
            println("#$(epoch) objective: $(objective)")
        end

        if objective < error_tol
            if verbose
                println("converged")
            end
            converged = true
        end
        
        epoch += 1
        if epoch >= max_iter
            break
        end
    end

    return Aᵏ, Eᵏ
end
