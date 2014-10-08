# Robust Principal Component Analysis (RPCA) using the inexact augumented
# Lagrange multiplier.
# Given a observation matrix D, find row-rank matrix A and sparse matrix E
# so that D = A + E.
function rpca_inexact_alm(D::AbstractMatrix;
                          max_iter::Int=1000, error_tol::Float64=1.0e-7,
                          ρ::Float64=1.5, verbose::Bool=false)
    const M, N = size(D)
    const λ = 1 / sqrt(M)

    A, E = zeros(M, N), zeros(M, N)
    
    # initialize
    Y = copy(D)
    const norm² = svdvals(Y)[1] # can be tuned
    const norm∞ = maximum(abs(Y)) / λ
    const dual_norm = max(norm², norm∞)
    const d_norm = norm(D)
    Y /= dual_norm

    μ = 1.25 / norm²
    const μ̄ = μ * 1.0e+7

    converged = false
    epoch = 0
    total_svd = 0
    sv = 10
    while !converged
        # update sparse matrix E
        temp_T = D - A + (1.0 / μ) * Y;
        E = max(temp_T - λ / μ, 0) + min(temp_T + λ / μ, 0)

        # force non-negative
        E = max(E, 0) # heuristic
        U, S, V = svd(D - E + 1.0 / μ * Y)

        # trancate dimention
        svp = int(sum(S .> 1 / μ))
        if svp < sv
            sv = min(svp + 1, N);
        else
            sv = min(svp + 0.05 * N + 0.5, N)
        end

        # update A
        S_th = diagm(S[1:svp] - 1.0 / μ)
        A = U[:,1:svp] * S_th * V[:,1:svp]'

        # force non-negative
        A = max(A, 0)

        total_svd += 1;
        Z = D - A - E;
        Y = Y + μ * Z;
        μ = min(μ * ρ, μ̄)

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

    return A, E
end
