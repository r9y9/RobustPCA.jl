using PyCall

if false
    @pyimport scipy.sparse.linalg as splinalg
end

# rearange_scipy_sparse_svds converts a result of scipy.sparse.linalg.svds to
# that of the julia svd.
function rearange_scipy_sparse_svds!(U, S, V)
    const M, K = size(U)
    V = V'
    for k=1:K
        U[:, k] = U[:, end+1-k]
        V[:, k] = V[:, end+1-k]
    end
    reverse!(S)

    return U, S, V
end

# Soft thresholding function
soft_threshold(x, ϵ::Float64) = max(x - ϵ, 0) + min(x - ϵ, 0)

# RPCA using the inexact Augumented Lagrange Multiplier (ALM).
# Given a observation matrix D, find row-rank matrix A and sparse matrix E
# so that D = A + E.
function inexact_alm_rpca(D::AbstractMatrix;
                          max_iter::Int=1000, error_tol::Float64=1.0e-7,
                          ρ::Float64=1.5, verbose::Bool=false,
                          nonnegativeA::Bool=true, nonnegativeE::Bool=true)
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
    k = 0
    sv⁰ = 10

    Yᵏ, Aᵏ, Eᵏ, μᵏ, svᵏ = Y⁰, A⁰, E⁰, μ⁰, sv⁰
    while !converged
        # update sparse matrix E
        Eᵏ = soft_threshold(D - Aᵏ + μᵏ^-1 * Yᵏ, λ * μᵏ^-1)
        # force non-negative (heuristic)
        if nonnegativeE
            Eᵏ = max(Eᵏ, 0)
        end

        U, S, V = svd(D - Eᵏ + μᵏ^-1 * Yᵏ)
        if false
            U, S, V = splinalg.svds(D - Eᵏ + μᵏ^-1 * Yᵏ, k=min(svᵏ, M-1, N-1))
            U, S, V = rearange_scipy_sparse_svds!(U, S, V)
        end

        # trancate dimention
        svpᵏ = int(sum(S .> μᵏ^-1))
        if svpᵏ < svᵏ
            svᵏ = min(svpᵏ + 1, N)
        else
            svᵏ = min(svpᵏ + round(0.05 * N), N)
        end

        # update row-rank matrix A
        Aᵏ = U[:,1:svpᵏ] * diagm(S[1:svpᵏ] - μᵏ^-1) * V[:,1:svpᵏ]'
        # force non-negative (heuristic)
        if nonnegativeA
            Aᵏ = max(Aᵏ, 0)
        end

        Z = D - Aᵏ - Eᵏ

        Yᵏ = Yᵏ + μᵏ * Z
        μᵏ = min(μᵏ * ρ, μ̄)

        objective = norm(Z) / d_norm
        if verbose
            println("#$(k) objective: $(objective)")
        end

        if objective < error_tol
            if verbose
                println("converged")
            end
            converged = true
        end
        
        k = k + 1
        if k >= max_iter
            break
        end
    end

    return Aᵏ, Eᵏ
end
