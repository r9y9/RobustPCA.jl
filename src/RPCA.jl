module RPCA

# Robust Principal Component Analysis (RPCA) in Julia

# Reference:
# Lin, Zhouchen, Minming Chen, and Yi Ma. "The augmented lagrange multiplier
# method for exact recovery of corrupted low-rank matrices." arXiv preprint
# arXiv:1009.5055 (2010)
# http://arxiv.org/pdf/1009.5055.pdf)

export inexact_alm_rpca

include("inexact_alm.jl")

end # module
