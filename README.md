# RobustPCA

[![Build Status](https://travis-ci.org/r9y9/RobustPCA.jl.svg?branch=master)](https://travis-ci.org/r9y9/RobustPCA.jl)

RobustPCA.jl provides a support for Robust Principal Component Analysis (RPCA) in Julia language. This implementation basically follows the following paper: 

- [Lin, Zhouchen, Minming Chen, and Yi Ma. "The augmented lagrange multiplier method for exact recovery of corrupted low-rank matrices." arXiv preprint arXiv:1009.5055 (2010)](http://arxiv.org/pdf/1009.5055.pdf).

## Install

```julia
Pkg.clone("https://github.com/r9y9/RobustPCA.jl")
```

## Applications

### Singing-voice separation from monaural recordings

- [P.S Huang, S.D. Chen, P. Smaragdis, and M. HasegawaJohnson, "Singing-voice separation from monaural recordings using robust principal component analysis", Proc. ICASSP 2012.](http://www.mirlab.org/conference_papers/International_Conference/ICASSP%202012/pdfs/0000057.pdf)

```julia
cd demo
julia Huang2012_singing_source_separation.jl
```
