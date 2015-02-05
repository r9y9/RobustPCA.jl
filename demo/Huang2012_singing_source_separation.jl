# Singing voice separation demo using Robust PCA

using WAV
using RobustPCA

include(joinpath(dirname(@__FILE__), "stft.jl"))

filename = "titon_2_07_SNR5.wav"
x, fs = wavread(joinpath(dirname(@__FILE__), filename))
x = vec(x)

framelen = 1024
hopsize = div(framelen, 4)
win = hanning(framelen)
Xᶜ = stft(x, framelen, hopsize, win)

X, P = abs(Xᶜ), angle(Xᶜ)
@show size(X)

# performs RPCA
# A: row-rank matrix
# S: sparse matrix
elapsed_rpca = @elapsed begin
    s = 1.0/sqrt(maximum(size(X)))
    A, E = inexact_alm_rpca(X; verbose=true, error_tol=1.0e-7,
                            sparseness=s)
end
println("Elapsed time in Robust PCA: $(elapsed_rpca) sec.")

# Frequency masking
const binary_mask = true
if binary_mask
    m = abs(E) .> abs(A)
    Emask = zeros(size(E))
    Emask[m] = X[m]
    Amask = X - Emask
else
    Emask = E
    Amask = A
end

# back to time domain
a = istft(Amask .* exp(im * P), framelen, hopsize, win)
e = istft(Emask .* exp(im * P), framelen, hopsize, win)

name, suffix = splitext(filename)
wavwrite(float(a), string(name, "_A", suffix), Fs=fs)
wavwrite(float(e), string(name, "_E", suffix), Fs=fs)
