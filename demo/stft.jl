# STFT/ISTFT

function blackman(n::Integer)
    const a0, a1, a2 = 0.42, 0.5, 0.08
    t = 2*pi/(n-1)
    [a0 - a1*cos(t*k) + a2*cos(t*k*2) for k=0:n-1]
end

function hanning(n::Integer)
    [0.5*(1-cos(2*pi*k/(n-1))) for k=0:n-1]
end

# countframes returns the number of frames that will be processed.
function countframes{T<:Number}(x::Vector{T}, framelen::Int, hopsize::Int)
    div(length(x) - framelen, hopsize) + 1
end

# splitframes performs overlapping frame splitting.
function splitframes{T<:Number}(x::Vector{T}, 
                                framelen::Int=1024,
                                hopsize::Int=framelen/2)
    const N = countframes(x, framelen, hopsize)
    frames = Array(eltype(x), framelen, N)

    for i=1:N
        frames[:,i] = x[(i-1)*hopsize+1:(i-1)*hopsize+framelen]
    end

    return frames
end

# stft performs the Short-Time Fourier Transform (STFT) for real signals.
function stft{T<:Real}(x::Vector{T}, 
                       framelen::Int=1024,
                       hopsize::Int=div(framelen,2),
                       window=hanning(framelen))
    frames = splitframes(x, framelen, hopsize)

    const freqbins = div(framelen, 2) + 1
    spectrogram = Array(Complex64, freqbins, size(frames,2))
    for i=1:size(frames,2)
        spectrogram[:,i] = rfft(frames[:,i] .* window)
    end

    return spectrogram
end

# istft peforms the Inverse STFT to recover the original signal from STFT 
# coefficients.
function istft{T<:Complex}(spectrogram::Matrix{T},
                           framelen::Int=1024,
                           hopsize::Int=div(framelen,2),
                           window=hanning(framelen))
    const numframes = size(spectrogram, 2)

    expectedlen = framelen + (numframes-1)*hopsize
    reconstructed = zeros(expectedlen)
    windowsum = zeros(expectedlen)
    const windowsquare = window .* window

    # Overlapping addition
    for i=1:numframes
        s, e = (i-1)*hopsize+1, (i-1)*hopsize+framelen
        r = irfft(spectrogram[:,i], framelen)
        reconstructed[s:e] += r .* window
        windowsum[s:e] += windowsquare
    end

    # Normalized by window
    for i=1:endof(reconstructed)
        # avoid zero division
        if windowsum[i] > 1.0e-7
            reconstructed[i] /= windowsum[i]
        end
    end

    return reconstructed
end
