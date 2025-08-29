using LinearAlgebra
using FFTW

# even/odd indexing function
function even_RL(x)
    return Int(1 - mod(x, 2.0)); # helps with using odd data points
end

# function to return center M indicies of array with size N
function center_idcs(N, M)
    nd = length(N)
    center_idx = ntuple(d -> floor(Int, N[d] / 2) + even_RL(N[d]), nd)
    M_idcs = ntuple(d -> (-floor(Int, M[d] / 2):floor(Int, M[d] / 2) - even_RL(M[d])), nd)
    return ntuple(d -> center_idx[d] .+ M_idcs[d], nd)
end

# gaussian window function (reflects matlab gausswin.m)
function gausswin(N; α=0.5)
    L = N - 1
    n = (0:L) .- L / 2
    return exp.(-0.5 * (α * n / (L / 2)) .^ 2)
end

# hanning window function
function hanningwin(N; α=1)
    if N == 1
        return 1.0
    else
        n = 0:(N - 1)
        return 0.54 .- 0.46 * cos.(α * 2 * π * n / (N - 1))
    end
end

# function to generate grid points
function grid(xs::AbstractVector...)
    nd = length(xs)
    N = prod(length.(xs))
    xgrd = zeros(eltype(xs[1]), N, nd)
    for d in 1:nd
        shape = ntuple(j -> j == d ? length(xs[d]) : 1, nd)
        rep = ntuple(j -> j == d ? 1 : length(xs[d]), nd)
        xgrd[:, d] = repeat(reshape(xs[d], shape), rep...)[:]
    end
    return xgrd
end

# function to zero pad array x to size N
function zero_pad(x; N=size(x))
    sz = size(x);
    x_pad = zeros(eltype(x), N)
    idcs = center_idcs(N, sz)
    x_pad[idcs...] .= x
    return x_pad
end

# function to compute full fourier transform with shifts
function ftnd(x; N=size(x), dims=1:ndims(x))
    x = zero_pad(x; N=N)
    return fftshift(fft(ifftshift(x, dims), dims), dims)
end

# function to compute full inverse fourier transform with shifts
function iftnd(X; N=size(X), dims=1:ndims(X))
    X = zero_pad(X; N=N);
    return fftshift(ifft(ifftshift(X, dims), dims), dims)
end