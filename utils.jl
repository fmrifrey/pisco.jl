using LinearAlgebra

# even/odd indexing function
function even_RL(x)
    return Int(1 - mod(x, 2.0)); # helps with using odd data points
end

# gaussian window function (reflects matlab gausswin.m)
function gausswin(N, α)
    L = N - 1
    n = (0:L) .- L / 2
    return exp.(-0.5 * (α * n / (L / 2)) .^ 2)
end

# hanning window function
function hanningwin(N)
    if N == 1
        return 1.0
    else
        n = 0:(N - 1)
        return 0.54 .- 0.46 * cos.(2 * π * n / (N - 1))
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