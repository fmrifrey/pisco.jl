using LinearAlgebra

function C_matrix(kcal, Λ_cidx, N_cal)

    # get sizes and parameters
    nd = ndims(kcal) - 1; # number of spatial dimensions
    Q = size(kcal, nd+1); # number of coils
    Λ_len = size(Λ_cidx, 1); # number of kernel points
    τ = floor(Int, maximum(abs.(Λ_cidx[:]))); # kernel radius in each dimension

    # calculate convolution matrix C
    C = zeros(ComplexF64, (N_cal .- 2*τ .- even_RL.(N_cal))^nd, Λ_len, Q);
    k_cidx = grid(ntuple(d -> τ+1+even_RL(N_cal):N_cal-τ, nd)...); # grid of shift points
    for (i, row) in enumerate(eachrow(k_cidx))
        idcs = ntuple(d -> row[d] .+ Λ_cidx[:, d], nd); # shifted indicies
        for q in 1:Q
            C[i,:,q] .= getindex.(Ref(kcal), idcs..., q);
        end
    end
    C = reshape(C, ((N_cal .- 2*τ .- even_RL.(N_cal))^nd, Λ_len*Q)); # reshape

    return C;
end

function ChC_matrix_fft(kcal, Λ_cidx, N_cal)

    # get sizes and parameters
    nd = ndims(kcal) - 1; # number of spatial dimensions
    Q = size(kcal, nd+1); # number of coils
    Λ_len = size(Λ_cidx, 1); # number of kernel points
    τ = floor(Int, maximum(abs.(Λ_cidx[:]))); # kernel radius in each dimension

    # precompute ft of zero-padded s_q for q = 1,...Q
    N_pad = 2^(ceil(log2(N_cal + 2 * τ)))
    ρ = ftnd(kcal, N=(Int.(N_pad * ones(nd))..., Q), dims=1:nd)

    # columns of each C_p^H C_q block
    ChC_blocks = zeros(ComplexF64, Λ_len, Λ_len, Q, Q)
    pad_cidx = grid(ntuple(_ -> -floor(Int, N_pad / 2):floor(Int, N_pad / 2)-even_RL(N_pad / 2), nd)...) # linear indices for padded calibration region
    patch_idcs = ntuple(d -> round(Int, N_pad / 2) + even_RL(N_pad / 2) .+ Λ_cidx[:, d], nd) # subscript indices for filling patches in ChC matrix
    φ = exp.(-1im * 2 * pi * (Λ_cidx / N_pad) * pad_cidx') # phase kernel for applying shifts in fourier domain
    for p in 1:Q, q in p:Q
        # compute ft(s_p[n] ⊗ conj(s_q[-n])) = ρ_p* * ρ_q
        ρρ_pq = conj(ρ[ntuple(_ -> Colon(), nd)..., p]) .* ρ[ntuple(_ -> Colon(), nd)..., q]

        for i in 1:Λ_len # loop through all shift indicies
            # calculate δ[n - n_i] ⊗ s_p[n] ⊗ conj(s_q[-n])
            φρρ_pqi = reshape(φ[i, :], size(ρρ_pq)) .* ρρ_pq # φ_i * conj(ρ_p) * ρ_q
            δss_pqi = iftnd(φρρ_pqi) # δ[n - n_i] ⊗ s_p[n] ⊗ conj(s_q[-n]) = ift(φ_i * conj(ρ_p) * ρ_q)

            # extract patch and write to ChC pq block
            ChC_blocks[:, i, p, q] .= getindex.(Ref(δss_pqi), patch_idcs...)
        end

        # fill qp block by exploiting hermitian symmetry
        if p != q
            ChC_blocks[:, :, q, p] .= ChC_blocks[:, :, p, q]'
        end

    end

    # reshape blocks
    ChC = reshape(permutedims(ChC_blocks, (1, 3, 2, 4)), (Λ_len * Q, Λ_len * Q))

    return ChC
end

function G_matrix(W, Λ_cidx, N_cal, N_gzp)

    # get sizes and parameters
    Λ_len = size(Λ_cidx, 1); # number of kernel points
    Q = Int.(size(W,1)/Λ_len); # number of coils
    nd = size(Λ_cidx, 2); # number of spatial dimensions
    τ = floor(Int, maximum(abs.(Λ_cidx[:]))); # kernel radius in each dimension

    # reshape W into blocks for placing into G
    W_blocks = permutedims(reshape(W, Λ_len, Q, Λ_len, Q), (1, 2, 4, 3)) # Λ_len x Q x Q x Λ_len

    # get target indices for placing W blocks into G
    G_target_sidx = ntuple(d -> (2 * τ + 1) + 1 .+ Λ_cidx[end:-1:1, d] .+ Λ_cidx[:, d]', nd) # matrix of subscript indicies for each dim
    G_target_lidx = LinearIndices(ones([2 * (2 * τ + 1) for d in 1:nd]...))[map(CartesianIndex, G_target_sidx...)] # matrix of linear indicies

    # form the fourier transform of G(x)
    grid_size = 2 * (2 * τ + 1)
    ft_G = zeros(ComplexF64, (grid_size)^nd, Q, Q)
    for s in 1:Λ_len
        ft_G[G_target_lidx[s, :], :, :] .+= W_blocks[:, :, :, s]
    end
    ft_G = reshape(ft_G, grid_size * ones(Int, nd)..., Q, Q)

    # create checkerboard modulation pattern for G
    mod_pattern = ones(Float64, grid_size * ones(Int, nd)...)
    for d in 1:nd
        shape = ntuple(i -> i == d ? grid_size : 1, nd)
        mod_pattern .*= reshape((-1) .^ (0:(grid_size-1)), shape)
    end

    # create phase kernel (why?)
    gzp_sidx = ntuple(d -> -floor(Int, (N_cal + N_gzp) / 2):floor(Int, (N_cal + N_gzp) / 2)-even_RL((N_cal + N_gzp) / 2), nd) # subscript indices for G zero-pad region
    gzp_cidx = grid(gzp_sidx...) # coordinate indices for G zero-pad region
    φ = -exp.(-1im * 2 * pi * (gzp_cidx / (N_cal + N_gzp)) * ((N_cal + N_gzp) - (2 * τ + 1)) * ones(nd)) # keep an eye on the sign
    φ = reshape(φ, (N_cal + N_gzp) * ones(Int, nd)...)

    # compute ft of modulated G
    ft_G_zp = zero_pad(conj.(ft_G) .* mod_pattern; N=((N_cal + N_gzp) * ones(Int, nd)..., Q, Q))
    G = 1/Λ_len * fft(ft_G_zp, 1:nd) .* φ

    return G
end