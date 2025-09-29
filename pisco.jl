using LinearAlgebra
using FLoops

function pisco_smaps(kdata;
    # PISCO tecnique flags
    kernel_shape=1, # (0 for rect, 1 for circle)
    fft_C_mtx=1, # option to approximate ChC with FFTs
    sketched_SVD=1, # option to used sketched (randomized) SVD
    subspace_itr_G=0, # option to use subspace iteration to compute nullspace vectors of G

    # PISCO parameters
    τ=Int(3), # neighborhood size (radius)
    N_cal=32, # size of calibration region
    σ_thresh=0.002, # threshold for singular values
    d_sk=50, # sketch dimension for SVD of ChC (overestimation of the rank)
    N_gzp=24, # number of vo/pixels to interpolate (zero-pad) in each dimension of G matrix
    α=100, # Gaussian window parameter for phase normalization
    L=1, # number of sensitivity map sets to estimate

    # other options
    verbose=0 # option to print out debug info
)

    # get sizes
    nd = ndims(kdata) - 1 # number of image dimensions
    N = size(kdata)[1:nd] # image size
    Q = size(kdata, nd + 1) # number of channels
    T = eltype(kdata) # data type

    # extract calibration data from kcal
    cal_sidx = center_idcs(N, N_cal * ones(nd)) # subscript indices for calibration region (vector of vectors for each dim)
    kcal = kdata[cal_sidx..., :]

    # create kernel neighborhood
    Λ_cidx = grid([-τ:τ for d in 1:nd]...) # coordinate indicies for kernel neighborhood
    if kernel_shape == 1
        cmask = vec(sum(Λ_cidx .^ 2, dims=2)) .<= τ^2
        Λ_cidx = Λ_cidx[cmask, :] # mask out edges if using ellipsoidal kernel
    end
    Λ_len = size(Λ_cidx, 1) # final kernel (patch) size

    # print update
    if verbose == 1
        println("PISCO parameters:")
        println("\tkernel shape: ", kernel_shape == 0 ? "rectangular" : "circular")
        println("\tkernel radius (τ): ", τ)
        println("\tcalibration region size: ", N_cal)
        println("\tnumber of coils: ", Q)
        println("\tnumber of kernel points: ", Λ_len)
        println("\tfft_C_mtx: ", fft_C_mtx == 0 ? "no" : "yes")
        println("\tsketch_SVD: ", sketched_SVD == 0 ? "no" : "yes")
        println("\tsubspace_itr_G: ", subspace_itr_G == 0 ? "no" : "yes")
        if sketched_SVD == 1
            println("\tsketch dimension (d_sk): ", d_sk)
        end
        println("\tσ_thresh: ", σ_thresh)
        println("\tN_gzp: ", N_gzp)
        println("\tα (Gaussian window parameter): ", α)
        println("\tL (number of sensitivity maps to estimate): ", L)
        println("forming convolution gram matrix ChC... ")
    end

    # form the convolution gram matrix ChC
    t0_ChC = time();
    if fft_C_mtx == 0 # naive approach
        C = C_matrix(kcal, Λ_cidx, N_cal) # calculate convolution matrix C explicitly
        ChC = C' * C # calculate ChC via matrix product
    else # direct FFT-based approach
        ChC = ChC_matrix_fft(kcal, Λ_cidx, N_cal)
    end
    tend_ChC = time();

    # print update with computation time
    t_ChC = tend_ChC - t0_ChC
    if verbose == 1
        println("done.")
        println("\tcomputation time: ", t_ChC, " seconds")
        println("computing nullspace of ChC... ")
    end

    # compute the null space of ChC
    t0_nullspace_ChC = time();
    if sketched_SVD == 0 # full SVD of ChC
        (~, σ, V) = svd(ChC)
    else # sketched SVD of ChC
        S_sk = 1 / sqrt(d_sk) * (randn(d_sk, Λ_len * Q) + 1im * randn(d_sk, Λ_len * Q))
        (~, σ, V) = svd(S_sk * ChC)
    end
    r = count(σ / σ[1] .> σ_thresh)
    Vr = V[:, 1:r] # get the column space basis of C
    W = I - Vr * Vr' # get null space projection matrix
    tend_nullspace_ChC = time();

    # print update with computation time
    t_nullspace_ChC = tend_nullspace_ChC - t0_nullspace_ChC
    if verbose == 1
        println("done.")
        println("\tcomputation time: ", t_nullspace_ChC, " seconds")
        println("\testimated rank(ChC): ", r)
        println("forming G matrix... ")
    end

    # calculate the G matrix
    t0_G = time();
    G = G_matrix(W, Λ_cidx, N_cal, N_gzp) # form the G matrix
    G = reshape(G, (N_cal + N_gzp)^nd, Q, Q) # reshape G into (vo/pixels) x Q x Q array
    G = permutedims(G, (2,3,1))
    tend_G = time();

    # print update with computation time
    t_G = tend_G - t0_G
    if verbose == 1
        println("done.")
        println("\tcomputation time: ", t_G, " seconds")
        println("estimating sensitivity maps from nullspace of G... ")
    end

    # estimate sensitivity maps from G
    t0_nullspace_G = time();
    if subspace_itr_G == 1
        let G_local = G
            result_smaps = Vector{Array{T}}(undef, (N_cal + N_gzp)^nd)
            result_λ = Vector{Vector{T}}(undef, (N_cal + N_gzp)^nd)
            @floop ThreadedEx() for x in 1:(N_cal+N_gzp)^nd # loop through vo/pixels
                σ,V = subspace_iteration((@view G_local[:, :, x]), L; maxit=30, tol=1e-6)
                result_smaps[x] = V[:, end-L+1:end]
                result_λ[x] = σ
            end
            smaps_lores = Array{T}(undef, Q, L, (N_cal + N_gzp)^nd)
            λ_lores = Array{T}(undef, L, (N_cal + N_gzp)^nd)
            for x in 1:(N_cal + N_gzp)^nd
                smaps_lores[:, :, x] .= result_smaps[x]
                λ_lores[:, x] .= result_λ[x]
            end
        end
    else # SVD
        let G_local = G
            result_smaps = Vector{Array{T}}(undef, (N_cal + N_gzp)^nd)
            result_λ = Vector{Vector{T}}(undef, (N_cal + N_gzp)^nd)
            @floop ThreadedEx() for x in 1:(N_cal+N_gzp)^nd # loop through vo/pixels
                res = svd((@view G_local[:, :, x]))
                result_smaps[x] = res.V[:, end-L+1:end]
                result_λ[x] = res.S
            end
            smaps_lores = Array{T}(undef, Q, L, (N_cal + N_gzp)^nd)
            λ_lores = Array{T}(undef, Q, (N_cal + N_gzp)^nd)
            for x in 1:(N_cal + N_gzp)^nd
                smaps_lores[:, :, x] .= result_smaps[x]
                λ_lores[:, x] .= result_λ[x]
            end
        end
    end
    smaps_lores = permutedims(reshape(smaps_lores, Q, L, (N_cal + N_gzp) * ones(Int, nd)...), ((3:nd+2)..., 1, 2))
    λ_lores = permutedims(reshape(λ_lores, Q, (N_cal + N_gzp) * ones(Int, nd)...), ((2:nd+1)..., 1))
    tend_nullspace_G = time();

    # print update with computation time
    t_nullspace_G = tend_nullspace_G - t0_nullspace_G
    if verbose == 1
        println("done.")
        println("\tcomputation time: ", t_nullspace_G, " seconds")
        println("normalizing phase of the sensitivity maps... ")
    end

    # create nd apodizing window for phase normalization
    apodizing_window = ones(T, (N_cal + N_gzp) * ones(Int, nd)...)
    for d in 1:nd
        shape = ntuple(i -> i == d ? (N_cal + N_gzp) : 1, nd)
        apodizing_window .*= reshape(gausswin(N_cal + N_gzp; α), shape)
    end

    # get low-resolution image data
    ft_idata_lores = zero_pad(kcal; N=((N_cal + N_gzp) * ones(Int, nd)..., Q))
    idata_lores = iftnd(ft_idata_lores .* apodizing_window; dims=1:nd)

    # normalize the phase of the sensitivity maps based on the low-resolution image data
    cim = sum(conj.(smaps_lores) .* idata_lores; dims=nd + 1) ./ sum(abs2.(smaps_lores); dims=nd + 1)
    phase_norm = sign.(cim)
    smaps_lores_corr = smaps_lores .* phase_norm # apply phase normalization

    # print update
    if verbose == 1
        println("done.")
        println("interpolating sensitivity maps... ")
    end

    # create nd hanning window for fft interpolation
    w_sm = ones(T, (N_cal + N_gzp) * ones(Int, nd)...)
    for d in 1:nd
        shape = ntuple(i -> i == d ? (N_cal + N_gzp) : 1, nd)
        w_sm .*= reshape(hanningwin(N_cal + N_gzp), shape)
    end

    # interpolate sensitivity maps using fft
    smaps = iftnd(ftnd(smaps_lores_corr; dims=1:nd) .* w_sm; dims=1:nd, N=(N..., Q, L))
    λ = iftnd(ftnd(λ_lores; dims=1:nd) .* w_sm; dims=1:nd, N=(N..., Q))

    # print update
    if verbose == 1
        println("done.")
        println("total time: ", t_ChC + t_nullspace_ChC + t_G + t_nullspace_G, " seconds")
    end

    return smaps, λ
end

function C_matrix(kcal, Λ_cidx, N_cal)

    # get sizes and parameters
    nd = ndims(kcal) - 1; # number of spatial dimensions
    Q = size(kcal, nd+1); # number of coils
    Λ_len = size(Λ_cidx, 1); # number of kernel points
    τ = floor(Int, maximum(abs.(Λ_cidx[:]))); # kernel radius in each dimension
    T = eltype(kcal) # data type

    # calculate convolution matrix C
    C = Array{T}(undef, (N_cal .- 2*τ .- even_RL.(N_cal))^nd, Λ_len, Q);
    k_cidx = grid(ntuple(d -> τ+1+even_RL(N_cal):N_cal-τ, nd)...); # grid of shift points
    for (i, row) in enumerate(eachrow(k_cidx))
        idcs = ntuple(d -> row[d] .+ Λ_cidx[:, d], nd); # shifted indicies
        C[i,:,:] .= getindex.(Ref(kcal), idcs..., 1:Q);
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
    T = eltype(kcal) # data type

    # precompute ft of zero-padded s_q for q = 1,...Q
    N_pad = 2^(ceil(log2(N_cal + 2 * τ)))
    ρ = ftnd(kcal, N=(Int.(N_pad * ones(nd))..., Q), dims=1:nd)

    # columns of each C_p^H C_q block
    pad_cidx = grid(ntuple(_ -> -floor(Int, N_pad / 2):floor(Int, N_pad / 2)-even_RL(N_pad / 2), nd)...) # linear indices for padded calibration region
    patch_idcs = floor(Int, N_pad / 2) + 1 .+ Λ_cidx # subscript indices for filling patches in ChC matrix
    φ = cis.(-2 * pi * (Λ_cidx / N_pad) * pad_cidx') # phase kernel for applying shifts in fourier domain
    ChC_blocks = Array{T}(undef, Λ_len, Λ_len, Q, Q) # preallocate ChC blocks
    @floop ThreadedEx() for (q, p) in Iterators.flatmap(q -> ((q, p) for p in q:Q), 1:Q)
        # compute ft(s_p[n] ⊗ conj(s_q[-n])) = ρ_p* * ρ_q
        ρρ_pq = conj.(ρ[ntuple(_ -> Colon(), nd)..., p]) .* ρ[ntuple(_ -> Colon(), nd)..., q]

        # calculate δ[n - n_i] ⊗ s_p[n] ⊗ conj(s_q[-n])
        φρρ_ipq = reshape(conj.(φ'), (Int.(N_pad*ones(nd))..., Λ_len)) .* ρρ_pq # φ_i * conj(ρ_p) * ρ_q: i in (nd+1)th dimension
        δss_ipq = iftnd(φρρ_ipq; dims=1:nd) # δ[n - n_i] ⊗ s_p[n] ⊗ conj(s_q[-n]) = ift(φ_i * conj(ρ_p) * ρ_q)

        # extract patch and write to ChC pq block
        ChC_blocks[:,:,p,q] .= (@view δss_ipq[CartesianIndex.(Tuple.(eachrow(patch_idcs))), :, :])

        # Hermitian symmetry
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
    T = eltype(W) # data type

    # reshape W into blocks for placing into G
    W_blocks = permutedims(reshape(W, Λ_len, Q, Λ_len, Q), (1, 2, 4, 3)) # Λ_len x Q x Q x Λ_len

    # get target indices for placing W blocks into G
    G_target_sidx = ntuple(d -> (2 * τ + 1) + 1 .+ Λ_cidx[end:-1:1, d] .+ Λ_cidx[:, d]', nd) # matrix of subscript indicies for each dim
    G_target_lidx = LinearIndices(ones([2 * (2 * τ + 1) for d in 1:nd]...))[map(CartesianIndex, G_target_sidx...)] # matrix of linear indicies

    # form the fourier transform of G(x)
    grid_size = 2 * (2 * τ + 1)
    ft_G = zeros(T, (grid_size)^nd, Q, Q)
    for s in 1:Λ_len
        ft_G[G_target_lidx[s, :], :, :] .+= W_blocks[:, :, :, s]
    end
    ft_G = reshape(ft_G, grid_size * ones(Int, nd)..., Q, Q)

    # create checkerboard modulation pattern for G
    mod_pattern = ones(T, grid_size * ones(Int, nd)...)
    for d in 1:nd
        shape = ntuple(i -> i == d ? grid_size : 1, nd)
        mod_pattern .*= reshape((-1) .^ (0:(grid_size-1)), shape)
    end

    # create phase kernel (why?)
    gzp_sidx = ntuple(d -> -floor(Int, (N_cal + N_gzp) / 2):floor(Int, (N_cal + N_gzp) / 2)-even_RL((N_cal + N_gzp) / 2), nd) # subscript indices for G zero-pad region
    gzp_cidx = grid(gzp_sidx...) # coordinate indices for G zero-pad region
    φ = -cis.(-2 * pi * (gzp_cidx / (N_cal + N_gzp)) * ((N_cal + N_gzp) - (2 * τ + 1)) * ones(nd)) # keep an eye on the sign
    φ = reshape(φ, (N_cal + N_gzp) * ones(Int, nd)...)

    # compute ft of modulated G
    ft_G_zp = zero_pad(conj.(ft_G) .* mod_pattern; N=((N_cal + N_gzp) * ones(Int, nd)..., Q, Q))
    G = 1/Λ_len * fft(ft_G_zp, 1:nd) .* φ

    return G
end