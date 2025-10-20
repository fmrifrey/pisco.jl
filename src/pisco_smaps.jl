using FLoops
using LinearAlgebra
using Revise

# export all functions
export pisco_smaps, C_matrix, ChC_matrix_fft, G_matrix

function pisco_smaps(kcal;
    # PISCO tecnique flags
    kernel_shape=1, # (0 for rect, 1 for circle)
    fft_C_mtx=true, # option to approximate ChC with FFTs
    sketched_SVD=true, # option to used sketched (randomized) SVD
    subspace_itr_G=false, # option to use subspace iteration to compute nullspace vectors of G
    fft_interp=true, # option to use fft interpolation for smaps

    # PISCO parameters
    τ=Int(3), # neighborhood size (radius)
    N_cal=nothing, # size of calibration region (nothing=use full size of kcal)
    N=nothing, # size of smaps (nothing=same as kcal)
    σ_thresh=0.002, # threshold for singular values
    d_sk=50, # sketch dimension for SVD of ChC (overestimation of the rank)
    N_gzp=24, # number of vo/pixels to interpolate (zero-pad) in each dimension of G matrix
    α=100, # Gaussian window parameter for phase normalization
    L=1, # number of sensitivity map sets to estimate

    # other options
    verbose=true # option to print out debug info
)

    # get sizes
    nd = ndims(kcal) - 1 # number of image dimensions
    if isnothing(N_cal)
        N_cal = size(kcal)[1:nd] # use full size of kcal if N_cal not provided
    else
        # extract calibration data from kcal
        cal_sidx = center_idcs(size(kcal)[1:nd], N_cal); # subscript indices for calibration region (vector of vectors for each dim)
        kcal = kcal[cal_sidx..., :];
    end
    Q = size(kcal, nd + 1) # number of channels
    T = eltype(kcal) # data type

    # set N if not provided
    if isnothing(N)
        N = N_cal
    end

    # create kernel neighborhood
    Λ_cidx = grid([-τ:τ for d in 1:nd]...) # coordinate indicies for kernel neighborhood
    if kernel_shape == 1
        cmask = vec(sum(Λ_cidx .^ 2, dims=2)) .<= τ^2
        Λ_cidx = Λ_cidx[cmask, :] # mask out edges if using ellipsoidal kernel
    end
    Λ_len = size(Λ_cidx, 1) # final kernel (patch) size

    # print update
    if verbose
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
    if fft_C_mtx # direct fft computation of ChC
        ChC = ChC_matrix_fft(kcal, Λ_cidx, N_cal)
    else # naive computation of ChC
        C = C_matrix(kcal, Λ_cidx, N_cal) # calculate convolution matrix C explicitly
        ChC = C' * C # calculate ChC via matrix product
    end
    tend_ChC = time();

    # print update with computation time
    t_ChC = tend_ChC - t0_ChC
    if verbose
        println("done.")
        println("\tcomputation time: ", t_ChC, " seconds")
        println("computing nullspace of ChC... ")
    end

    # compute the null space of ChC
    t0_nullspace_ChC = time();
    if sketched_SVD # full SVD of ChC
        S_sk = 1 / sqrt(d_sk) * (randn(d_sk, Λ_len * Q) + 1im * randn(d_sk, Λ_len * Q))
        (~, σ, V) = svd(S_sk * ChC)
    else # sketched SVD of ChC
        (~, σ, V) = svd(ChC)
    end
    r = count(σ / σ[1] .> σ_thresh)
    Vr = V[:, 1:r] # get the column space basis of C
    W = I - Vr * Vr' # get null space projection matrix
    tend_nullspace_ChC = time();

    # print update with computation time
    t_nullspace_ChC = tend_nullspace_ChC - t0_nullspace_ChC
    if verbose
        println("done.")
        println("\tcomputation time: ", t_nullspace_ChC, " seconds")
        println("\testimated rank(ChC): ", r)
        println("forming G matrix... ")
    end

    # calculate the G matrix
    t0_G = time();
    G = G_matrix(W, Λ_cidx, N_cal, N_gzp) # form the G matrix
    G = reshape(G, prod(N_cal .+ N_gzp), Q, Q) # reshape G into (vo/pixels) x Q x Q array
    G = permutedims(G, (2,3,1))
    tend_G = time();

    # print update with computation time
    t_G = tend_G - t0_G
    if verbose
        println("done.")
        println("\tcomputation time: ", t_G, " seconds")
        println("estimating sensitivity maps from nullspace of G... ")
    end

    # estimate sensitivity maps from G
    t0_nullspace_G = time();
    if subspace_itr_G
        let G_local = G
            result_smaps = Vector{Array{T}}(undef, prod(N_cal .+ N_gzp))
            result_λ = Vector{Vector{T}}(undef, prod(N_cal .+ N_gzp))
            @floop ThreadedEx() for x in 1:prod(N_cal .+ N_gzp) # loop through vo/pixels
                σ,V = subspace_iteration((@view G_local[:, :, x]), L; maxit=30, tol=1e-6)
                result_smaps[x] = V[:, end-L+1:end]
                result_λ[x] = σ
            end
            smaps_lores = Array{T}(undef, Q, L, prod(N_cal .+ N_gzp))
            λ_lores = Array{T}(undef, L, prod(N_cal .+ N_gzp))
            for x in 1:prod(N_cal .+ N_gzp)
                smaps_lores[:, :, x] .= result_smaps[x]
                λ_lores[:, x] .= result_λ[x]
            end
        end
    else # SVD
        let G_local = G
            result_smaps = Vector{Array{T}}(undef, prod(N_cal .+ N_gzp))
            result_λ = Vector{Vector{T}}(undef, prod(N_cal .+ N_gzp))
            @floop ThreadedEx() for x in 1:prod(N_cal .+ N_gzp) # loop through vo/pixels
                res = svd((@view G_local[:, :, x]))
                result_smaps[x] = res.V[:, end-L+1:end]
                result_λ[x] = res.S
            end
            smaps_lores = Array{T}(undef, Q, L, prod(N_cal .+ N_gzp))
            λ_lores = Array{T}(undef, Q, prod(N_cal .+ N_gzp))
            for x in 1:prod(N_cal .+ N_gzp)
                smaps_lores[:, :, x] .= result_smaps[x]
                λ_lores[:, x] .= result_λ[x]
            end
        end
    end
    smaps_lores = permutedims(reshape(smaps_lores, Q, L, (N_cal .+ N_gzp)...), ((3:nd+2)..., 1, 2))
    λ_lores = permutedims(reshape(λ_lores, Q, (N_cal .+ N_gzp)...), ((2:nd+1)..., 1))
    tend_nullspace_G = time();

    # print update with computation time
    t_nullspace_G = tend_nullspace_G - t0_nullspace_G
    if verbose
        println("done.")
        println("\tcomputation time: ", t_nullspace_G, " seconds")
        println("normalizing phase of the sensitivity maps... ")
    end

    # create nd apodizing window for phase normalization
    apodizing_window = ones(T, (N_cal .+ N_gzp)...)
    for d in 1:nd
        shape = ntuple(i -> i == d ? (N_cal[d] + N_gzp) : 1, nd)
        apodizing_window .*= reshape(gausswin(N_cal[d] + N_gzp; α), shape)
    end

    # get low-resolution image data
    ft_idata_lores = zero_pad(kcal; N=((N_cal .+ N_gzp)..., Q))
    idata_lores = iftnd(ft_idata_lores .* apodizing_window; dims=1:nd)

    # normalize the phase of the sensitivity maps based on the low-resolution image data
    cim = sum(conj.(smaps_lores) .* idata_lores; dims=nd + 1) ./ sum(abs2.(smaps_lores); dims=nd + 1)
    phase_norm = sign.(cim)
    smaps_lores_corr = smaps_lores .* phase_norm # apply phase normalization

    # print update
    if verbose
        println("done.")
        println("interpolating sensitivity maps... ")
    end

    if fft_interp
        # create nd hanning window for fft interpolation
        w_sm = ones(T, (N_cal .+ N_gzp)...)
        for d in 1:nd
            shape = ntuple(i -> i == d ? (N_cal[d] + N_gzp) : 1, nd)
            w_sm .*= reshape(hanningwin(N_cal[d] + N_gzp), shape)
        end

        # interpolate sensitivity maps using fft
        smaps = iftnd(ftnd(smaps_lores_corr; dims=1:nd) .* w_sm; dims=1:nd, N=(N..., Q, L))
        λ = iftnd(ftnd(λ_lores; dims=1:nd) .* w_sm; dims=1:nd, N=(N..., Q))
    else
        smaps = smaps_lores_corr
        λ = λ_lores
    end

    # print update
    if verbose
        println("done.")
        println("total time: ", t_ChC + t_nullspace_ChC + t_G + t_nullspace_G, " seconds")
    end

    return smaps, λ, σ
end

function C_matrix(kcal, Λ_cidx, N_cal)

    # get sizes and parameters
    nd = ndims(kcal) - 1; # number of spatial dimensions
    Q = size(kcal, nd+1); # number of coils
    Λ_len = size(Λ_cidx, 1); # number of kernel points
    τ = floor(Int, maximum(abs.(Λ_cidx[:]))); # kernel radius in each dimension
    T = eltype(kcal) # data type

    # calculate convolution matrix C
    C = Array{T}(undef, prod(N_cal .- 2*τ .- even_RL.(N_cal)), Λ_len, Q);
    k_cidx = grid(ntuple(d -> τ+1+even_RL(N_cal[d]):N_cal[d]-τ, nd)...); # grid of shift points
    for (i, row) in enumerate(eachrow(k_cidx))
        idcs = ntuple(d -> row[d] .+ Λ_cidx[:, d], nd); # shifted indicies
        C[i,:,:] .= getindex.(Ref(kcal), idcs..., 1:Q);
    end
    C = reshape(C, prod(N_cal .- 2*τ .- even_RL.(N_cal)), Λ_len*Q); # reshape

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
    N_pad = Int.(2 .^(ceil.(log2.(N_cal .+ 2 * τ))))
    ρ = ftnd(kcal, N=(N_pad...,Q), dims=1:nd)
    patch_cidx = ntuple(d -> vec(floor(Int, N_pad[d] / 2) + 1 .- Λ_cidx[:,d]' .+ Λ_cidx[:,d]), nd) # subscript indices for filling patches in ChC matrix
    patch_lidx = LinearIndices(ones(Int, N_pad...))[map(CartesianIndex, patch_cidx...)] # linear indices for filling patches in ChC matrix
    ChC_blocks = Array{T}(undef, Λ_len, Λ_len, Q, Q) # preallocate ChC blocks
    @floop ThreadedEx() for q in 1:Q # loop through coil pairs

        # compute ft(s_p[n] ⊗ conj(s_q[-n])) = ρ_p* * ρ_q
        ρρ_pq = conj.(ρ[ntuple(_ -> Colon(), nd)..., q:Q]) .* ρ[ntuple(_ -> Colon(), nd)..., q]
        ss_pq = iftnd(ρρ_pq; dims=1:nd) # s_p[n] ⊗ conj(s_q[-n])

        # reshape ss_pq into 2D array for indexing
        b = reshape(ss_pq, :, Q-q+1)[patch_lidx, :]

        # extract patch and write to ChC pq block
        ChC_blocks[:,:,q:Q,q] .= reshape(b, Λ_len, Λ_len, Q-q+1)
        ChC_blocks[:,:,q,q+1:Q] .= permutedims(conj.(ChC_blocks[:,:,q+1:Q,q]), (2,1,3)) # Hermitian symmetry

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
    gzp_sidx = ntuple(d -> -floor(Int, (N_cal[d] + N_gzp) / 2):floor(Int, (N_cal[d] + N_gzp) / 2)-even_RL((N_cal[d] + N_gzp) / 2), nd) # subscript indices for G zero-pad region
    gzp_cidx = grid(gzp_sidx...) # coordinate indices for G zero-pad region
    φ = -cis.(-2 * pi * (gzp_cidx ./ [(N_cal .+ N_gzp)...]') .* ([(N_cal .+ N_gzp)...]' .- (2 * τ + 1)) * ones(nd)) # keep an eye on the sign
    φ = reshape(φ, (N_cal .+ N_gzp)...)

    # compute ft of modulated G
    ft_G_zp = zero_pad(conj.(ft_G) .* mod_pattern; N=((N_cal .+ N_gzp)..., Q, Q))
    G = Array{T}(undef, (N_cal .+ N_gzp)..., Q, Q)
    let φ_local = φ
        @floop for q in 1:Q
            G[ntuple(_ -> Colon(), nd+1)...,q] = 1/Λ_len * fft(ft_G_zp[ntuple(_ -> Colon(), nd+1)...,q], 1:nd) .* φ_local
        end
    end

    return G
end