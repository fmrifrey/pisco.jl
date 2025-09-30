# load packages
using MIRTjim
using Plots
using FFTW
using Random
using LinearAlgebra
using MAT
using LinearMapsAA
using pisco

# read in data from mat file
kdata = matread("./data/3D_GRE_data.mat")["kData"];
N = size(kdata)[1:3]; # image dimensions
Q = size(kdata, 4); # number of coils

# correct for the z-offset in the data (only for this dataset)
nslices = N[3]
dz = -22 # slice offset
kdata .*= reshape(exp.(1im * 2π * dz * collect(center_idcs(0, nslices)...) / nslices), (1, 1, nslices, 1)); # apply linear phase

# get image space data
idata = iftnd(kdata; dims=1:3);

# set PISCO techniques
kernel_shape = 1; # (0 for rect, 1 for circle)
fft_C_mtx = true; # option to approximate ChC with FFTs
sketched_SVD = true; # option to used sketched (randomized) SVD
subspace_itr_G = false; # option to use subspace iteration to compute nullspace vectors of G
fft_interp = true; # option to use FFTs to interpolate G matrix

# set PISCO parameters
τ = Int(3); # neighborhood size (radius)
N_cal = (32,32,32); # size of calibration region
σ_thresh = 0.002; # threshold for singular values
d_sk = 50; # sketch dimension for SVD of ChC (overestimation of the rank)
N_gzp = 24; # number of vo/pixels to interpolate (zero-pad) in each dimension of G matrix
α = 100; # Gaussian window parameter for phase normalization
L = 1; # number of sensitivity map sets to estimate

# estimate the sensitivity maps by calling the pisco_smaps function
smaps, λ = pisco_smaps(kdata;
    kernel_shape=kernel_shape,
    fft_C_mtx=fft_C_mtx,
    sketched_SVD=sketched_SVD,
    subspace_itr_G=subspace_itr_G,
    fft_interp=fft_interp,
    τ=τ,
    N_cal=N_cal,
    N=N,
    σ_thresh=σ_thresh,
    d_sk=d_sk,
    N_gzp=N_gzp,
    α=α,
    L=L,
    verbose=true);

# create sensitivity encoding operator
S = LinearMapAA(
    x -> x .* smaps, # forward
    y -> sum(conj.(smaps) .* y, dims=4), # adjoint
    (prod(N) * Q, prod(N)); # operator size
    T=ComplexF64, # data type
    idim=N, # input dimensions
    odim=(N..., Q) # output dimensions
);

# solve for coil-combined image using conjugate gradient
idata_coil_combined = cg(S, zeros(ComplexF64, N...), idata; niter=30);

# get representation error (NPR)
@show npr = norm(idata[:] - (S*idata_coil_combined)[:]) / norm(idata[:]);