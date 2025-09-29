# load packages
using MIRTjim
using Plots
using FFTW
using Random
using LinearAlgebra
using MAT
using LinearMapsAA

# load helper functions
include("pisco.jl");
include("utils.jl");

# read in data from mat file
kdata = ComplexF32.(1e8 * matread("./data/2D_T1_data.mat")["kData"]);

# get sizes
nd = ndims(kdata) - 1; # number of image dimensions
N = size(kdata)[1:nd]; # image size
Q = size(kdata, nd + 1); # number of channels

# show the multi-channel image data
idata = iftnd(kdata; dims=1:nd);

# set PISCO techniques
kernel_shape = 1; # (0 for rect, 1 for circle)
fft_C_mtx = 1; # option to approximate ChC with FFTs
sketched_SVD = 1; # option to used sketched (randomized) SVD
subspace_itr_G = 0; # option to use subspace iteration to compute nullspace vectors of G

# set PISCO parameters
τ = Int(3); # neighborhood size (radius)
N_cal = 32; # size of calibration region
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
    σ_thresh=σ_thresh,
    d_sk=d_sk,
    N_gzp=N_gzp,
    α=α,
    L=L,
    verbose=1);

# create sensitivity encoding operator
S = LinearMapAA(
    x -> x.*smaps, # forward
    y -> sum(conj.(smaps) .* y, dims=nd+1), # adjoint
    (prod(N)*Q, prod(N)); # operator size
    T=ComplexF64, # data type
    idim=N, # input dimensions
    odim=(N..., Q) # output dimensions
    );

# solve for coil-combined image using conjugate gradient
idata_coil_combined = cg(S, zeros(ComplexF64, N...), idata; niter=30);

# get representation error (NPR)
@show npr = norm(idata[:] - (S * idata_coil_combined)[:]) / norm(idata[:]);