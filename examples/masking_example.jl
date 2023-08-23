using LinearAlgebra
using Flux
using Flux:unsqueeze
using StatsBase
using CUDA

include("../src/layers.jl")
include("../src/rotational_utils.jl")

N_frames = 7
dim = 32

mask = right_to_left_mask(N_frames) |> gpu 

S = Float32.(randn(dim,N_frames,batch)) |> gpu
T = (get_rotation(N_frames,batch), get_translation(N_frames,batch)) |> gpu

ipma = IPA(IPA_settings(dim)) |> gpu
si = ipma(T,S,mask=mask)