using LinearAlgebra
using StatsBase
using Flux

include("../src/layers.jl")
include("../src/rotational_utils.jl")

N_frames = 7
dim = 32
batch_size = 1

mask = right_to_left_mask(N_frames)

S = Float32.(randn(dim, N_frames, batch_size))
T = (get_rotation(N_frames, batch_size), get_translation(N_frames, batch_size))

ipa = IPA(IPA_settings(dim))
si = ipa(T, S, mask=mask)