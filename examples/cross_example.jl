using LinearAlgebra
using StatsBase
using Flux
using CUDA

include("../src/layers.jl")
include("../src/rotational_utils.jl")

len_L, len_R, dim, batch = 10,10,128,5
T_L = (get_rotation(len_L, batch), randn(3,len_L, batch)) 
T_R = (get_rotation(len_R, batch), randn(3,len_R, batch))
S_L = randn(dim,len_L, batch)
S_R = randn(dim,len_R, batch)
T_L, S_L, T_R, S_R = (T_L, S_L, T_R, S_R) |> gpu

# For structure module: 
#ipca = IPCrossAStructureModuleLayer(IPA_settings(dim, N_query_points = 7)) |> gpu
#ipca(T_L, S_L, T_R, S_R) 

# For layer: 
ipca = IPCrossA(IPA_settings(dim, N_query_points = 7)) |> gpu

# For self layer: 
# ipa = IPA(IPA_settings(dim, N_query_points = 7)) |> gpu
params = Flux.params(ipca)
for i in 1:1000
    loss, grads = Flux.withgradient(Flux.params(ipca)) do
        # For structure model output is T,s ; for layer output is s 
        s = ipca(T_L, S_L, T_L, S_L)
        mean(s)
    end 
end

CUDA.memory_status() # ≈ 100% GPU memory usage, and doesn't go down after stopping. 