using LinearAlgebra
using StatsBase
using Flux

include("../src/layers.jl")
include("../src/rotational_utils.jl")

len_L, len_R, dim, batch = 5, 5, 4, 10
T_L = (get_rotation(len_L, batch), randn(Float32, 3, len_L, batch)) 
T_R = (get_rotation(len_R, batch), randn(Float32, 3, len_R, batch))
S_L = randn(Float32, dim, len_L, batch)
S_R = randn(Float32, dim, len_R, batch)
T_L, S_L, T_R, S_R = (T_L, S_L, T_R, S_R)

# For structure module: 
#ipca = IPCrossAStructureModuleLayer(IPA_settings(dim, N_query_points = 7))
#ipca(T_L, S_L, T_R, S_R) 

# For layer: 
ipca = IPCrossA(IPA_settings(dim, N_query_points = 7))

# For self layer: 
# ipa = IPA(IPA_settings(dim, N_query_points = 7))
params = Flux.params(ipca)
for i in 1:100
    loss, grads = Flux.withgradient(Flux.params(ipca)) do
        # For structure model output is T,s ; for layer output is s 
        s = ipca(T_L, S_L, T_R, S_R)
        sum(s) / length(s)
    end 
end
