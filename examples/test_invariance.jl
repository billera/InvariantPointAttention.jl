include("../src/attention.jl")

batch_size = 32
frames = 64
dim = 768

si = Float32.(randn(dim,frames,batch_size)) |> gpu

T_loc = (get_rotation(frames,batch_size), get_translation(frames,batch_size)) |> gpu

# Get 1 global SE(3) transformation for each batch.
T_glob = (get_rotation(batch_size), get_translation(batch_size)) 

# Replicating the global SE(3) transformation for each frame so it can be applied to each frame's local SE(3) transformation.
T_glob = (stack([T_glob[1] for i in 1:frames],dims = 3), stack([T_glob[2] for i in 1:frames],dims = 3)) |> gpu
T_new = T_T(T_glob,T_loc)

# Define IPA block. 
ipa = IPA(dim, 16, 12, 4, 8) |> gpu

si_loc, tmp = ipa(si, T_loc)
si_glob, tmp = ipa(si, T_new)

invariance_error = maximum(abs.(si_glob .- si_loc)) 

@show invariance_error # ≈ 2f-6