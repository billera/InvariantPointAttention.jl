using Pkg; Pkg.activate(".")
using Revise
using InvariantPointAttention
using InvariantPointAttention: get_rotation, get_translation
using InvariantPointAttention: T_T
using InvariantPointAttention:sumdrop


batch_size = 3
framesL = 100
framesR = 101
dim = 768


siL = Float32.(randn(dim,framesL,batch_size)) 
siR = Float32.(randn(dim,framesR,batch_size)) 

T_locL = (get_rotation(framesL,batch_size), get_translation(framesL,batch_size)) 
T_locR = (get_rotation(framesR,batch_size), get_translation(framesR,batch_size)) 
# Get 1 global SE(3) transformation for each batch.
T_glob = (get_rotation(batch_size), get_translation(batch_size)) 

T_GlobL = (stack([T_glob[1] for i in 1:framesL],dims = 3), stack([T_glob[2] for i in 1:framesL],dims = 3)) 

T_GlobR = (stack([T_glob[1] for i in 1:framesR],dims = 3), stack([T_glob[2] for i in 1:framesR],dims = 3))

T_newL = T_T(T_GlobL,T_locL)
T_newR = T_T(T_GlobR,T_locR)

ipa = IPAStructureModuleLayer(IPA_settings(dim; use_softmax1 = true)) #|> gpu

T_loc, si_loc = ipa(T_locL,siL, T_locR, siR)
T_glob, si_glob = ipa(T_newL, siL, T_newR, siR)

invariance_error = maximum(abs.(si_glob .- si_loc)) # ≈ 2f-6