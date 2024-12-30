using Pkg; Pkg.activate(".")
using InvariantPointAttention
using InvariantPointAttention: get_rotation

len_L, len_R, dim, batch = 10,10,128,5
T_L = (get_rotation(len_L, batch), randn(Float32, 3,len_L, batch)) 
T_R = (get_rotation(len_R, batch), randn(Float32, 3,len_R, batch))
S_L = randn(Float32, dim,len_L, batch)
S_R = randn(Float32, dim,len_R, batch)
T_L, S_L, T_R, S_R = (T_L, S_L, T_R, S_R) 

ipca = IPCrossA(IPA_settings(dim, N_query_points = 7, c = 16)) 
# rope dim is given by the query embedding dimension "c" and the max length of sequences is passed in the second argument. 
rope = IPARoPE(ipca.settings.c, 100)
# index the rope to the proper length when passing to the layer. 
a = ipca(T_L, S_L, T_L, S_L, rope = rope[1:10])
