using Statistics, Distances
using Flux
using Diffusions
using LinearAlgebra

include("rotational_utils.jl")

"""
Taken from the documentation of Flux. Allows one input to be split into multiple outputs in the context of Flux networks. 
"""
struct Split{T}
    paths::T
end

Split(paths...) = Split(paths)
Flux.@functor Split
(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)

struct IPA
    settings::NamedTuple
    layers::NamedTuple
end

Flux.@functor IPA # provides parameter collection, gpu movement and more

function IPA(   dims::Integer,
                c::Integer,
                N_head::Integer,
                N_query_points::Integer,
                N_point_values::Integer,
                ;
                c_z = 0, 
                Typ = Float32,
            )
    if c_z == 0
        pairwise = false
    else
        pairwise = true
    end
    settings = (
        dims = dims,
        c = c,
        N_head = N_head,
        N_query_points = N_query_points,
        N_point_values = N_point_values, 
        c_z = c_z,
        pairwise = pairwise,
        Typ = Typ
    )
    if pairwise
        pair = Chain(Dense(c_z => N_head))
        pair_linear = Chain(Dense(N_head*c_z + N_head*c + 4*N_head*N_point_values => dims),
                 LayerNorm(dims)
                 )
    else
        pair = nothing
        pair_linear = nothing
    end
    layers = (
            ipa_j = Chain(
                # output in the order: q_i^h, k_i^h, v_i^h, q_i^hp, k_i^hp, v_i^hp 
                Split(
                        Dense(dims => c*N_head, bias = false), # q_i^h
                        Dense(dims => c*N_head, bias = false), # k_i^h
                        Dense(dims => c*N_head, bias = false), # v_i^h
                        Dense(dims => 3*N_head*N_query_points, bias = false), # q_i^hp
                        Dense(dims => 3*N_head*N_query_points, bias = false), # k_i^hp
                        Dense(dims => 3*N_head*N_point_values, bias = false)  # v_i^hp
                    ),
                ),
            ipa_linear = Chain(
                Dense(N_head*c + 4*N_head*N_point_values => dims),
                 LayerNorm(dims)
                 ),
            backbone_update = Chain(
                    Dropout(0.1),
                    LayerNorm(dims), 
                    Dense(dims=> dims, relu), 
                    Dense(dims => dims, relu),
                    Dense(dims => dims),
                    Dropout(0.1),
                    LayerNorm(dims), 
                    Dense(dims => 6)  
                ),
            pair = pair,
            pair_linear = pair_linear,
            gamma_h = ones(Typ, N_head) .* Typ(0.541)
            )
    
    return IPA(settings, layers)
end

function (ipa::IPA)(si::AbstractArray,Ti::Tuple{AbstractArray,AbstractArray})
    l = ipa.layers
    c, N_head, N_query_points, N_point_values, c_z = ipa.settings.c, ipa.settings.N_head, ipa.settings.N_query_points, ipa.settings.N_point_values, ipa.settings.c_z
    pairwise, Typ = ipa.settings.pairwise, ipa.settings.Typ
    rot_Ti, translate_Ti = Ti
    gamma_h = softmax(l.gamma_h)

    # Get relevant parameters from our ipa struct. 
    paramtahs = l.ipa_j(si)

    #c_z = mod.c_z
    num_blobs = size(si,2)

    # Constants described in the supplementary paper. 
    w_C = Typ(sqrt(2/(9*N_query_points)))
    w_L = Typ(sqrt(1/3))
    
    qh = reshape(paramtahs[1],(c,N_head,num_blobs,:)) 
    kh = reshape(paramtahs[2],(c,N_head,num_blobs,:)) 
    vh = reshape(paramtahs[3],(c,N_head,num_blobs,:)) 
    qhp = reshape(paramtahs[4],(3,N_head*N_query_points,num_blobs,:)) 
    khp = reshape(paramtahs[5],(3,N_head*N_query_points,num_blobs,:)) 
    vhp = reshape(paramtahs[6],(3,N_head*N_point_values,num_blobs,:)) 
    
    #=
    # Performing a trick to get all i,j permutations of qih^T kjh. 
    qih = stack([qh[:,:,ceil(Int, i/num_blobs),:] for i in 1:num_blobs^2],dims = 3)
    kjh = stack([kh[:,:,i % num_blobs + 1,:] for i in 0:num_blobs^2 - 1],dims = 3)
    qhTkh = reshape(sum(qih .*kjh, dims = 1),(N_head,num_blobs,num_blobs,:)) # the first num_blobs represents our i's and the second represents our j's 
    =#

    #This is much faster. But also try the strategy used for the Tiqihp here instead? Should be similar I expect.
    qhT = permutedims(qh, (3, 1, 2, 4))
    kh = permutedims(kh, (1, 3, 2, 4))
    qhTkh = permutedims(batched_mul(qhT,kh),(3,2,1,4)) #Why isn't this (3,1,2,4)??

    # Applying our transformations to the queries, keys, and values. 
    Tqhp = reshape(T_R3(qhp, rot_Ti,translate_Ti),3,N_head,N_query_points,num_blobs,:) 
    Tkhp = reshape(T_R3(khp, rot_Ti,translate_Ti),3,N_head,N_query_points,num_blobs,:)
    Tvhp = T_R3(vhp, rot_Ti, translate_Ti)

    #Tiqihp = reshape(stack([Tqhp[:,:,:,ceil(Int, i/num_blobs),:] for i in 1:num_blobs^2],dims = 5),(3,N_head,N_query_points,num_blobs^2,:))
    #Tjkjhp = reshape(stack([Tkhp[:,:,:,i % num_blobs + 1,:] for i in 0:num_blobs^2 - 1],dims = 5),(3,N_head,N_query_points,num_blobs^2,:))

    #Faster version:
    Tiqihp = Tqhp[:,:,:,repeat(1:num_blobs, inner=num_blobs),:]
    Tjkjhp = Tkhp[:,:,:,repeat(1:num_blobs, outer=num_blobs),:]

    norm_arg = Tiqihp .- Tjkjhp
    norm = reshape(sqrt.(sum(norm_arg.^2,dims = 1)),N_head,N_query_points,num_blobs,num_blobs,:)
    sum_norm = reshape(sum(norm, dims = 2),N_head,num_blobs,num_blobs,:)

    att_arg = reshape(w_L*(1/Typ(sqrt(c)) .* qhTkh .- w_C/2 .* gamma_h .* sum_norm),(N_head,num_blobs,num_blobs, :))

    if pairwise
        bij = reshape(l.pair(zij),(N_head,num_blobs,num_blobs,:))
        att = Flux.softmax(att_arg .+ w_L.*bij, dims = 3) 
    else
        att = Flux.softmax(att_arg, dims = 3)
    end

    # Applying the attention weights to the values.
    broadcast_att_oh = reshape(att,(1,N_head,num_blobs,num_blobs,:))
    broadcast_vh = reshape(vh, (c,N_head,num_blobs,1,:))
    oh = reshape(sum(broadcast_att_oh .* broadcast_vh,dims = 4), c,N_head,num_blobs,:)

    broadcast_att_ohp = reshape(att,(1,N_head,1,num_blobs,num_blobs,:))
    broadcast_tvhp = reshape(Tvhp,(3,N_head,N_point_values,1,num_blobs,:))

    ohp_r = reshape(sum(broadcast_att_ohp.*broadcast_tvhp,dims=5),3,N_head*N_point_values,num_blobs,:)
    ohp = T_R3_inv(ohp_r, rot_Ti, translate_Ti)
    normed_ohp = sqrt.(sum(ohp.^2,dims = 1))

    catty = vcat(
        reshape(oh, N_head*c, num_blobs,:),
        reshape(ohp, 3*N_head*N_point_values, num_blobs,:),
        reshape(normed_ohp, N_head*N_point_values, num_blobs,:)
        )
    
    if pairwise
        broadcast_zij = reshape(zij,(c_z,1,num_blobs,num_blobs,:))
        broadcast_att_zij = reshape(att,(1,N_head,num_blobs,num_blobs,:))
        obh = sum(broadcast_zij .* broadcast_att_zij,dims = 4)
        catty = vcat(catty, reshape(obh, N_head*c_z, num_blobs,:))
    end
    
    si = si .+ l.ipa_linear(catty)
    backbone_update = l.backbone_update(si) 
    arr = reshape(backbone_update,3,2,size(si,2),:) 
    bcds = reshape(arr[:,1,:,:],3,:) 
    quats = bcds2flatquats(bcds) 
    rm = rotmatrix_from_quat(quats) 
    T_new = (
        reshape(rm,3,3,size(si,2),:),
        reshape(arr[:,2,:,:],3,1,size(si,2),:)
            )
    T = T_T(Ti,T_new)

    return si,T
end