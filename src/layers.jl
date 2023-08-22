"""
Projects the frame embedding => 6, and uses this to transform the input frames.
"""
struct BackboneUpdate
    layers::NamedTuple
end
Flux.@functor BackboneUpdate
function BackboneUpdate(s_dim::Int)
    layers = (
        linear = Dense(s_dim => 6), 
    )
    return BackboneUpdate(layers)
end
function (backboneupdate::BackboneUpdate)(Ti, si)
    bu = backboneupdate.layers.linear(si)
    arr = reshape(bu,3,2,size(si,2),:) 
    T = update_frame(Ti, arr)
    return T
end

"""
Returns a tuple of the IPA settings, with defaults for everything except dims. This can be passed to the IPA and IPCrossAStructureModuleLayer.
"""
IPA_settings(
    dims;
    c = 16,
    N_head = 12,
    N_query_points = 4,
    N_point_values = 8,
    c_z = 0,
    Typ = Float32
) = (
    dims = dims,
    c = c,
    N_head = N_head,
    N_query_points = N_query_points,
    N_point_values = N_point_values,
    c_z = c_z,
    Typ = Typ,
    pairwise = c_z > 0
)


"""
Invariant Point Cross Attention (IPCrossA). Information flows from L (Keys, Values) to R (Queries).
"""
struct IPCrossA
    settings::NamedTuple
    layers::NamedTuple
end


Flux.@functor IPCrossA # provides parameter collection, gpu movement and more

function IPCrossA(settings::NamedTuple)
    dims, c, N_head, N_query_points, N_point_values, c_z, Typ, pairwise = settings 
    # Needs a slighyly unusual initialization - hat-tip: Kenta
    init = Flux.kaiming_uniform(gain = 1.0)
    if pairwise
        pair = Dense(c_z => N_head, bias = false; init)
        ipa_linear = Dense(N_head*c_z + N_head*c + 4*N_head*N_point_values => dims)
    else
        pair = nothing
        ipa_linear = Dense(N_head*c + 4*N_head*N_point_values => dims)
    end
    layers = (
            proj_qh = Dense(dims => c*N_head, bias = false; init),
            proj_kh = Dense(dims => c*N_head, bias = false; init),
            proj_vh = Dense(dims => c*N_head, bias = false; init),
            proj_qhp = Dense(dims => 3*N_head*N_query_points, bias = false; init),
            proj_khp = Dense(dims => 3*N_head*N_query_points, bias = false; init),
            proj_vhp = Dense(dims => 3*N_head*N_point_values, bias = false; init),
            ipa_linear = ipa_linear,
            pair = pair,
            gamma_h = ones(Typ, N_head) .* Typ(0.541)
            )
    
    return IPCrossA(settings, layers)
end



"""
Strictly Self-IPA initialization
"""
# We could skip making this struct and just have it be a cross IPA struct. 
struct IPA
    settings::NamedTuple
    layers::NamedTuple
end

Flux.@functor IPA

function IPA(settings::NamedTuple)
    crossL = IPCrossA(settings)
    return IPA(crossL.settings, crossL.layers)
end

"""
Self-IPA can be run from both IPA and cross IPA, allowing for flexibility. Simply calls cross IPA on itself. 
"""
function (ipa::Union{IPA, IPCrossA})(T::Tuple{AbstractArray,AbstractArray}, S::AbstractArray; Z = nothing)
    return ipa(T, S, T, S; zij = Z)
end


#Attention props from L (Keys, Values) to R (Queries).
#Because IPA uses Q'K, our pairwise matrices are R-by-L
function (ipa::Union{IPCrossA, IPA})(TiL::Tuple{AbstractArray,AbstractArray}, siL::AbstractArray, TiR::Tuple{AbstractArray,AbstractArray}, siR::AbstractArray; zij = nothing)
    if zij != nothing
        #This is assuming the dims of zij are c, N_frames_L, N_frames_R, batch
        @assert size(zij,2) == size(siR,2)
        @assert size(zij,3) != size(siL,2)
    end
    
    # Get relevant parameters from our ipa struct.
    l = ipa.layers
    dims, c, N_head, N_query_points, N_point_values, c_z, Typ, pairwise = ipa.settings 
    
    rot_TiL, translate_TiL = TiL
    rot_TiR, translate_TiR = TiR
    
    N_frames_L = size(siL,2)
    N_frames_R = size(siR,2)

    gamma_h = softplus(l.gamma_h)

    w_C = Typ(sqrt(2/(9*N_query_points)))
    dim_scale = Typ(1/sqrt(c))

    qh = reshape(l.proj_qh(siR),(c,N_head,N_frames_R,:))
    kh = reshape(l.proj_kh(siL),(c,N_head,N_frames_L,:))
    vh = reshape(l.proj_vh(siL),(c,N_head,N_frames_L,:))
    qhp = reshape(l.proj_qhp(siR),(3,N_head*N_query_points,N_frames_R,:))
    khp = reshape(l.proj_khp(siL),(3,N_head*N_query_points,N_frames_L,:))
    vhp = reshape(l.proj_vhp(siL),(3,N_head*N_point_values,N_frames_L,:))
    
    # This should be Q'K, following IPA, which isn't like the regular QK'
    # Dot products between queries and keys.
                        #FramesR, c, N_head, Batch
    qhT = permutedims(qh, (3, 1, 2, 4))
                         #c, FramesR, N_head, Batch
    kh = permutedims(kh, (1, 3, 2, 4))
    qhTkh = permutedims(#FramesR, #FramesL, N_head, Batch
                        batched_mul(qhT,kh)
                        #N_head, FramesR, FramesL, Batch when we use (3,1,2,4)
                            ,(3,1,2,4))
    
    # Applying our transformations to the queries, keys, and values to put them in the global frame.
    Tqhp = reshape(T_R3(qhp, rot_TiR,translate_TiR),3,N_head,N_query_points,N_frames_R,:) 
    Tkhp = reshape(T_R3(khp, rot_TiL,translate_TiL),3,N_head,N_query_points,N_frames_L,:)
    Tvhp = T_R3(vhp, rot_TiL, translate_TiL)

    diffs_glob = unsqueeze(Tqhp, dims = 5) .- unsqueeze(Tkhp, dims = 4)
    sum_norms_glob = reshape(sum(diffs_glob.^2, dims = [1,3]),N_head,N_frames_R,N_frames_L,:) #Sum over points for each head
    

    att_arg = reshape(dim_scale .* qhTkh .- w_C/2 .* gamma_h .* sum_norms_glob,(N_head,N_frames_R,N_frames_L, :))
    if pairwise
        w_L = Typ(sqrt(1/3))
        bij = reshape(l.pair(zij),(N_head,N_frames_R,N_frames_L,:))
        att = Flux.softmax(w_L .* (att_arg .+ bij), dims = 3) 
    else
        w_L = Typ(sqrt(1/2))
        att = Flux.softmax(w_L .* att_arg, dims = 3)
    end

    # Applying the attention weights to the values.
    broadcast_att_oh = reshape(att,(1,N_head,N_frames_R,N_frames_L,:))
    broadcast_vh = reshape(vh, (c,N_head,1,N_frames_L,:))

    oh = reshape(sum(broadcast_att_oh .* broadcast_vh,dims = 4), c,N_head,N_frames_R,:)

    broadcast_att_ohp = reshape(att,(1,N_head,1,N_frames_R,N_frames_L,:))

    broadcast_tvhp = reshape(Tvhp,(3,N_head,N_point_values,1,N_frames_L,:))

    ohp_r = reshape(sum(broadcast_att_ohp.*broadcast_tvhp,dims=5),3,N_head*N_point_values,N_frames_R,:)

    #ohp_r were in the global frame, so we put those back in the recipient local
    ohp = T_R3_inv(ohp_r, rot_TiR, translate_TiR) 
    normed_ohp = sqrt.(sum(ohp.^2,dims = 1))

    catty = vcat(
        reshape(oh, N_head*c, N_frames_R,:),
        reshape(ohp, 3*N_head*N_point_values, N_frames_R,:),
        reshape(normed_ohp, N_head*N_point_values, N_frames_R,:)
        )
    
    if pairwise
        broadcast_zij = reshape(zij,(c_z,1,N_frames_R,N_frames_L,:))
        broadcast_att_zij = reshape(att,(1,N_head,N_frames_R,N_frames_L,:))
        obh = sum(broadcast_zij .* broadcast_att_zij, dims = 4)
        catty = vcat(catty, reshape(obh, N_head*c_z, N_frames_R,:))
    end
    
    si = l.ipa_linear(catty)
    return si 
end

"""
Cross IPA Partial Structure Module initialization - single layer - adapted from AF2. From left to right. 
"""
struct IPCrossAStructureModuleLayer
    settings::NamedTuple
    layers::NamedTuple
end
Flux.@functor IPCrossAStructureModuleLayer
function IPCrossAStructureModuleLayer(settings::NamedTuple; dropout_p = 0.1, af = Flux.relu)
    dims = settings.dims
    layers = (
        ipa = IPCrossA(settings),
        ipa_norm = Chain(Dropout(dropout_p), LayerNorm(dims)),
        trans = Chain(Dense(dims => dims, af), Dense(dims => dims, af), Dense(dims => dims)),
        trans_norm = Chain(Dropout(dropout_p), LayerNorm(dims)),
        backbone = BackboneUpdate(dims),
    )
    return IPCrossAStructureModuleLayer(settings, layers)
end

"""
Self IPA Partial Structure Module initialization - single layer - adapted from AF2. 
"""

# We could skip making this struct and just have it be a cross IPA struct. 
struct IPAStructureModuleLayer
    settings::NamedTuple
    layers::NamedTuple
end
Flux.@functor IPAStructureModuleLayer

function IPAStructureModuleLayer(settings::NamedTuple)
    crossL = IPCrossAStructureModuleLayer(settings)
    return IPAStructureModuleLayer(crossL.settings, crossL.layers)
end
function (structuremodulelayer::Union{IPAStructureModuleLayer, IPCrossAStructureModuleLayer} )(T, S; zij = nothing)
    return structuremodulelayer(T, S, T, S; zij = zij)
end

"""
Cross IPA Partial Structure Module - single layer - adapted from AF2. From left to right. 
"""
function (structuremodulelayer::Union{IPCrossAStructureModuleLayer, IPAStructureModuleLayer})(T_L, S_L, T_R, S_R; zij = nothing)
    settings = structuremodulelayer.settings
    if settings.c_z > 0 && zij === nothing
        error("zij must be provided if c_z > 0")
    end

    l = structuremodulelayer.layers
    S_R = S_R .+ l.ipa(T_L, S_L,T_R,S_R, zij = zij)
    S_R = l.ipa_norm(S_R)
    S_R = S_R .+ l.trans(S_R)
    S_R = l.trans_norm(S_R)
    T_R = l.backbone(T_R, S_R)
    return T_R, S_R
end
