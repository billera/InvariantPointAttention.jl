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
Returns a tuple of the IPA settings, with defaults for everything except dims. This can be passed to the IPA and IPAStructureModuleLayer.
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
Invariant Point Attention
"""
struct IPA
    settings::NamedTuple
    layers::NamedTuple
end

Flux.@functor IPA # provides parameter collection, gpu movement and more

function IPA(settings::NamedTuple)
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
    
    return IPA(settings, layers)
end

function (ipa::IPA)(Ti::Tuple{AbstractArray,AbstractArray}, si::AbstractArray; zij = nothing)
    # Get relevant parameters from our ipa struct.
    l = ipa.layers
    dims, c, N_head, N_query_points, N_point_values, c_z, Typ, pairwise = ipa.settings 
    rot_Ti, translate_Ti = Ti
    
    N_frames = size(si,2)
    gamma_h = softplus(l.gamma_h)

    # Constants described in the supplementary paper. 
    w_C = Typ(sqrt(2/(9*N_query_points)))
    w_L = Typ(sqrt(1/3))

    qh = reshape(l.proj_qh(si),(c,N_head,N_frames,:))
    kh = reshape(l.proj_kh(si),(c,N_head,N_frames,:))
    vh = reshape(l.proj_vh(si),(c,N_head,N_frames,:))
    qhp = reshape(l.proj_qhp(si),(3,N_head*N_query_points,N_frames,:))
    khp = reshape(l.proj_khp(si),(3,N_head*N_query_points,N_frames,:))
    vhp = reshape(l.proj_vhp(si),(3,N_head*N_point_values,N_frames,:))
    
    # Dot products between queries and keys.
    qhT = permutedims(qh, (3, 1, 2, 4))
    kh = permutedims(kh, (1, 3, 2, 4))
    qhTkh = permutedims(batched_mul(qhT,kh),(3,2,1,4))

    # Applying our transformations to the queries, keys, and values to put them in the global frame.
    Tqhp = reshape(T_R3(qhp, rot_Ti,translate_Ti),3,N_head,N_query_points,N_frames,:) 
    Tkhp = reshape(T_R3(khp, rot_Ti,translate_Ti),3,N_head,N_query_points,N_frames,:)
    Tvhp = T_R3(vhp, rot_Ti, translate_Ti)

    Tiqihp = Tqhp[:,:,:,repeat(1:N_frames, inner=N_frames),:]
    Tjkjhp = Tkhp[:,:,:,repeat(1:N_frames, outer=N_frames),:]

    norm_arg = Tiqihp .- Tjkjhp
    norm = reshape(sqrt.(sum(norm_arg.^2,dims = 1)),N_head,N_query_points,N_frames,N_frames,:)
    sum_norm = reshape(sum(norm, dims = 2),N_head,N_frames,N_frames,:)

    att_arg = reshape(w_L*(1/Typ(sqrt(c)) .* qhTkh .- w_C/2 .* gamma_h .* sum_norm),(N_head,N_frames,N_frames, :))

    if pairwise
        bij = reshape(l.pair(zij),(N_head,N_frames,N_frames,:))
        att = Flux.softmax(att_arg .+ w_L.*bij, dims = 3) 
    else
        att = Flux.softmax(att_arg, dims = 3)
    end

    # Applying the attention weights to the values.
    broadcast_att_oh = reshape(att,(1,N_head,N_frames,N_frames,:))

    #Maybe fixed a bug?
    #broadcast_vh = reshape(vh, (c,N_head,N_frames,1,:))
    broadcast_vh = reshape(vh, (c,N_head,1,N_frames,:))

    oh = reshape(sum(broadcast_att_oh .* broadcast_vh,dims = 4), c,N_head,N_frames,:)

    broadcast_att_ohp = reshape(att,(1,N_head,1,N_frames,N_frames,:))
    broadcast_tvhp = reshape(Tvhp,(3,N_head,N_point_values,1,N_frames,:))

    ohp_r = reshape(sum(broadcast_att_ohp.*broadcast_tvhp,dims=5),3,N_head*N_point_values,N_frames,:)
    ohp = T_R3_inv(ohp_r, rot_Ti, translate_Ti)
    normed_ohp = sqrt.(sum(ohp.^2,dims = 1))

    catty = vcat(
        reshape(oh, N_head*c, N_frames,:),
        reshape(ohp, 3*N_head*N_point_values, N_frames,:),
        reshape(normed_ohp, N_head*N_point_values, N_frames,:)
        )
    
    if pairwise
        broadcast_zij = reshape(zij,(c_z,1,N_frames,N_frames,:))
        broadcast_att_zij = reshape(att,(1,N_head,N_frames,N_frames,:))
        obh = sum(broadcast_zij .* broadcast_att_zij,dims = 4)
        catty = vcat(catty, reshape(obh, N_head*c_z, N_frames,:))
    end
    
    #Note: the skip connect happens outside of IPA itself!
    si = l.ipa_linear(catty)
    return si
end

"""
Partial Structure Module - single layer - from AF2. Not a faithful repro, and doesn't include the losses etc.
"""
struct IPAStructureModuleLayer
    settings::NamedTuple
    layers::NamedTuple
end
Flux.@functor IPAStructureModuleLayer
function IPAStructureModuleLayer(settings::NamedTuple; dropout_p = 0.1, af = Flux.relu)
    dims = settings.dims
    layers = (
        ipa = IPA(settings),
        ipa_norm = Chain(Dropout(dropout_p), LayerNorm(dims)),
        trans = Chain(Dense(dims => dims, af), Dense(dims => dims, af), Dense(dims => dims)),
        trans_norm = Chain(Dropout(dropout_p), LayerNorm(dims)),
        backbone = BackboneUpdate(dims),
    )
    return IPAStructureModuleLayer(settings, layers)
end
function (structuremodulelayer::IPAStructureModuleLayer)(Ti, si; zij = nothing)
    settings = structuremodulelayer.settings
    if settings.c_z > 0 && zij === nothing
        error("zij must be provided if c_z > 0")
    end

    l = structuremodulelayer.layers

    if zij === nothing
        si = si .+ l.ipa(Ti, si)
    else
        si = si .+ l.ipa(Ti, si, zij = zij)
    end
    si = l.ipa_norm(si)
    si = si .+ l.trans(si)
    si = l.trans_norm(si)
    Ti = l.backbone(Ti, si)
    return Ti, si
end



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
#Attention props from L (Keys, Values) to R (Queries).
#Because IPA uses Q'K, our pairwise matrices are R-by-L
function (ipa::IPCrossA)(TiL::Tuple{AbstractArray,AbstractArray}, siL::AbstractArray, TiR::Tuple{AbstractArray,AbstractArray}, siR::AbstractArray; zij = nothing)
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

    # Constants described in the supplementary paper. 
    w_C = Typ(sqrt(2/(9*N_query_points)))
    w_L = Typ(sqrt(1/3))

    qh = reshape(l.proj_qh(siR),(c,N_head,N_frames_R,:))
    kh = reshape(l.proj_kh(siL),(c,N_head,N_frames_L,:))
    vh = reshape(l.proj_vh(siL),(c,N_head,N_frames_L,:))
    qhp = reshape(l.proj_qhp(siR),(3,N_head*N_query_points,N_frames_R,:))
    khp = reshape(l.proj_khp(siL),(3,N_head*N_query_points,N_frames_L,:))
    vhp = reshape(l.proj_vhp(siL),(3,N_head*N_point_values,N_frames_L,:))
    
    #This should is Q'K, following IPA, which isn't like the regular QK'
    # Dot products between queries and keys.
                        #FramesR, c, N_head, Batch
    qhT = permutedims(qh, (3, 1, 2, 4))
                         #c, FramesR, N_head, Batch
    kh = permutedims(kh, (1, 3, 2, 4))
    qhTkh = permutedims(#FramesR, #FramesL, N_head, Batch
                        batched_mul(qhT,kh)
                        #N_head, FramesL, FramesR, Batch when we use (3,2,1,4) (original version)
                        #N_head, FramesR, FramesL, Batch when we use (3,1,2,4)
                        ,(3,1,2,4))

    #Result should be R-by-L
    #@show size(qhTkh)
    
    #=
    #Attention props from L (Keys, Values) to R (Queries). 
    L,R,c = 4,3,2
    K = randn(c, L)
    Q = randn(c, R)
    short = Q'K #3x4

    long = reshape(sum(Q[:,repeat(1:R,outer=L)] .* K[:,repeat(1:L, inner=R)],dims = 1), R, L)
    @show short .- long
    =#

    # Applying our transformations to the queries, keys, and values to put them in the global frame.
    Tqhp = reshape(T_R3(qhp, rot_TiR,translate_TiR),3,N_head,N_query_points,N_frames_R,:) 
    Tkhp = reshape(T_R3(khp, rot_TiL,translate_TiL),3,N_head,N_query_points,N_frames_L,:)
    Tvhp = T_R3(vhp, rot_TiL, translate_TiL)

                #3, heads, points, frames, batch
    Tiqihp = Tqhp[:,:,:,repeat(1:N_frames_R, outer=N_frames_L),:]
    Tjkjhp = Tkhp[:,:,:,repeat(1:N_frames_L, inner=N_frames_R),:]
    #The above is now correct, but will wind up with the opposite dim to the regular attention, because that got transposed.

    norm_arg = Tiqihp .- Tjkjhp

    #3,heads,points,framesR*framesL,1
    #@show size(norm_arg)

    norm = reshape(sqrt.(sum(norm_arg.^2,dims = 1)),N_head,N_query_points,N_frames_R,N_frames_L,:) #Euc between points
    #@show size(norm)

    sum_norm = reshape(sum(norm, dims = 2),N_head,N_frames_R,N_frames_L,:) #Sum over points for each head
    #@show size(sum_norm)
    
    att_arg = reshape(w_L*(1/Typ(sqrt(c)) .* qhTkh .- w_C/2 .* gamma_h .* sum_norm),(N_head,N_frames_R,N_frames_L, :))
    #@show size(att_arg)
    
    if pairwise
        bij = reshape(l.pair(zij),(N_head,N_frames_R,N_frames_L,:))
        att = Flux.softmax(att_arg .+ w_L.*bij, dims = 3) 
    else
        att = Flux.softmax(att_arg, dims = 3)
    end
    #@show size(att)

    # Applying the attention weights to the values.
    broadcast_att_oh = reshape(att,(1,N_head,N_frames_R,N_frames_L,:))
    #@show size(broadcast_att_oh)
    
    #I think this may have been some sort of bug before.
    #The "N_frames" was in the wrong dim, and previously this wasn't erroring because the dims were equal
    #Or maybe this is because we're using (3,1,2,4) insteasd of (3,2,1,4) above?
    #broadcast_vh = reshape(vh, (c,N_head,N_frames_L,1,:))
    broadcast_vh = reshape(vh, (c,N_head,1,N_frames_L,:))
    #@show size(broadcast_vh)

    oh = reshape(sum(broadcast_att_oh .* broadcast_vh,dims = 4), c,N_head,N_frames_R,:)
    #@show size(oh)

    broadcast_att_ohp = reshape(att,(1,N_head,1,N_frames_R,N_frames_L,:))
    #@show size(broadcast_att_ohp)

    broadcast_tvhp = reshape(Tvhp,(3,N_head,N_point_values,1,N_frames_L,:))
    #@show size(broadcast_tvhp)

    ohp_r = reshape(sum(broadcast_att_ohp.*broadcast_tvhp,dims=5),3,N_head*N_point_values,N_frames_R,:)
    #@show size(ohp_r)

    ohp = T_R3_inv(ohp_r, rot_TiR, translate_TiR) #ohp_r were in the global frame, so we put those back in the recipient local
    normed_ohp = sqrt.(sum(ohp.^2,dims = 1))
    #@show size(normed_ohp)

    catty = vcat(
        reshape(oh, N_head*c, N_frames_R,:),
        reshape(ohp, 3*N_head*N_point_values, N_frames_R,:),
        reshape(normed_ohp, N_head*N_point_values, N_frames_R,:)
        )
    
    if pairwise
        broadcast_zij = reshape(zij,(c_z,1,N_frames_R,N_frames_L,:))
        broadcast_att_zij = reshape(att,(1,N_head,N_frames_R,N_frames_L,:))
        obh = sum(broadcast_zij .* broadcast_att_zij,dims = 4)
        catty = vcat(catty, reshape(obh, N_head*c_z, N_frames_R,:))
    end
    
    #Note: the skip connect happens outside of IPA itself!
    si = l.ipa_linear(catty)
    return si 
end

#= Testing
len_L, len_R, dim = 4,5,8
T_L = (get_rotation(len_L), randn(3,len_L))
T_R = (get_rotation(len_R), randn(3,len_R))
S_L = randn(dim,len_L)
S_R = randn(dim,len_R)
T_L, S_L, T_R, S_R = (T_L, S_L, T_R, S_R) |> gpu
ipca = IPCrossA(IPA_settings(dim, N_query_points = 7)) |> gpu
ipca(T_L, S_L, T_R, S_R)

loss, grads = Flux.withgradient(params) do
    mean(ipca(T_L, S_L, T_R, S_R) .^ 2)
end
=#