# taking from https://github.com/MurrellGroup/Jjama3.jl/blob/main/src/model.jl
# todo: re-write to avoid the O(N) permutedims. 
struct RoPE{A<:AbstractArray}
    cos::A
    sin::A
end
Base.getindex(rope::RoPE, i) = RoPE(rope.cos[:,i,:,:], rope.sin[:,i,:,:])

function Base.getindex(rope::RoPE, pos_matrix::AbstractMatrix{<:Integer})
    n_seq, n_batch = size(pos_matrix)
    # Reshape pos_matrix to (seq_len, 1, 1, batch) for broadcasting
    pos_idx = reshape(pos_matrix, (n_seq, 1, 1, n_batch))
    
    # Index cos and sin, preserving the head_dim but using custom positions for each batch
    # Result shape will be (head_dim, seq_len, 1, batch)
    new_cos = rope.cos[:, pos_idx]
    new_sin = rope.sin[:, pos_idx]
    
    return RoPE(new_cos, new_sin)
end
Flux.@layer RoPE trainable=()

function apply_scaling!(freqs::AbstractVector; scale_factor=8)
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192
    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    for (i, freq) in enumerate(freqs)
        wavelen = 2π / freq
        if wavelen > low_freq_wavelen
            freqs[i] = freq / scale_factor
        elseif wavelen > high_freq_wavelen
            @assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / 
                    (high_freq_factor - low_freq_factor)
            freqs[i] = (1 - smooth) * freq / scale_factor + smooth * freq
        end
    end
    return freqs
end

function RoPE(
    dim::Int, end_pos::Int; 
    theta::T=10000f0, use_scaled=true, scale_factor=8, start_pos=0
) where T
    freqs = 1f0 ./ (theta .^ (T.(0:2:dim-1)[1:dim÷2] ./ dim))
    use_scaled && apply_scaling!(freqs; scale_factor)
    freqs_complex = cis.(T.(start_pos:end_pos-1) * freqs')
    c = permutedims(real(freqs_complex), (2, 1))  # (head_dim/2, seq_len)
    s = permutedims(imag(freqs_complex), (2, 1))
    c = reshape(c, (dim÷2, end_pos - start_pos, 1))
    s = reshape(s, (dim÷2, end_pos - start_pos, 1))
    return RoPE(c, s)
end
# Note about Huggingface weights and rotary embeddings:
# https://discuss.huggingface.co/t/is-llama-rotary-embedding-implementation-correct/44509
# Use this one if you're using the Hugging Face weights.
function (rope::RoPE)(x)
    head_dim = size(x, 1)
    x1 = x[1:head_dim÷2, :, :, :]
    x2 = x[head_dim÷2+1:end, :, :, :]
    return vcat(  
        x1 .* rope.cos .- x2 .* rope.sin,
        x2 .* rope.cos .+ x1 .* rope.sin
    )
end

function unrope(rope, x)
    head_dim = size(x, 2)
    x1 = x[1:head_dim÷2, :, :, :]
    x2 = x[head_dim÷2+1:end, :, :, :]
    return vcat(  
        x1 .* rope.cos .+ x2 .* rope.sin,
        x2 .* rope.cos .- x1 .* rope.sin
    )
end

struct FixedRoPE{A <: AbstractArray}
    angle::A  # One angle per dimension pair
end
Flux.@layer FixedRoPE 
Flux.trainable(m::FixedRoPE) = (m.angle,)

function (rope::FixedRoPE)(x)
    head_dim = size(x, 1)
    x1 = x[1:head_dim÷2, :, :, :]
    x2 = x[head_dim÷2+1:end, :, :, :]
    c = cos.(rope.angle)
    s = sin.(rope.angle)
    rotx = vcat(
        x1 .* c .- x2 .* s,
        x2 .* c .+ x1 .* s
    )
    return rotx
end

struct IPARoPE
    rope::RoPE
    fixed_rope::FixedRoPE
end
Base.getindex(rope::IPARoPE, i) = IPARoPE(rope.rope[i], rope.fixed_rope)

function IPARoPE(dim::Int, end_pos::Int; 
    theta::T=10000f0, use_scaled=true, scale_factor=8, start_pos=0) where T
    return IPARoPE(
        RoPE(dim, end_pos; theta, use_scaled, scale_factor, start_pos),
        FixedRoPE([theta])
    )   
end

function dotproducts(qh::AbstractArray{T, 4}, kh::AbstractArray{T, 4}) where T<: Real
    qhT = permutedims(qh, (3, 1, 2, 4))
                         #c, FramesL, N_head, Batch
    kh = permutedims(kh, (1, 3, 2, 4))
    qhTkh = permutedims(#FramesR, #FramesL, N_head, Batch
                        batched_mul(qhT,kh)
                        #N_head, FramesR, FramesL, Batch when we use (3,1,2,4)
                            ,(3,1,2,4))
    return qhTkh
end

"""
function RoPEdotproducts(iparope::IPARoPE, q, k; chain_diffs = nothing)

    chain_diffs is either nothing or a array of 0's and 1's describing the ij-pair as pertaining to the same chain if the entry at ij is 1, else 0. 
"""
function dotproducts(iparope::IPARoPE, qh::AbstractArray{T, 4}, kh::AbstractArray{T, 4}; chain_diffs = 1) where T<: Real
    qropshape = permutedims(qh, (1,3,2,4))
    kropshape = permutedims(kh, (1,3,2,4)) 
    rotq, rotk = permutedims(iparope.rope(qropshape), (2,1,3,4)), iparope.rope(kropshape)
    rotqTrotk = permutedims(batched_mul(
        rotq,
        rotk
    ), (3,1,2,4))

    # when things are from different chain, we rotate only the queries by a fixed amount
    if chain_diffs != 1
        #return qropshape 
        rotq2 = permutedims(iparope.fixed_rope(qropshape), (2,1,3,4))
        rotq2Trotk2 = permutedims(batched_mul(
            rotq2,
            kropshape
        ), (3,1,2,4))
        # unsqueeze chain diffs to shape 1, framesR, framesL 
        rotqTrotk = Flux.unsqueeze(chain_diffs, 1) .* rotqTrotk .+ (1 .- Flux.unsqueeze(chain_diffs, 1) .* rotq2Trotk2)
    end
    return rotqTrotk
end
export dotproducts 
Flux.@layer IPARoPE
Flux.trainable(m::IPARoPE) = (m.fixed_rope,)
