# taking from https://github.com/MurrellGroup/Jjama3.jl/blob/main/src/model.jl
# todo: re-write to avoid the O(N) permutedims. 
struct RoPE{A<:AbstractArray}
    cos::A
    sin::A
end
Base.getindex(rope::RoPE, i) = RoPE(rope.cos[:,i,:,:], rope.sin[:,i,:,:])

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
    cos = permutedims(real(freqs_complex), (2, 1))  # (head_dim/2, seq_len)
    sin = permutedims(imag(freqs_complex), (2, 1))
    cos = reshape(cos, (dim÷2, end_pos - start_pos, 1))
    sin = reshape(sin, (dim÷2, end_pos - start_pos, 1))
    return RoPE(cos, sin)
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

struct FixedRoPE{T<:Real}
    angle::T  # One angle per dimension pair
end

Flux.@layer FixedRoPE trainable=:angle
function FixedRoPE(dim::Int; T = Float32)
    angle = T(π/4)
    return FixedRoPE(angle)
end

function (rope::FixedRoPE)(x)
    head_dim = size(x, 1)
    x1 = x[1:head_dim÷2, :, :, :]
    x2 = x[head_dim÷2+1:end, :, :, :]
    cos = reshape(cos.(rope.angle), (dim÷2, 1, 1, 1))
    sin = reshape(sin.(rope.angle), (dim÷2, 1, 1, 1))
    rotx = vcat(
        x1 .* rope.cos .- x2 .* rope.sin,
        x2 .* rope.cos .+ x1 .* rope.sin
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
        FixedRoPE(dim; theta)
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
function dotproducts(iparope::IPARoPE, qh::AbstractArray{T, 4}, kh::AbstractArray{T, 4}; chain_diffs = nothing) where T<: Real
    # O(N) permutedims, shouldn't be too bad. 
    qropshape = permutedims(qh, (1,3,2,4))
    kropshape = permutedims(kh, (1,3,2,4))
    rotq, rotk = permutedims(iparope.rope(qropshape), (1,3,2,4)), permutedims(iparope.rope(kropshape), (1,3,2,4))
    rotqTrotk = dotproducts(rotq, rotk)
    # when things are from different chain, we rotate only the queries by a fixed amount
    if !isnothing(chain_diffs)
        rotq2 = permutedims(iparope.fixed_rope(qropshape), (1,3,2,4))
        rotq2Trotk2 = dotproducts(rotq2, kh)
        # unsqueeze chain diffs to shape 1, framesR, framesL 
        rotqTrotk = unsqueeze(chain_diffs, 1) .* rotqTrotk .+ (1 .- unsqueeze(chain_diffs, 1)) .* rotq2Trotk2
    end
    return rotqTrotk
end

Flux.@layer IPARoPE
