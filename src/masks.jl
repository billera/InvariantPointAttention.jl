"""
    right_to_left_mask([T=Float32,] N::Integer)

Create a right-to-left mask for the self-attention mechanism.
The mask is a matrix of size `N x N` where the diagonal and the
lower triangular part are set to zero and the upper triangular part is set to infinity.
"""
function right_to_left_mask(T::Type{<:AbstractFloat}, N::Integer)
    mask = fill(T(-Inf), N, N)
    mask[tril!(trues(N, N))] .= zero(T)
    return mask
end

"""
    right_to_left_mask([T=Float32,] L::Integer, R::Integer; step::Integer = 10)
"""
function right_to_left_mask(T::Type{<:AbstractFloat}, L::Integer, R::Integer; step::Integer = 10)
    mask = fill(T(-Inf), L, R)
    for j in axes(mask, 2)
        for i in axes(mask, 1)
            if j <= step*(i-1) || j == 1
                mask[i, j] = zero(T)
            end
        end
    end
    return mask
end

right_to_left_mask(args...) = right_to_left_mask(Float32, args...)

"""
    left_to_right_mask([T=Float32,] L::Integer, R::Integer; step::Integer = 10)
"""
function left_to_right_mask(T::Type{<:AbstractFloat}, L::Integer, R::Integer; step::Integer = 10)
    mask = fill(T(-Inf), L, R)
    for j in axes(mask, 2)
        for i in axes(mask, 1)
            if i >= step*(j-1)
                mask[i, j] = zero(T)
            end
        end
    end
    return mask
end

left_to_right_mask(args...) = left_to_right_mask(Float32, args...)

function virtual_residues(
    S::AbstractArray, T::Tuple{AbstractArray, AbstractArray};
    step::Integer = 10, rand_start::Bool = false,
)
    Nr = size(S, 2)
    start = 1
    if rand_start 
        start = sample(1:step)
    end
    vr = start:step:Nr
    S_virt = S[:,vr,:]
    T_virt = (T[1][:,:,vr,:], T[2][:,:,vr,:])
    return S_virt, T_virt
end

