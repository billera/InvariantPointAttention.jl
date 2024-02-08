function right_to_left_mask(N_frames::T) where T<: Integer
    mask = fill(-Inf, N_frames, N_frames)
    mask[tril!(trues(N_frames, N_frames))] .= 0
    return mask
end

function right_to_left_mask(L::T, R::T; step::T = 10) where T <: Integer
    mask = ones(L, R) .* -Inf32
    for i in 1:R
        for j in 1:L
            if i <= step*(j-1) || i ==1  
                mask[j,i] = 0
            end
        end
    end
    return mask
end

function left_to_right_mask(L::T, R::T; step::T = 10) where T <: Integer
    mask = ones(L,R) .* -Inf32
    for i in 1:R
        for j in 1:L
            mask[j,i] = j < 10*(i-1) ? -Inf : 0
        end
    end
    return mask
end


function virtual_residues(S::AbstractArray, T::Tuple{AbstractArray, AbstractArray}; step = 10)
    Nr = size(S,2)
    vr = 1:step:Nr
    S_virt = S[:,vr,:]
    T_virt = (T[1][:,:,vr,:], T[2][:,:,vr,:])
    return S_virt, T_virt
end

