"""
Returns the rotation matrix form of N flat quaternions. 
"""
function rotmatrix_from_quat(q)
 
    sx = 2q[1, :] .* q[2, :]
    sy = 2q[1, :] .* q[3, :]
    sz = 2q[1, :] .* q[4, :]

    xx = 2q[2, :].^2
    xy = 2q[2, :] .* q[3, :]
    xz = 2q[2, :] .* q[4, :]

    yy = 2q[3, :].^2
    yz = 2q[3, :] .* q[4, :]
    zz = 2q[4, :] .^ 2  
    
    r1 = reshape(1 .- (yy .+ zz), 1, :)
    r2 = reshape(xy .- sz, 1, :)
    r3 = reshape(xz .+ sy, 1, :)

    r4 = reshape( xy .+ sz, 1, :)
    r5 = reshape(1 .- (xx .+ zz), 1, :)
    r6 = reshape( yz .- sx, 1, :)

    r7 = reshape(xz .- sy, 1, :)
    r8 = reshape(yz .+ sx, 1, :)
    r9 = reshape(1 .- (xx .+ yy), 1, :)

    return reshape(vcat(r1, r4, r7, r2, r5, r8, r3, r6, r9), 3, 3, :)
end

"""
Creates a quaternion (as a vector) from a triplet of values (pirated from Diffusions.jl)
"""
function bcds2quats(bcd::AbstractArray{<: Real, 2})
    denom = sqrt.(1 .+ bcd[1,:].^2 .+ bcd[2,:].^2 .+ bcd[3,:].^2)
    return vcat((1 ./ denom)', bcd ./ denom')
end

"""
Generates random rotation matrices of given size.  
"""
get_rotation(N, M; T = Float32) = reshape(rotmatrix_from_quat(bcds2quats(randn(T,3,N*M))),3,3,N,M)
get_rotation(N; T = Float32) = reshape(rotmatrix_from_quat(bcds2quats(randn(T, 3,N))),3,3,N)

"""
Generates random translations of given size.
"""
get_translation(N,M; T = Float32) = randn(T,3,1,N,M)
get_translation(N; T = Float32) = randn(T, 3,1,N) 


""" 
Applies the SE3 transformations T = (rot,trans) ∈ SE(E3)^N
to N batches of m points in R3, i.e., mat ∈ R^(3 x m x N) ↦ T(mat) ∈ R^(3 x m x N).
Note here that rotations here are represented in matrix form. 
"""
function T_R3(mat, rot,trans)
    size_mat = size(mat)
    rotc = reshape(rot, 3,3,:)  
    trans = reshape(trans, 3,1,:)
    matc = reshape(mat,3,size(mat,2),:) 
    rotated_mat = batched_mul(rotc,matc) .+ trans
    return reshape(rotated_mat,size_mat)
end 


""" 
Applys the group inverse of the SE3 transformations T = (R,t) ∈ SE(3)^N to N batches of m points in R3,
such that T^-1(T*x) = T^-1(Rx+t) =  R^T(Rx+t-t) = x.
"""
function T_R3_inv(mat,rot,trans)
    size_mat = size(mat)
    rotc = batched_transpose(reshape(rot, 3,3,:))
    matc = reshape(mat,3,size(mat,2),:)
    trans = reshape(trans, 3,1,:)
    rotated_mat = batched_mul(rotc,matc .- trans)

    return reshape(rotated_mat,size_mat)
end

"""
Returns the composition of two SE(3) transformations T_1 and T_2. If T1 = (R1,t1), and T2 = (R2,t2) then T1*T2 = (R1*R2, R1*t2 + t1).
"""
function T_T(T_1, T_2)
    R1, t1 = T_1 
    R2, t2 = T_2
    new_rot = batched_mul(R1,R2)
    new_trans = reshape(batched_mul(R1,t2), size(t1)) .+ t1
    return (new_rot,new_trans)
end

"""
Takes a 6-dim vec and maps to a rotation matrix and translation vector, which is then applied to the input frames.
"""
function update_frame(Ti, arr)
    bcds = reshape(arr[:,1,:,:],3,:)
    rotmat = rotmatrix_from_quat(bcds2quats(bcds))  
    T_new = (
        reshape(rotmat,3,3,size(arr,3),:),
        reshape(arr[:,2,:,:],3,1,size(arr,3),:)
            )
    T = T_T(Ti,T_new)
    return T
end

"""
Returns an N_frames x N_frames lower triangular matrix, with -Inf instead of zeros in the upper half. 
This is added to the argument of the attention weight softmax, resulting in all right to left communication being zeroed out, while retaining the softmax distribution for nonzero values. 
"""
function right_to_left_mask(N_frames::Int64)
    mask = fill(-Inf, N_frames, N_frames)
    mask[tril!(trues(N_frames, N_frames))] .= 1
    return mask
end
