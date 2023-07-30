using LinearAlgebra
using Diffusions: bcds2flatquats
using Flux: batched_mul, batched_transpose
using CUDA
function rotmatrix_from_quat(q)
    """
    Returns the rotation matrix form of N flat quaternions. 
    """
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

function get_rotation(N, M)
    """
    Gets N random rotation matrices formatted as an array of size 3x3xN. 
    """
    return reshape(rotmatrix_from_quat(bcds2flatquats(randn(3,N*M))),3,3,N,M)
end
function get_rotation(N) return reshape(rotmatrix_from_quat(bcds2flatquats(randn(3,N))),3,3,N) end
function get_translation(N,M)
    """
    Gets N random translations formatted as an array of size 3x1xN (for purposes of broadcasting to arrays of size 3 x m x N)
    """
    return randn(3,1,N,M)
end
function get_translation(N) return randn(3,1,N) end


function T_R3(mat, rot,trans)
    """ 
    Applies the SE3 transformations T = (rot,trans) ∈ SE(E3)^N
    to N batches of m points in R3, i.e., mat ∈ R^(3 x m x N) ↦ T(mat) ∈ R^(3 x m x N).
    Note here that rotations here are represented in matrix form. 
    """
    size_mat = size(mat)
    rotc = reshape(rot, 3,3,:)  
    trans = reshape(trans, 3,1,:)
    matc = reshape(mat,3,size(mat,2),:) 
    batched_mul(rotc, matc)
    if trans != 0
        rotated_mat = batched_mul(rotc,matc) .+ trans
    else 
        rotated_mat = batched_mul(rotc,matc)
    end
    return reshape(rotated_mat,size_mat)
end 

function T_R3_inv(mat,rot,trans)
    """ 
    Applys the group inverse of the SE3 transformations T = (rot,trans) ∈ SE(3)^N to N batches of m points in R3,
    i.e., mat ∈ R^(3 x m x N) ↦ T^(-1)(mat) ∈ R^(3 x m x N) such that T(T^-1(mat)) = mat = T^-1(T(mat)). 
    Note here that rotations here are represented in matrix form.  
    """
    size_mat = size(mat)
    rotc = batched_transpose(reshape(rot, 3,3,:))
    matc = reshape(mat,3,size(mat,2),:)
    trans = reshape(trans, 3,1,:)
    if trans != 0
        rot_trans = batched_mul(rotc,trans)
        rotated_mat = batched_mul(rotc,matc) .- rot_trans
    else 
        rotated_mat = batched_mul(rotc,matc)
    end
    return reshape(rotated_mat,size_mat)
end

function T_T(T_1, T_2)
    """
    Returns the composition of two SE(3) transformations T_1 and T_2. Note that if T1 = (R1,t1), and T2 = (R2,t2) then T1*T2 = (R1*R2, R1*t2 + t1).
    T here is stored as a tuple (R,t).
    """

    R1, t1 = T_1
    R2, t2 = T_2
    new_rot = batched_mul(R1, R2)
    new_trans = batched_mul(R1,t2) .+ t1
    return (new_rot,new_trans)
end
