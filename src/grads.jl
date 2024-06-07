# to do
# use chain rule config to avoid code repetition 
# unit test coverage of all custom grads, only have a couple and the overall IPA grads in runtests
# customgrad for ohp_r, make computation more efficient
# write customgrad for permutedims(batched_mul(permutedims(A,(...)),permutedims(B,(...))),(...))
# customgrad through the o-sums directly to aijh,q,k,v
# customgrad through aijh directly to qh,kh,qhp,khp

function sumabs2(x::AbstractArray{T}; dims = 1) where {T}
    sum(abs2, x; dims)
end

function ChainRulesCore.rrule(::typeof(sumabs2), x; dims = 1)
    function sumabs2_pullback(_Δ)
        Δ = unthunk(_Δ)
        xthunk = @thunk 2 .* x .* Δ
        return (NoTangent(), xthunk)
    end
    sumabs2(x; dims), sumabs2_pullback
end

function L2norm(x::AbstractArray{T}; dims = 1, eps = 1f-7) where {T}
    sqrt.(sumabs2(x; dims) .+ eps )
end

function ChainRulesCore.rrule(::typeof(L2norm), x::AbstractArray{T}; dims = 1, eps = 1f-7) where {T}
    normx = L2norm(x; dims, eps)
    function L2norm_pullback(_Δ)
        Δ = unthunk(_Δ)
        return (NoTangent(), @thunk(Δ .* x ./ normx))
    end
    return normx, L2norm_pullback
end

function pair_diff(A::AbstractArray{T}, B::AbstractArray{T}; dims = 4) where {T}
    return Flux.unsqueeze(A, dims = dims + 1) .- Flux.unsqueeze(B, dims = dims)
end

function ChainRulesCore.rrule(::typeof(pair_diff), A::AbstractArray{T}, B::AbstractArray{T}; dims=4) where {T}
    y = pair_diff(A, B; dims)
    function pair_diff_pullback(_Δ)
        Δ = unthunk(_Δ)
        return (NoTangent(), @thunk(sumdrop(Δ; dims=dims+1)), @thunk(-sumdrop(Δ; dims=dims)))
    end
    return y, pair_diff_pullback
end

function ChainRulesCore.rrule(::typeof(T_R3), x::AbstractArray{T,N}, R::AbstractArray{T,N}, t::AbstractArray{T,N}) where {T,N}
    function T_R3_pullback(_Δy)
        Δy = unthunk(_Δy)
        Δx = @thunk(batched_mul(_batched_transpose(R), Δy))
        ΔR = @thunk(batched_mul(Δy, _batched_transpose(x)))
        Δt = @thunk(sum(Δy, dims=2))
        return (NoTangent(), Δx, ΔR, Δt)
    end
    return T_R3(x, R, t), T_R3_pullback
end

function ChainRulesCore.rrule(::typeof(T_R3_inv), x::AbstractArray{T,N}, R::AbstractArray{T,N}, t::AbstractArray{T,N}) where {T,N}
    z = x .- t
    y = batched_mul(_batched_transpose(R), z)
    function T_R3_inv_pullback(_Δy)
        Δy = unthunk(_Δy)
        Δx = @thunk(batched_mul(R, Δy))
        ΔR = @thunk(batched_mul(z, _batched_transpose(Δy)))
        Δt = @thunk(-sum(Δx, dims=2)) # t is in the same position as x, but negated and broadcasted
        return (NoTangent(), Δx, ΔR, Δt)
    end
    return T_R3_inv(x, R, t), T_R3_inv_pullback
end

#=
function diff_sum_glob(T, q, k)
    bs = size(q) 
    qresh = reshape(q, size(q,1), size(q,2)*size(q,3), size(q,4),size(q,5))
    kresh = reshape(k, size(k,1), size(k,2)*size(k,3), size(k,4),size(k,5))
    Tq, Tk = T_R3(qresh,T[1],T[2]),T_R3(kresh,T[1],T[2])
    Tq, Tk = reshape(Tq, bs...), reshape(Tk, bs...)
    diffs = sumabs2(pair_diff(Tq, Tk, dims = 4),dims=[1,3])
end

function _diff_sum_glob_no_rrule(T,q,k)
    bs = size(q)
    qresh = reshape(q, size(q,1), size(q,2)*size(q,3), size(q,4),size(q,5))
    kresh = reshape(k, size(k,1), size(k,2)*size(k,3), size(k,4),size(k,5))
    Tq, Tk = T_R3_no_rrule(qresh,T[1],T[2]),T_R3_no_rrule(kresh,T[1],T[2])
    Tq, Tk = reshape(Tq, bs...), reshape(Tk, bs...)
    diffs = _sumabs2_no_rrule(_pair_diff_no_rrule(Tq, Tk, dims = 4),dims=[1,3])
end=#
#=
# not implemented grad with respect to T here, as is not needed in any applications for now
# this rrule provides ≈ 2x memory improvement by computing query and key grads simultaneously
function ChainRulesCore.rrule(::typeof(diff_sum_glob), T, q, k)

    bs = size(q) 
    qresh = reshape(q, size(q,1), size(q,2)*size(q,3), size(q,4),size(q,5))
    kresh = reshape(k, size(k,1), size(k,2)*size(k,3), size(k,4),size(k,5))
    (Tq, Tq_pullback), (Tk, Tk_pullback) = rrule(T_R3, qresh,T[1],T[2]), rrule(T_R3, kresh,T[1],T[2])
    Tq, Tk = reshape(Tq, bs...), reshape(Tk, bs...)
    pair_diffs, pair_diffs_pullback = rrule(pair_diff, Tq, Tk, dims = 4)
    sabs2, sabs2_pullback = rrule(sumabs2, pair_diffs, dims = [1,3])

    function diff_sum_glob_pullback(_Δ)
        # Our applications always use these for now, so no thunk since we easily save some compute by sharing ops
        Δ = unthunk(_Δ)
        _, Δdiffs = sabs2_pullback(Δ)
        _, ΔTq, ΔTk = pair_diffs_pullback(unthunk(Δdiffs))
        _, Δq, _, _ = Tq_pullback(reshape(unthunk(ΔTq), size(q,1), size(q,2)*size(q,3), size(q,4),size(q,5)))
        _, Δk, _, _ = Tk_pullback(reshape(unthunk(ΔTk), size(k,1), size(k,2)*size(k,3), size(k,4),size(k,5)))
        return (NoTangent(), ZeroTangent(), reshape(unthunk(Δq),bs...), reshape(unthunk(Δk),bs...))
    end

    return sabs2, diff_sum_glob_pullback
end

function qhTkh(q, k)
    qhT = permutedims(q, (3, 1, 2, 4))
    kh = permutedims(k, (1, 3, 2, 4))
    qhTkh = permutedims(batched_mul(qhT,kh),(3,1,2,4))
end

function _qhTkh_no_rrule(q, k)
    qhT = permutedims(q, (3, 1, 2, 4))
    kh = permutedims(k, (1, 3, 2, 4))
    qhTkh = permutedims(batched_mul(qhT,kh),(3,1,2,4))
end

function ChainRulesCore.rrule(::typeof(qhTkh), q, k)
    qhT = permutedims(q, (3, 1, 2, 4))
    kh = permutedims(k, (1, 3, 2, 4))
    bs = size(qhT)[3:4]
    qhTresh = reshape(qhT, size(qhT,1), size(qhT,2), size(qhT,3)*size(qhT,4)) #FramesR, c, N_head * Batch
    khresh = reshape(kh, size(kh,1), size(kh,2), size(kh,3)*size(kh,4)) # c, Frames L, N_head*Batch
    bm = batched_mul(qhTresh,khresh)
    qhTkh_unperm = reshape(bm, size(qhT,1), size(kh,2),bs...)#, FramesR, FramesL, N_head, Batch
    qhTkh = permutedims( qhTkh_unperm, (3,1,2,4))
    function qhTkh_pullback(_Δ)
        Δ = unthunk(_Δ)
        # Δ in N_head, N_R, N_L, Batch
        # Re-use the qhTkh memory to avoid GPU allocation
        Δcomp = reshape(permutedims!(qhTkh_unperm, Δ, (2, 3, 1, 4)),size(Δ,2), size(Δ,3), bs[1]*bs[2])
        Δq = @thunk begin 
            tmp = batched_mul(khresh, batched_adjoint(Δcomp))
            permutedims(reshape(tmp, size(tmp,1), size(tmp,2), bs...),(3,1,2,4))
        end
        Δk = @thunk begin 
            tmp = batched_mul(batched_adjoint(qhTresh), Δcomp)
            permutedims(reshape(tmp, size(tmp,1), size(tmp,2), bs...),(3,1,2,4))
        end
        return (NoTangent(), Δq, Δk)
    end
    return qhTkh, qhTkh_pullback
end
=#
"""
softmax1(x, dims = 1)

Behaves like softmax, but as though there was an additional logit of zero along dims (which is excluded from the output). So the values will sum to a value between zero and 1.
"""
function softmax1(x::AbstractArray{T}; dims = 1) where {T}
    _zero = T(0)
    max_ = max.(fast_maximum2(x; dims), _zero)
    @fastmath out = exp.(x .- max_)
    tmp = sum(out, dims = dims)
    out ./ (tmp + exp.(-max_))
end
# Pirated/adapted from NNlib
fast_maximum2(x::AbstractArray{T}; dims) where {T} = @fastmath reduce(max, x; dims, init = float(T)(-Inf))

function ∇softmax1_data(dy::AbstractArray{T}, y::AbstractArray{S}; dims = 1) where {T,S}
    dx = if NNlib.within_gradient(y)
        tmp = dy .* y
        tmp .- y .* sum(tmp; dims)
    else
        # This path is faster, only safe for 1st derivatives though.
        # Was previously `∇softmax!(dx, dy, x, y; dims)` to allow CUDA overloads,
        # but that was slow: https://github.com/FluxML/NNlibCUDA.jl/issues/30
        out = similar(y, promote_type(T,S))  # sure to be mutable
        out .= dy .* y
        out .= out .- y .* sum(out; dims)
    end
end

function ChainRulesCore.rrule(::typeof(softmax1), x; dims = 1)
    y = softmax1(x; dims)
    softmax_pullback(dy) = (NoTangent(), ∇softmax1_data(unthunk(dy), y; dims))
    return y, softmax_pullback
end


function pre_softmax_aijh(qh::AbstractArray{T},kh::AbstractArray{T},Ti,qhp::AbstractArray{T},khp::AbstractArray{T}, bij::AbstractArray{T}, gamma_h::AbstractArray{T}) where T
    w_C = T(sqrt(2f0/(9f0*size(qhp,3))))
    dim_scale = T(1f0/sqrt(size(qh,1)))
    w_L = T(1f0/sqrt(3f0))

    w_L.*(dim_scale.*qhTkh(qh,kh) .+ bij .- w_C/2 .* gamma_h .* dropdims(diff_sum_glob(Ti,qhp,khp),dims=(1,3)))
end

function test_version()
    println("Hello World! gradablateaij")
end