using InvariantPointAttention
using InvariantPointAttention: get_rotation, get_translation
using Test

@testset "InvariantPointAttention.jl" begin
    # Write your tests here.
    @testset "Softmax1" begin
        #Check if softmax1 is consistent with softmax, when adding an additional zero logit
        x = randn(4,3)
        xaug = hcat(x, zeros(4,1))
        @assert softmax1(x, dims = 2) ≈ softmax(xaug, dims = 2)[:,1:end-1]
    end

    @testset "IPACache" begin
        dims = 8
        c_z = 2
        settings = IPA_settings(dims; c_z)
        ipa = IPCrossA(settings)

        # generate random data
        L = 5
        R = 6
        B = 4
        siL = randn(Float32, dims, L, B)
        siR = randn(Float32, dims, R, B)
        zij = randn(Float32, c_z, R, L, B)
        TiL = (get_rotation(L, B), get_translation(L, B))
        TiR = (get_rotation(R, B), get_translation(R, B))

        # check the consistency
        cache = InvariantPointAttention.IPACache(settings, B)
        siR′, cache′ = InvariantPointAttention.expand(ipa, cache, TiL, siL, L, TiR, siR, R; zij)
        @test size(siR′) == size(siR)
        @test siR′ ≈ ipa(TiL, siL, TiR, siR; zij)

        # calculate in two steps
        cache = InvariantPointAttention.IPACache(settings, B)
        siR1, cache = InvariantPointAttention.expand(ipa, cache, TiL, siL, L, TiR, siR, 2; zij)
        siR2, cache = InvariantPointAttention.expand(ipa, cache, TiL, siL, 0, TiR, siR, 4; zij)
        @test cat(siR1, siR2, dims = 2) ≈ ipa(TiL, siL, TiR, siR; zij)
    end
end
