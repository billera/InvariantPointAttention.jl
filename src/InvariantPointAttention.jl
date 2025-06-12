module InvariantPointAttention

using LinearAlgebra
using Flux
using ChainRulesCore

include("utils.jl")
include("grads.jl")
include("rope.jl")
include("layers.jl")
include("masks.jl")

export IPA_settings
export IPA, IPCrossA
export IPARoPE
export IPAStructureModuleLayer, IPCrossAStructureModuleLayer
export BackboneUpdate
export right_to_left_mask
export left_to_right_mask 
export virtual_residues
export softmax1

end
