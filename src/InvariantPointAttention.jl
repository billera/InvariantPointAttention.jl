module InvariantPointAttention
using LinearAlgebra
using StatsBase
using Flux
using ChainRules
using ChainRulesCore

using Flux:unsqueeze

include("rotational_utils.jl")
include("layers.jl")
include("masks.jl")

export IPA
export IPAStructureModuleLayer
export BackboneUpdate
export IPA_settings
export IPCrossA
export right_to_left_mask
export left_to_right_mask 
export virtual_residues
export softmax1

end
