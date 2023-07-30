module InvariantPointAttention
using Statistics, Distances
using Flux, CUDA
using Diffusions: bcds2flatquats
using LinearAlgebra, Plots

greet() = print("Hello World!")
include("attention.jl")
end # module IPA
