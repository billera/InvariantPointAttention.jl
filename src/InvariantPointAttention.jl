module InvariantPointAttention
using Statistics, Distances
using Flux, CUDA
using Diffusions: bcds2flatquats
using LinearAlgebra, Plots

include("attention.jl")
end 
