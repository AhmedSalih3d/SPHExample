# Standalone runner for floating-body physics and CUDA graph regression checks.
using Test
using SPHExampleGPU
using CUDA
using StaticArrays
using StructArrays
using LinearAlgebra
using HDF5

include("floating_bodies.jl")
