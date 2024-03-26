#example 
using BenchmarkTools
using Bumper
using Parameters
using StaticArrays
# If one uses StrideArrays, code breaks for Bumper?


function update_arr1_basic!(arr1,indices)
    arr1   .= @view arr1[indices]

    return nothing
end

let
    arr1 = rand(Int,3027)
    indices = zeros(Int,length(arr1))
    u       = zeros(Int,length(arr1))
    sortperm!(indices, arr1)
    b1 = @benchmark update_arr1_basic!($arr1, $indices)
    display(b1)

    return nothing
end

function update_arr1_intricate!(arr1,u,indices)
    u    .= @view arr1[indices]
    arr1 .= u

    return nothing
end


let
    arr1 = rand(Int,3027)
    indices = zeros(Int,length(arr1))
    u       = zeros(Int,length(arr1))
    sortperm!(indices, arr1)

    b2 = @benchmark update_arr1_intricate!($arr1,$u,$indices)
    display(b2)
end

function update_arr1_permute!(arr1,indices)
    permute!(arr1, indices)
    return nothing
end

let
    arr1 = rand(Int,3027)
    indices = zeros(Int,length(arr1))
    u       = zeros(Int,length(arr1))
    sortperm!(indices, arr1)

    b3 = @benchmark update_arr1_permute!($arr1,$indices)
    display(b3)
end

function update_arr1_bumper!(arr1,indices)
    @no_escape begin
        temp  = @alloc(eltype(arr1),length(arr1))

        temp .= @view arr1[indices]
        arr1 .= temp
        
    end
end

let
    arr1 = rand(Int,3027)
    indices = zeros(Int,length(arr1))
    u       = zeros(Int,length(arr1))
    sortperm!(indices, arr1)

    b4 = @benchmark update_arr1_bumper!($arr1,$indices)
    display(b4)
end

struct ParticleTest
    Field::Int
    Val1::Int 
    Val2::Float64
end

let
    arr1    = [ParticleTest(rand(1:100), rand(1:500), rand()) for _ in 1:3027]
    indices = zeros(Int,length(arr1))
    u       = zeros(Int,length(arr1))
    b5      = @benchmark sort!($arr1, by = p -> p.Field)
    display(b5)
end

# Particle(Cell = CartesianIndex(0,0), Position = SVector(0.0,0.0), Acceleration = SVector(0.0,0.0), Velocity = SVector(0.0,0.0), Density = 1000.0, GravityFactor = 0.0, MotionLimiter = 1.0)
@with_kw struct Particle{D,T}
    Cell::CartesianIndex{D}
    Position::SVector{D,T}
    Acceleration::SVector{D,T}
    Velocity::SVector{D,T} 
    Density::T
    GravityFactor::T
    MotionLimiter::T
    ID::Int64
end

let
    arr1    = [Particle(Cell = CartesianIndex(rand(1:1000),rand(1:1000)), Position = SVector(rand(),rand()), Acceleration = SVector(0.0,0.0), Velocity = SVector(0.0,0.0), Density = 1000.0, GravityFactor = 0.0, MotionLimiter = 1.0, ID = rand(1:3027)) for _ in 1:3027]
    b5      = @benchmark sort!($arr1, by = p -> p.Cell)
    display(b5)
end