module AuxillaryFunctions
using StaticArrays

export ResetArrays!, to_3d

ResetArrays!(arrays...) = foreach(a -> fill!(a, zero(eltype(a))), arrays)

to_3d(vec_2d) = [SVector(v..., 0.0) for v in vec_2d]

end