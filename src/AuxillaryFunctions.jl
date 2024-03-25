module AuxillaryFunctions
using Bumper, StaticArrays

export RearrangeVector!, ResetArrays!, to_3d

ResetArrays!(arrays...) = foreach(a -> fill!(a, zero(eltype(a))), arrays)

to_3d(vec_2d) = [SVector(v..., 0.0) for v in vec_2d]

# Proper in-place re-arranging of vector
function RearrangeVector!(vec1,indices)
    buf = default_buffer()
    @no_escape buf begin
        temp  = @alloc(eltype(vec1),length(vec1))

        temp .= @view vec1[indices]
        vec1 .= temp
    end
end


end