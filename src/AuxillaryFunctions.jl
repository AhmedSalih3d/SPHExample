module AuxillaryFunctions
using Bumper

export RearrangeVector!, ResetArrays!

ResetArrays!(arrays...) = foreach(a -> fill!(a, zero(eltype(a))), arrays)

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