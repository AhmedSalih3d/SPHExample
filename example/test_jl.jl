using Base.Threads
using Bumper
using StrideArrays # Not necessary, but can make operations like broadcasting with Bumper.jl faster.
using Polyester
using BenchmarkTools
using LoopVectorization

# Sample data
buf = default_buffer()
A = rand(1:100, 1000) # Array of indices
B = A#rand(1:100, 1000) # Another array of indices, different values
X = zeros(Float64, 100)        # Shared array to update
Y = zeros(Float64, 100)        # Another shared array to update

# function yay(A,B,X,Y)
#     # Thread-local storage for each thread to avoid race conditions
#     local_X = [zeros(Float64, length(X)) for _ in 1:nthreads()]
#     local_Y = [zeros(Float64, length(Y)) for _ in 1:nthreads()]

#     @threads for iter in eachindex(A, B)
#         i = A[iter]
#         j = B[iter]

#         # Sample update values, replace with actual computation
#         update_val_X = 1 # Example value for X
#         update_val_Y = -2 # Example value for Y

#         # Accumulate in thread-local storage
#         local_X[threadid()][i] += update_val_X
#         local_X[threadid()][j] += -update_val_X
#         local_Y[threadid()][i] += update_val_Y
#         local_Y[threadid()][j] += -update_val_Y
#     end

#     # Reduce the thread-local storage into the shared arrays
#     for tid in 1:nthreads()
#         X .+= local_X[tid]
#         Y .+= local_Y[tid]
#     end
# end


# function f(A,B,X,Y)
#     # Set up a scope where memory may be allocated, and does not escape:
#     @no_escape begin
#         # Allocate a `PtrArray` (see StrideArraysCore.jl) using memory from the default buffer.
#         XT = eltype(X); XL = length(X)
#         local_X1 = @alloc(XT, XL)
#         local_X2 = @alloc(XT, XL)
#         local_X3 = @alloc(XT, XL)
#         local_X4 = @alloc(XT, XL)
#         local_X  = (local_X1, local_X2, local_X3, local_X4)

#         # local_Y1 = [@alloc(eltype(Y), length(Y)) for _ in 1:nthreads()]
#         YT = eltype(Y); YL = length(Y)
#         local_Y1 = @alloc(YT, YL)
#         local_Y2 = @alloc(YT, YL)
#         local_Y3 = @alloc(YT, YL)
#         local_Y4 = @alloc(YT, YL)
#         local_Y  = (local_Y1, local_Y2, local_Y3, local_Y4)
        
#         for x in local_X, y in local_Y
#             x .*= zero(eltype(X))
#             y .*= zero(eltype(Y))
#         end


#         # Check if local_X is being cleared
#         # println(sum.(local_X))

#         # for iter in eachindex(A, B)
#         @batch for iter in eachindex(A,B)
#             i = A[iter]
#             j = B[iter]

#             # Sample update values, replace with actual computation
#             update_val_X = 1 # Example value for X
#             update_val_Y = 1 # Example value for Y

#             # Accumulate in thread-local storage
#             local_X[threadid()][i] += update_val_X
#             local_X[threadid()][j] += -update_val_X
#             local_Y[threadid()][i] += update_val_Y
#             local_Y[threadid()][j] += -update_val_Y
#         end

#         # Reduce the thread-local storage into the shared arrays
#         for tid in 1:nthreads()
#             X .+= local_X[tid]
#             Y .+= local_Y[tid]
#         end


#     end
# end


# function f(A,B,X,Y, buf)
#     # Set up a scope where memory may be allocated, and does not escape:
#     @no_escape buf begin
#         # Allocate a `PtrArray` (see StrideArraysCore.jl) using memory from the default buffer.
#         XT = eltype(X); XL = length(X)
#         local_X1 = @alloc(XT, XL)
#         local_X2 = @alloc(XT, XL)
#         local_X3 = @alloc(XT, XL)
#         local_X4 = @alloc(XT, XL)
#         local_X  = (local_X1, local_X2, local_X3, local_X4)

#         # local_Y1 = [@alloc(eltype(Y), length(Y)) for _ in 1:nthreads()]
#         YT = eltype(Y); YL = length(Y)
#         local_Y1 = @alloc(YT, YL)
#         local_Y2 = @alloc(YT, YL)
#         local_Y3 = @alloc(YT, YL)
#         local_Y4 = @alloc(YT, YL)
#         local_Y  = (local_Y1, local_Y2, local_Y3, local_Y4)
        
#         # for x in local_X, y in local_Y
#         #     x .*= zero(eltype(X))
#         #     y .*= zero(eltype(Y))
#         # end


#         # Check if local_X is being cleared
#         println(sum.(local_X))

#         # for iter in eachindex(A, B)
#         @batch for iter in eachindex(A,B)
#             i = A[iter]
#             j = B[iter]

#             # Sample update values, replace with actual computation
#             update_val_X = rand() # Example value for X
#             update_val_Y = rand() # Example value for Y

#             # Accumulate in thread-local storage
#             local_X[threadid()][i] += update_val_X
#             local_X[threadid()][j] += -update_val_X
#             local_Y[threadid()][i] += update_val_Y
#             local_Y[threadid()][j] += -update_val_Y
#         end

#         # Reduce the thread-local storage into the shared arrays
#         for tid in 1:nthreads()
#             X .+= local_X[tid]
#             Y .+= local_Y[tid]
#         end


#     end
# end

function f(A,B,X,Y, buf)
    # Set up a scope where memory may be allocated, and does not escape:
    @no_escape buf begin
        # Allocate a `PtrArray` (see StrideArraysCore.jl) using memory from the default buffer.
        XT = eltype(X); XL = length(X)
        local_X  = @alloc(XT, XL)
        local_Y  = @alloc(XT, XL)

        for iter in eachindex(A,B)
            i = A[iter]
            j = B[iter]

            # Sample update values, replace with actual computation
            update_val_X = 10 # Example value for X
            update_val_Y = 1 # Example value for Y

            # Accumulate in thread-local storage
            local_X[iter] += update_val_X
            local_X[iter] += -update_val_X
            local_Y[iter] += update_val_Y
            local_Y[iter] += -update_val_Y
        end

        # Check if local_X is being cleared
        println(sum.(local_X))

        # Reduce the thread-local storage into the shared arrays
        for tid in 1:100
            X[tid] += local_X[tid]
            Y[tid] += local_Y[tid]
        end

        # X .+= 10
    end
end


# display(@benchmark yay($A,$B,$X,$Y))

# display(@benchmark   f($A,$B,$X,$Y))

# @benchmark f(a,b,x,y,buf) setup = begin
#     buf = default_buffer()
#     a = rand(1:100, 1000) # Array of indices
#     b = rand(1:100, 1000) # Another array of indices, different values
#     x = zeros(Float64, 100)        # Shared array to update
#     y = zeros(Float64, 100)        # Another shared array to update
# end