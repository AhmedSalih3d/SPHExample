module OutputQueueTests

using Test, SPHExampleGPU, CUDA, StaticArrays

const DRIVER = SPHExampleGPU.SPHCellList

function frame_prototype(D, T; n = 16, auxiliaries = true)
    position = fill(zero(SVector{D, T}), n)
    buffers = Vector[position, zeros(T, n)]
    ghost_index = auxiliaries ? zeros(Int32, n) : nothing
    floating = auxiliaries ? (; center = copy(position[1:2]), velocity = copy(position[1:2]),
        omega = zeros(T, 2)) : nothing
    return DRIVER.OutputDownload{D}([:Position, :Density], buffers,
        auxiliaries ? zeros(Int32, n) : Int32[], ghost_index, floating,
        CUDA.HostMemory[], CuEvent(CUDA.EVENT_DISABLE_TIMING))
end

@testset "output frames allocate on demand and reuse returned storage" begin
    for D in (2, 3), T in (Float32, Float64), auxiliaries in (false, true)
        prototype = frame_prototype(D, T; auxiliaries)
        pool = DRIVER.OutputFramePool(prototype, 3)
        @test pool.allocated == 0
        @test !isready(pool.free)
        first_frame = DRIVER.acquire_output_frame!(pool)
        @test pool.allocated == 1
        for (frame_buffer, source) in zip(first_frame.buffers, prototype.buffers)
            @test frame_buffer !== source
            @test typeof(frame_buffer) === typeof(source)
            @test length(frame_buffer) == length(source)
        end
        slim_pool = DRIVER.OutputFramePool(prototype, 2; fields = Symbol[])
        slim_frame = DRIVER.acquire_output_frame!(slim_pool)
        @test all(isempty, slim_frame.buffers)
        @test slim_frame.ghost_index === nothing
        @test (slim_frame.floating === nothing) == !auxiliaries
        close(slim_pool.free)
        @test isempty(first_frame.memory)
        @test (first_frame.ghost_index === nothing) == !auxiliaries
        @test (first_frame.floating === nothing) == !auxiliaries
        if auxiliaries
            @test first_frame.cid !== prototype.cid
            @test first_frame.ghost_index !== prototype.ghost_index
            for (frame_buffer, source) in zip(first_frame.floating, prototype.floating)
                @test frame_buffer !== source
                @test length(frame_buffer) == length(source)
            end
        end
        put!(pool.free, first_frame)
        @test DRIVER.acquire_output_frame!(pool) === first_frame
        @test pool.allocated == 1
        second_frame = DRIVER.acquire_output_frame!(pool)
        third_frame = DRIVER.acquire_output_frame!(pool)
        @test pool.allocated == 3
        @test first_frame.buffers[1] !== second_frame.buffers[1]
        @test second_frame.buffers[1] !== third_frame.buffers[1]
        close(pool.free)
    end

    prototype = frame_prototype(3, Float64; n = 4096)
    DRIVER.OutputFramePool(prototype, 64) # Warm the constructor before counting allocations.
    allocated_bytes = @allocated DRIVER.OutputFramePool(prototype, 64)
    @test allocated_bytes < 64 * 1024
    @test_throws ArgumentError DRIVER.OutputFramePool(prototype, 0)
end

@testset "output frame capacity applies backpressure without losing frame order" begin
    for capacity in (1, 3)
        prototype = frame_prototype(2, Float32)
        pool = DRIVER.OutputFramePool(prototype, capacity)
        ready = Channel{typeof(prototype)}(capacity)
        start_writer = Channel{Nothing}(1)
        collected = Channel{Int}(capacity)
        expected = collect(1:20)
        written = Int[]
        writer = Threads.@spawn begin
            take!(start_writer)
            for frame in ready
                push!(written, Int(first(frame.buffers[2])))
                put!(pool.free, frame)
            end
        end
        bind(pool.free, writer)
        bind(ready, writer)
        collector = Threads.@spawn begin
            try
                for index in expected
                    frame = DRIVER.acquire_output_frame!(pool)
                    fill!(frame.buffers[2], index)
                    put!(ready, frame)
                    index <= capacity && put!(collected, index)
                end
            finally
                close(ready)
            end
        end
        bind(collected, collector)
        try
            # Hold the writer until the pool is full, then require storage reuse
            # and ordered completion while producer and writer run concurrently.
            @test [take!(collected) for _ in 1:capacity] == expected[1:capacity]
            @test timedwait(() -> istaskdone(collector), 0.05; pollint = 0.001) == :timed_out
            @test pool.allocated == capacity
            put!(start_writer, nothing)
            completion = timedwait(() -> istaskdone(collector) && istaskdone(writer), 10)
            @test completion == :ok
            completion == :ok || error("Output queue workers did not finish")
            wait(collector)
            wait(writer)
            @test written == expected
            @test pool.allocated == capacity
        finally
            close(start_writer)
            close(pool.free)
            close(ready)
        end
    end
end

@testset "output writer failure stops frame acquisition" begin
    for capacity in (1, 3)
        pool = DRIVER.OutputFramePool(frame_prototype(2, Float32), capacity)
        DRIVER.acquire_output_frame!(pool)
        fail_writer = Channel{Nothing}(1)
        writer = Threads.@spawn begin
            take!(fail_writer)
            error("test output writer failure")
        end
        bind(pool.free, writer)
        waiting = capacity == 1 ? Threads.@spawn(DRIVER.acquire_output_frame!(pool)) : nothing
        try
            if waiting !== nothing
                @test timedwait(() -> istaskdone(waiting), 0.05; pollint = 0.001) == :timed_out
            end
            put!(fail_writer, nothing)
            @test_throws TaskFailedException wait(writer)
            @test timedwait(() -> !isopen(pool.free), 10) == :ok
            if waiting === nothing
                # Spare capacity must not permit allocation after writer failure.
                @test_throws TaskFailedException DRIVER.acquire_output_frame!(pool)
            else
                @test timedwait(() -> istaskdone(waiting), 10) == :ok
                @test_throws TaskFailedException fetch(waiting)
            end
            @test pool.allocated == 1
        finally
            close(fail_writer)
            close(pool.free)
        end
    end

    pool = DRIVER.OutputFramePool(frame_prototype(2, Float32), 3)
    close(pool.free)
    @test_throws InvalidStateException DRIVER.acquire_output_frame!(pool)
    @test pool.allocated == 0
end

end
