using TimerOutputs

mutable struct SimpleTimer
    time::Float64
    allocations::Int64
end

SimpleTimer() = SimpleTimer(0.0, 0)

macro customtimer(label, expr)
    return quote
        lab = $(esc(label))
        local timer_result = @timed begin
            $(esc(expr))
        end
        global custom_timer
        if isdefined(Main, :custom_timer)
            custom_timer.time += timer_result.time
            custom_timer.allocations += timer_result.bytes
        else
            custom_timer = SimpleTimer(timer_result.time, timer_result.bytes)
        end
        #println("$lab Time: ", timer_result.time, "s, Allocations: ", timer_result.bytes, " bytes")
    end
end

# Initialize the global custom timer
global custom_timer = SimpleTimer()


function f()
    to = TimerOutput()

    function g()
        @timeit to "Outer operation" begin
            @timeit to "Inner operation 1" begin
                # Some computation
            end
            @timeit to "Inner operation 2" begin
                # Another computation
            end
        end
    end

    function cust()
        @customtimer "Outer operation" begin
            @customtimer "Inner operation 1" begin
                # Some computation
            end
            @customtimer "Inner operation 2" begin
                # Another computation
            end
       end
    end

    g()
    cust()

    to
end

# Display the cumulative results
println("Total Time: ", custom_timer.time, "s, Total Allocations: ", custom_timer.allocations, " bytes")



f()