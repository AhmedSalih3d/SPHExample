module ParticleMemoryLoggingTests

using Test, SPHExample, StaticArrays, StructArrays

const LOGGER = SPHExample.SimulationLoggerConfiguration

@testset "CPU particle memory logging" begin
    for D in (2, 3), T in (Float32, Float64)
        density = fill(T(1000), 3)
        particles = StructArray((
            Position = fill(zero(SVector{D, T}), 3),
            Density = density, DensityAlias = density,
            Type = [Fluid, Fixed, Moving], GroupMarker = UInt[1, 2, 3],
        ))
        expected = 3 * (D * sizeof(T) + sizeof(T) + sizeof(ParticleType) + sizeof(UInt))
        @test LOGGER.host_particle_bytes(particles) == expected

        mktempdir() do dir
            geometry = [Geometry{D, T}(CSVFile = "logging-test.csv", GroupMarker = 1, Type = Fluid)]
            constants = SimulationConstants{T}()
            metadata = SimulationMetaData{D, T}(SimulationName = "Logging", SaveLocation = dir,
                OutputTimes = T(0.02))
            kernel = SPHKernelInstance{D, T}(WendlandC2(); dx = constants.dx)
            logger = SimulationLogger(dir)
            try
                redirect_stdout(devnull) do
                    InitializeLogger(logger, constants, metadata, kernel, ZeroViscosity(),
                        ZeroDensityDiffusion(), geometry, particles)
                end
            finally
                close(logger.LoggerIo)
            end
            output = read(joinpath(dir, "SimulationOutput.log"), String)
            @test occursin("Total number of particles: 3", output)
            @test occursin("Simulation dimensions: $D", output)
            @test occursin("Particle precision: physics $T; CPU positions $T", output)
            @test occursin("CPU particle arrays: $expected bytes", output)
            @test occursin("MiB;", output)
            @test occursin("bytes/particle)", output)
            @test occursin("excludes geometry inputs, support/threaded/sorting arrays", output)
        end
    end

    empty = StructArray((Density = Float32[],))
    @test LOGGER.host_particle_bytes(empty) == 0
    @test LOGGER.memory_usage_line("CPU particle arrays", 0, 0) ==
        "CPU particle arrays: 0 bytes (0.0000 MiB; n/a bytes/particle)"
    @test LOGGER.memory_usage_line("CPU particle arrays", 2^20, 1024) ==
        "CPU particle arrays: 1048576 bytes (1.0000 MiB; 1024.00 bytes/particle)"
end

end
