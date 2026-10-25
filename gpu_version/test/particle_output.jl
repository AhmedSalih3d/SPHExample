module ParticleOutputTests

using Test, SPHExampleGPU, StaticArrays, StructArrays, HDF5

const OUTPUT = SPHExampleGPU.ProduceHDFVTK

@testset "particle output preserves frames across bounded chunks and partial batches" begin
    # Keep tiny batches intact, cross chunks between/within larger frames,
    # and close with a partial batch in both the staged and synchronous paths.
    large_n = OUTPUT.MAX_PARTICLE_CHUNK_BYTES ÷ (3 * sizeof(Float64)) + 7
    for n in (17, 33_020, large_n), T in (Float32, Float64), staged in (false, true)
        mktempdir() do dir
            position = [SVector{2,T}(i / n, -i / n) for i in 1:n]
            particles = StructArray((; Position = copy(position),
                Velocity = fill(SVector{2,T}(1, -2), n), Density = fill(T(1001), n),
                ID = collect(Int32, 1:n), Type = fill(Fluid, n), GroupMarker = fill(Int8(2), n)))
            constants = SimulationConstants{T}(dx = 0.02)
            kernel = SPHKernelInstance{2,T}(WendlandC2(); dx = constants.dx)
            frame_bytes = n * (3sizeof(Float64) + 5sizeof(T) + sizeof(Int32) + 2sizeof(Int8))
            metadata = SimulationMetaData{2,T}(SimulationName = "Chunks", SaveLocation = dir,
                ExportSingleVTKHDF = true, GPUOutputBufferBytes = 2frame_bytes)
            output = SetupVTKOutput(metadata, particles, kernel, 2; constants)
            @test output.frames_per_flush == 2
            @test output.writer_buffer_bytes == sizeof(output.frame_writer.positions) +
                sum(sizeof, output.frame_writer.data)
            fields = [:Position, :Velocity, :Density, :ID, :Type, :GroupMarker]
            download = (; fields, buffers = [getproperty(particles, f) for f in fields], ghost_index = nothing)
            expected_positions = Matrix{T}[]
            try
                for k in 1:3
                    particles.Position .= position .+ Ref(SVector{2,T}(k, 2k))
                    particles.Density .= T(1000 + k)
                    push!(expected_positions, stack(particles.Position))
                    staged && output.stage_particles(download)
                    output.save_particles(k, 0.1k; staged_slot = staged)
                end
                @test frames_written(output.frame_writer) == 2
                @test frames_pending(output.frame_writer) == 1
            finally
                output.close_files()
            end
            h5open(joinpath(dir, "Chunks.vtkhdf"), "r") do io
                root = io["VTKHDF"]
                points = root["Points"]
                chunks = HDF5.get_chunk(points)
                @test prod(chunks) * sizeof(eltype(points)) <= OUTPUT.MAX_PARTICLE_CHUNK_BYTES
                for name in keys(root["PointData"])
                    dataset = root["PointData/$name"]
                    @test prod(HDF5.get_chunk(dataset)) * sizeof(eltype(dataset)) <= OUTPUT.MAX_PARTICLE_CHUNK_BYTES
                end
                @test n == 17 ? chunks[2] == 2n : chunks[2] < 2n
                @test size(points) == (3, 3n)
                @test HDF5.read_attribute(root["Steps"], "NSteps") == 3
                @test read(root["Steps/Values"]) == [0.1k for k in 1:3]
                @test read(root["Steps/PointOffsets"]) == [0, n, 2n]
                @test read(root["Steps/CellOffsets"]) == zeros(Int64, 4, 3)
                @test read(root["Vertices/NumberOfCells"]) == zeros(Int64, 3)
                saved = read(points)
                for k in 1:3
                    cols = (k-1)*n+1:k*n
                    @test saved[1:2, cols] == expected_positions[k]
                    @test all(iszero, saved[3, cols])
                    @test root["PointData/Velocity"][:, cols] == repeat(T[1, -2, 0], 1, n)
                    @test root["PointData/Density"][cols] == fill(T(1000 + k), n)
                    @test root["PointData/Pressure"][cols] == fill(
                        EquationOfStateGamma7(T(1000 + k), constants.c₀, constants.ρ₀), n)
                    @test root["PointData/ID"][cols] == particles.ID
                    @test root["PointData/Type"][cols] == fill(Int8(Fluid), n)
                    @test root["PointData/GroupMarker"][cols] == particles.GroupMarker
                end
            end
        end
    end
end

end
