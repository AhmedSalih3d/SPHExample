using Test
using StaticArrays: SVector
using TimerOutputs: TimerOutput, @timeit
using SPHExampleGPU: Fixed, Fluid, write_simulation_report

function report_fixture(output_dir)
    particles = (
        Type = [Fluid, Fluid, Fixed],
        Density = [990.0, 1000.0, 2000.0],
        Velocity = [SVector(3.0, 4.0), SVector(0.0, 0.0), SVector(100.0, 0.0)],
        Position = [SVector(0.0, 0.0), SVector(0.0, 1.0), SVector(0.0, 2.0)],
        GroupMarker = [2, 2, 1],
    )
    timer = TimerOutput()
    @timeit timer "Simulation" begin
        @timeit timer "Neighbor update" sleep(0.001)
    end
    result = (;
        final_state = (; particles, time = 0.1, iteration = 5, complete = true, timestep = 0.02),
        performance = (;
            steps = 5,
            simulated_time = 0.1,
            wall_time_seconds = 1.5,
            mean_timestep = 0.02,
            milliseconds_per_step = 300.0,
            mparticle_steps_per_second = 0.01,
            output_write_seconds = 0.2,
            phase_timings = timer,
            phase_timings_synchronized = false,
        ),
        diagnostics = (;
            particle_count = 3,
            output_frames = 2,
            cell_list_rebuilds = 1,
            grid_dims = (8, 4),
            frozen_particles = 0,
            step_state_readbacks = 5,
            grid_status_readbacks = 0,
            capacity_growths = 1,
            bbox_host_readbacks = 0,
            captured_graphs = 0,
            periodic_source_rebuilds = 0,
            periodic_source_count = 3,
        ),
        termination_reason = :completed,
        configuration = (;
            requested = (; use_graph = true, double_position = true, lanes_per_particle = 2),
            capabilities = (; radial_interactions = true, scalar_sps_viscosity = false),
            effective = (;
                sph_kernel = :WendlandC2,
                density_diffusion_operator = :symmetric_reciprocal_density,
                viscosity_path = :reference_ordered,
                position_type = Float64,
                dimensions = 2,
            ),
            fallbacks = ((; option = :test_option, requested = true, effective = false, reason = :unsupported),),
            execution = (; path = :direct, graph_batches = 0, direct_batches = 1, graph_capture_failures = 0),
        ),
    )
    constants = (; dx = 0.02, ρ₀ = 1000.0, c₀ = 20.0, g = 9.81, m₀ = 0.4)
    metadata = (;
        SimulationName = "Wedge test",
        SaveLocation = output_dir,
        HourGlass = timer,
        Iteration = 0,
        OutputEach = 0.01,
        OutputTimes = 0.01,
        OutputIterationCounter = 0,
        StepsTakenForLastOutput = 0,
        CurrentTimeStep = 0.0,
        TotalTime = 0.0,
        SimulationTime = 0.09,
        IndexCounter = 0,
        VisualizeInParaview = true,
        ExportSingleVTKHDF = true,
        ExportGridCells = true,
        OutputVariables = ["Density", "Pressure"],
        OpenLogFile = false,
        TimeSteppingMode = :SymplecticTimeStepping,
        GPUUseGraph = true,
        GPUCacheInteractionFields = true,
    )
    geometry = [
        (;
            CSVFile = "",
            GroupMarker = 1,
            Type = Fixed,
            Motion = nothing,
            Floating = nothing,
            Particles = [1],
        ),
        (;
            CSVFile = "fluid.csv",
            GroupMarker = 2,
            Type = Fluid,
            Motion = nothing,
            Floating = nothing,
            Particles = [1, 2],
        ),
    ]
    kernel = (; kernel = "WendlandC2", h = 0.03, dx = 0.02)
    viscosity = (; model = "ArtificialViscosity", boundary_factor = 1.0)
    density_diffusion = (; model = "LinearDensityDiffusion")
    return (; result, constants, metadata, geometry, kernel, viscosity, density_diffusion)
end

function write_mock_log(output_dir)
    log = """
Julia Version 1.12.0
Platform: x86_64-w64-mingw32
Julia threads: 2
CUDA information:
CUDA runtime: 12.4
CUDA driver: 555.12
GPU: Test GPU, compute capability 8.6, 123456 bytes total VRAM, CUDA.jl 6.4
GPU float type: Float32
GPU position type: Float64
Logger Start Time: 30-10-2026 10:00:00
CPU particle arrays: 512 bytes (0.0005 MiB; 170.67 bytes/particle)
CPU particle memory scope: array payload only; excludes unused capacity.
GPU live-buffer memory (startup, before first cell-list rebuild):
GPU particle arrays (state + reorder): 100 bytes (0.0001 MiB; 33.33 bytes/particle)
GPU support arrays: 200 bytes (0.0002 MiB; 66.67 bytes/particle)
GPU cell-list workspace: 300 bytes (0.0003 MiB; 100.00 bytes/particle)
GPU auxiliary arrays (motion, reductions, step state, floating): 400 bytes (0.0004 MiB; 133.33 bytes/particle)
GPU accounted buffers total: 1000 bytes (0.0010 MiB; 333.33 bytes/particle)
CUDA pool: 2.0000 MiB used / 3.0000 MiB reserved (4.0000 MiB free device VRAM)
Output staging & buffer capacity: 1.0000 MiB pinned + 2.0000 MiB queue (up to 4 frames) + 8.0000 MiB HDF5 writer buffer
GPU memory scope: live array payload; excludes process or peak VRAM.
PART [-] PartTime [s] TotalSteps [-] Steps [-] Run Time [s] Time/Sec [-] Remaining Time [Date]
Part_0001 0.010000 10 10 1.250 125.00 30-10-2026 10:30:00
Part_0002 0.020000 20 10 1.500 75.00 30-10-2026 10:31:00
Configuration fallback: TestOption, requested=true, effective=false, reason=unsupported
Output frame write time (writer task, overlapped with GPU work): 0.20 [s], 2 frames buffered per file flush
Simulation finished at: 30-10-2026 10:31:00
Simulation took 1.50[s]
"""
    write(joinpath(output_dir, "SimulationOutput.log"), log)
    return nothing
end

function generate_report(fixture, output_dir; compile_pdf)
    return write_simulation_report(
        fixture.result,
        fixture.constants;
        output_dir,
        simulation_time = 0.09,
        simulation_name = "Wedge \"draft\"",
        metadata = fixture.metadata,
        geometry = fixture.geometry,
        kernel = fixture.kernel,
        viscosity = fixture.viscosity,
        density_diffusion = fixture.density_diffusion,
        compile_pdf,
    )
end

@testset "shared GPU Typst report" begin
    mktempdir() do output_dir
        write_mock_log(output_dir)
        fixture = report_fixture(output_dir)
        files = generate_report(fixture, output_dir; compile_pdf = false)

        @test isfile(files.typ_path)
        @test basename(files.typ_path) == "Wedge_draft_report.typ"
        @test files.pdf_path === nothing
        source = read(files.typ_path, String)
        @test !occursin("{{", source)
        @test occursin("Wedge \\\"draft\\\"", source)
        @test occursin("density", lowercase(source))
        @test occursin("[#text(\"990\")]", source)
        @test occursin("[#text(\"995\")]", source)
        @test occursin("[#text(\"5\")]", source)
        @test !occursin("[#text(\"2000\")]", source)
        @test occursin("Julia version", source)
        @test occursin("Test GPU, compute capability 8.6", source)
        @test occursin("GPU accounted buffers total", source)
        @test occursin("Part_0001 / 0.010000 s", source)
        @test occursin("30-10-2026 10:31:00", source)
        @test occursin("symmetric_reciprocal_density", source)
        @test occursin("Neighbor update", source)
        @test occursin("table.header", source)
        @test occursin("fill: (x, y) => if y == 0 { navy }", source)
        @test occursin("Density diffusion", source)
        @test occursin("Output frame write time", source)
        @test occursin("Simulation dimensions", source)
        @test !occursin("Simulation finished at: 30-10-2026", source)
    end

    mktempdir() do output_dir
        write_mock_log(output_dir)
        fixture = report_fixture(output_dir)
        files = write_simulation_report(
            fixture.result,
            fixture.constants;
            output_dir,
            simulation_time = 0.09,
            metadata = fixture.metadata,
            geometry = fixture.geometry,
            kernel = fixture.kernel,
            viscosity = fixture.viscosity,
            density_diffusion = fixture.density_diffusion,
            compile_pdf = true,
        )
        @test basename(files.typ_path) == "Wedge_test_report.typ"
        @test isfile(files.typ_path)
        @test isfile(files.pdf_path)
        @test occursin("Wedge test", read(files.typ_path, String))
        @test read(files.pdf_path, String)[1:5] == "%PDF-"
    end
end
