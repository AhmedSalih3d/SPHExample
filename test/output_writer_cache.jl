using Test
using HDF5
using SPHExample
using StaticArrays
using StructArrays

function WriterCachePaddedVectors(Data, ::Val{D}) where {D}
    T = eltype(eltype(Data))
    Result = zeros(T, 3, length(Data))
    for i in eachindex(Data), d in 1:D
        Result[d, i] = Data[i][d]
    end
    return Result
end

function WriterCacheGridPoints(Cells, Kernel, ::Val{D}) where {D}
    T = typeof(Kernel.H)
    Corners = ((-1, -1), (1, -1), (1, 1), (-1, 1))
    Points = SVector{3,T}[]
    for Cell in Cells, Layer in (D == 3 ? (-1, 1) : (0,)), Corner in Corners
        push!(Points, SVector{3,T}(
            Kernel.H * (T(Cell[1]) + T(Corner[1]) / 2),
            Kernel.H * (T(Cell[2]) + T(Corner[2]) / 2),
            D == 3 ? Kernel.H * (T(Cell[3]) + T(Layer) / 2) : zero(T),
        ))
    end
    return reduce(hcat, Points)
end

function WriterCacheGridIds(Cells, ::Val{D}) where {D}
    Axes = ntuple(d -> minimum(Cell -> Cell[d], Cells):maximum(Cell -> Cell[d], Cells), D)
    AllCells = vec(collect(CartesianIndices(Axes)))
    return [findfirst(==(Cell), AllCells) for Cell in Cells]
end

@testset "transient writer reuses and closes cached handles" begin
    mktempdir() do Directory
        h5open(joinpath(Directory, "cached_handles.vtkhdf"), "w") do File
            Root = HDF5.create_group(File, "VTKHDF")
            Steps = HDF5.create_group(Root, "Steps")
            Steps["Values"] = [0.1, 0.2]
            close(Steps)
            Cache = SPHExample.ProduceHDFVTK.CachedHDFGroup(Root)
            CachedSteps = Cache["Steps"]
            Values = CachedSteps["Values"]
            @test Cache["Steps"] === CachedSteps
            @test Cache["Steps"]["Values"] === Values
            @test isvalid(Values)
            @test read(Values) == [0.1, 0.2]
            close(Cache)
            @test !isvalid(Values)
            @test !isvalid(CachedSteps.Root)
            @test !isvalid(Root)
            @test isvalid(File)
        end
    end
end

@testset "transient writer preserves many particle and grid frames" begin
    for (D, T) in ((3, Float32), (2, Float64))
        @testset "$D dimensions, $T" begin
            mktempdir() do Directory
                Dimensions = Val(D)
                Count = 7
                FrameCount = 12
                CellCounts = [1, 4, 2, 6, 3, 5, 1, 6, 2, 4, 3, 5]
                Kernel = SPHKernelInstance{D,T}(WendlandC2(); h=T(0.5))
                Particles = StructArray((
                    Cells=fill(CartesianIndex(ntuple(_ -> 0, D)), Count),
                    Position=zeros(SVector{D,T}, Count),
                    Velocity=zeros(SVector{D,T}, Count),
                    Acceleration=zeros(SVector{D,T}, Count),
                    Density=zeros(T, Count),
                    Type=fill(Fluid, Count),
                    BoundaryBool=fill(UInt8(9), Count),
                    ID=Int32.(1:Count),
                ))
                MetaData = SimulationMetaData{D,T}(
                    SimulationName="writer_history",
                    SaveLocation=Directory,
                    OutputTimes=T(0.01),
                    ExportSingleVTKHDF=true,
                    ExportGridCells=true,
                    ExportGridCellParticleCounts=true,
                    VisualizeInParaview=false,
                    OpenLogFile=false,
                )
                Output = SetupVTKOutput(MetaData, Particles, Kernel, D)
                ExpectedParticles = NamedTuple[]
                ExpectedGrids = NamedTuple[]
                ExpectedTimes = T[]
                try
                    for Frame in 1:FrameCount
                        for i in eachindex(Particles.Position)
                            Particles.Position[i] = SVector{D,T}(ntuple(d -> T(100Frame + 10i + d), D))
                            Particles.Velocity[i] = SVector{D,T}(ntuple(d -> T(Frame + i + d / 4), D))
                            Particles.Acceleration[i] = SVector{D,T}(ntuple(d -> -T(Frame + 2i + d / 8), D))
                            Particles.Density[i] = T(1000 + 10Frame + i)
                            Particles.Type[i] = (Fluid, Fixed, Moving)[mod1(i + Frame, 3)]
                            Particles.ID[i] = Int32(100Frame + i)
                        end
                        MetaData.TotalTime = T(Frame) / T(100)
                        push!(ExpectedTimes, MetaData.TotalTime)
                        push!(ExpectedParticles, (
                            Position=WriterCachePaddedVectors(Particles.Position, Dimensions),
                            Velocity=WriterCachePaddedVectors(Particles.Velocity, Dimensions),
                            Acceleration=WriterCachePaddedVectors(Particles.Acceleration, Dimensions),
                            Density=copy(Particles.Density),
                            Type=Int8.(Particles.Type),
                            BoundaryBool=UInt8.(Particles.Type .!= Fluid),
                            ID=copy(Particles.ID),
                        ))
                        Cells = [CartesianIndex(ntuple(d -> d == 1 ? 2i - Frame :
                                                     d == 2 ? (-1)^i * mod(Frame, 3) :
                                                     mod(i + Frame, 4) - 2, D))
                                 for i in 1:CellCounts[Frame]]
                        ParticleCounts = Frame .+ collect(1:length(Cells))
                        NeighborCounts = 100Frame .+ collect(1:length(Cells))
                        push!(ExpectedGrids, (
                            Points=WriterCacheGridPoints(Cells, Kernel, Dimensions),
                            CellData=WriterCacheGridIds(Cells, Dimensions),
                            ParticleCount=copy(ParticleCounts),
                            ParticleNeighborsPerCell=copy(NeighborCounts),
                        ))
                        Output.enqueue_particles(Frame)
                        Output.enqueue_grid(Frame, Cells;
                                            cell_particle_counts=ParticleCounts,
                                            cell_neighbor_counts=NeighborCounts)
                        # More frames than either queue capacity exercise handle
                        # reuse while every job must retain its own snapshot.
                        fill!(Particles.Position, zero(SVector{D,T}))
                        fill!(Particles.Velocity, zero(SVector{D,T}))
                        fill!(Particles.Acceleration, zero(SVector{D,T}))
                        fill!(Particles.Density, zero(T))
                        fill!(Particles.Type, Moving)
                        fill!(Particles.ID, Int32(-1))
                        fill!(Cells, CartesianIndex(ntuple(_ -> -1000, D)))
                        fill!(ParticleCounts, -1)
                        fill!(NeighborCounts, -1)
                    end
                finally
                    Output.close_files()
                end
                @test !isopen(Output.file_handles.particle_files)
                @test !isopen(Output.file_handles.grid_files)

                h5open(joinpath(Directory, "writer_history.vtkhdf"), "r") do File
                    Root = File["VTKHDF"]
                    Steps = Root["Steps"]
                    PointData = Root["PointData"]
                    PointOffsets = Int64.(Count .* (0:(FrameCount - 1)))
                    @test HDF5.read_attribute(Root, "Version") == Int32[2, 3]
                    @test HDF5.read_attribute(Root, "Type") == "PolyData"
                    @test Set(keys(PointData)) == Set(["Velocity", "Acceleration", "Density", "Type", "BoundaryBool", "ID"])
                    @test read(Root["NumberOfPoints"]) == fill(Int64(Count), FrameCount)
                    @test read(Root["Points"]) == reduce(hcat, (State.Position for State in ExpectedParticles))
                    @test eltype(Root["Points"]) == T
                    for Name in ("Velocity", "Acceleration")
                        @test read(PointData[Name]) == reduce(hcat, (getproperty(State, Symbol(Name)) for State in ExpectedParticles))
                        @test eltype(PointData[Name]) == T
                    end
                    for (Name, FieldType) in (("Density", T), ("Type", Int8), ("BoundaryBool", UInt8), ("ID", Int32))
                        @test read(PointData[Name]) == reduce(vcat, (getproperty(State, Symbol(Name)) for State in ExpectedParticles))
                        @test eltype(PointData[Name]) == FieldType
                    end
                    @test read(Steps["Values"]) == Float64.(ExpectedTimes)
                    @test HDF5.read_attribute(Steps, "NSteps") == FrameCount
                    @test read(Steps["PointOffsets"]) == PointOffsets
                    @test read(Steps["PartOffsets"]) == Int64.(0:(FrameCount - 1))
                    @test read(Steps["NumberOfParts"]) == ones(Int64, FrameCount)
                    @test read(Steps["CellOffsets"]) == zeros(Int64, 4, FrameCount)
                    @test read(Steps["ConnectivityIdOffsets"]) == zeros(Int64, 4, FrameCount)
                    for Name in keys(PointData)
                        @test read(Steps["PointDataOffsets"][Name]) == PointOffsets
                    end
                    for Topology in ("Vertices", "Lines", "Polygons", "Strips")
                        for Name in ("NumberOfCells", "NumberOfConnectivityIds", "Offsets", "Connectivity")
                            @test read(Root[Topology][Name]) == zeros(Int64, FrameCount)
                        end
                    end
                end

                h5open(joinpath(Directory, "writer_history_GridCells.vtkhdf"), "r") do File
                    Root = File["VTKHDF"]
                    Steps = Root["Steps"]
                    PointsPerCell = 2^D
                    PointCounts = PointsPerCell .* CellCounts
                    CellOffsets = cumsum([0; CellCounts[1:(end - 1)]])
                    PointOffsets = PointsPerCell .* CellOffsets
                    @test HDF5.read_attribute(Root, "Version") == Int32[2, 3]
                    @test HDF5.read_attribute(Root, "Type") == "UnstructuredGrid"
                    @test Set(keys(Root["CellData"])) == Set(["CellData", "ParticleCount", "ParticleNeighborsPerCell"])
                    @test isempty(keys(Root["FieldData"]))
                    @test read(Root["Points"]) == reduce(hcat, (State.Points for State in ExpectedGrids))
                    @test eltype(Root["Points"]) == T
                    @test read(Root["NumberOfPoints"]) == PointCounts
                    @test read(Root["NumberOfCells"]) == CellCounts
                    @test read(Root["NumberOfConnectivityIds"]) == PointCounts
                    @test read(Root["Connectivity"]) == reduce(vcat, (collect(0:(Count - 1)) for Count in PointCounts))
                    @test read(Root["Offsets"]) == reduce(vcat, (collect(0:PointsPerCell:Count) for Count in PointCounts))
                    @test read(Root["Types"]) == fill(D == 3 ? UInt8(12) : UInt8(9), sum(CellCounts))
                    @test eltype(Root["Types"]) == UInt8
                    @test read(Steps["Values"]) == Float64.(ExpectedTimes)
                    @test HDF5.read_attribute(Steps, "NSteps") == FrameCount
                    @test read(Steps["PointOffsets"]) == PointOffsets
                    @test read(Steps["ConnectivityIdOffsets"]) == PointOffsets
                    @test read(Steps["CellOffsets"]) == CellOffsets
                    @test read(Steps["PartOffsets"]) == collect(0:(FrameCount - 1))
                    @test read(Steps["NumberOfParts"]) == ones(Int64, FrameCount)
                    for Name in ("CellData", "ParticleCount", "ParticleNeighborsPerCell")
                        @test read(Root["CellData"][Name]) == reduce(vcat, (getproperty(State, Symbol(Name)) for State in ExpectedGrids))
                        @test read(Steps["CellDataOffsets"][Name]) == CellOffsets
                        @test eltype(Root["CellData"][Name]) == Int64
                    end
                end
            end
        end
    end
end

@testset "public append helpers accept ordinary HDF5 groups" begin
    mktempdir() do Directory
        h5open(joinpath(Directory, "direct_append.vtkhdf"), "w") do File
            Root = HDF5.create_group(File, "Particles")
            Position = [SVector{3,Float32}(i, 2i, 3i) for i in 1:2]
            Density = Float32[1001, 1002]
            GenerateGeometryStructure(Root, ["Density"], Density; fType=Float32)
            GenerateStepStructure(Root, ["Density"], Density)
            AppendVTKHDFData(Root, 0.1, Position, ["Density"], Density)
            push!(Position, SVector{3,Float32}(3, 6, 9))
            push!(Density, Float32(1003))
            AppendVTKHDFData(Root, 0.2, Position, ["Density"], Density)
            @test read(Root["NumberOfPoints"]) == [2, 3]
            @test read(Root["Steps"]["PointOffsets"]) == [0, 2]
            @test read(Root["Steps"]["Values"]) == [0.1, 0.2]
            @test read(Root["PointData"]["Density"]) == Float32[1001, 1002, 1001, 1002, 1003]
            @test HDF5.read_attribute(Root["Steps"], "NSteps") == 2

            GridRoot = HDF5.create_group(File, "Grid")
            Kernel = SPHKernelInstance{2,Float64}(WendlandC2(); h=0.5)
            Names = ["CellData", "ParticleCount", "ParticleNeighborsPerCell"]
            GenerateGeometryStructure(GridRoot; vtk_file_type="UnstructuredGrid", cell_data_names=Names)
            GenerateStepStructure(GridRoot; vtk_file_type="UnstructuredGrid", cell_data_names=Names)
            AppendVTKHDFGridData(GridRoot, 0.1, Kernel, [CartesianIndex(0, 0), CartesianIndex(2, 0)], [3, 4], [5, 6])
            AppendVTKHDFGridData(GridRoot, 0.2, Kernel, [CartesianIndex(1, 1)], [7], [8])
            @test read(GridRoot["NumberOfCells"]) == [2, 1]
            @test read(GridRoot["Steps"]["CellOffsets"]) == [0, 2]
            @test read(GridRoot["Steps"]["PointOffsets"]) == [0, 8]
            @test read(GridRoot["CellData"]["ParticleCount"]) == [3, 4, 7]
            @test read(GridRoot["CellData"]["ParticleNeighborsPerCell"]) == [5, 6, 8]
            @test HDF5.read_attribute(GridRoot["Steps"], "NSteps") == 2
        end
    end
end
