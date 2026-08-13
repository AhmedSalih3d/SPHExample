
module ProduceHDFVTK

"""
Utility functions for exporting simulation data to the VTKHDF file format.

Two modes are supported:

* **Static** – every simulation step is written to a new file.
* **Transient** – a single file is kept open and new steps are appended.

The functions in this module are fairly low level.  `SetupVTKOutput` is
the main entry point and returns closures for saving particle and grid
data during a simulation run.
"""

export SaveVTKHDF, GenerateGeometryStructure, GenerateStepStructure,
       AppendVTKHDFData, SaveCellGridVTKHDF, AppendVTKHDFGridData,
       SetupVTKOutput

    using Base.Threads
    using Bumper: AllocBuffer, @alloc, @no_escape
    using HDF5
    using StaticArrays

    using ..AuxiliaryFunctions: to_3d!
    using ..SimulationGeometry


    const idType = Int64
    const fType = Float64

    """Schema information needed to create a VTK point-data dataset."""
    struct VTKFieldSchema{T, N} end

    struct ParticleSnapshot{P, V}
        positions::P
        output_data::V
    end

    struct ParticleWriteJob{T <: AbstractFloat, P, V}
        iteration::Int
        time::T
        snapshot::ParticleSnapshot{P, V}
    end

    struct GridWriteJob{N, T <: AbstractFloat}
        iteration::Int
        time::T
        cells::Vector{CartesianIndex{N}}
        cell_particle_counts::Union{Nothing, Vector{Int}}
        cell_neighbor_counts::Union{Nothing, Vector{Int}}
    end

    """Write an ASCII attribute `name => value` to `grp`."""
    function write_ascii_attribute(grp, name, value)
        dtype = HDF5.datatype(value)
        HDF5.API.h5t_set_cset(dtype.id, HDF5.API.H5T_CSET_ASCII)
        dspace = HDF5.dataspace(value)
        attr = HDF5.create_attribute(grp, name, dtype, dspace)
        HDF5.write_attribute(attr, dtype, value)
    end

    @inline GridPointCount(::Val{2}, CellCount) = 4 * CellCount
    @inline GridPointCount(::Val{3}, CellCount) = 8 * CellCount

    function FillGridPoints!(Points, SimKernel, UniqueCells, ::Val{2})
        H = SimKernel.H
        HalfH = H / 2
        T = eltype(eltype(Points))
        @inbounds for (CellIndex, Cell) in pairs(UniqueCells)
            xi, yi = Cell.I
            x = xi * H
            y = yi * H
            PointIndex = 4 * (CellIndex - 1)
            Points[PointIndex + 1] = SVector{3, T}(x - HalfH, y - HalfH, zero(T))
            Points[PointIndex + 2] = SVector{3, T}(x + HalfH, y - HalfH, zero(T))
            Points[PointIndex + 3] = SVector{3, T}(x + HalfH, y + HalfH, zero(T))
            Points[PointIndex + 4] = SVector{3, T}(x - HalfH, y + HalfH, zero(T))
        end
        return Points
    end

    function FillGridPoints!(Points, SimKernel, UniqueCells, ::Val{3})
        H = SimKernel.H
        HalfH = H / 2
        T = eltype(eltype(Points))
        @inbounds for (CellIndex, Cell) in pairs(UniqueCells)
            xi, yi, zi = Cell.I
            x = xi * H
            y = yi * H
            z = zi * H
            PointIndex = 8 * (CellIndex - 1)
            Points[PointIndex + 1] = SVector{3, T}(x - HalfH, y - HalfH, z - HalfH)
            Points[PointIndex + 2] = SVector{3, T}(x + HalfH, y - HalfH, z - HalfH)
            Points[PointIndex + 3] = SVector{3, T}(x + HalfH, y + HalfH, z - HalfH)
            Points[PointIndex + 4] = SVector{3, T}(x - HalfH, y + HalfH, z - HalfH)
            Points[PointIndex + 5] = SVector{3, T}(x - HalfH, y - HalfH, z + HalfH)
            Points[PointIndex + 6] = SVector{3, T}(x + HalfH, y - HalfH, z + HalfH)
            Points[PointIndex + 7] = SVector{3, T}(x + HalfH, y + HalfH, z + HalfH)
            Points[PointIndex + 8] = SVector{3, T}(x - HalfH, y + HalfH, z + HalfH)
        end
        return Points
    end

    function GridCellIds(UniqueCells::AbstractVector{CartesianIndex{2}})
        minx, maxx = extrema(Cell -> Cell[1], UniqueCells)
        miny = minimum(Cell -> Cell[2], UniqueCells)
        nx = maxx - minx + 1
        return [
            (Cell[2] - miny) * nx + (Cell[1] - minx) + 1
            for Cell in UniqueCells
        ]
    end

    function GridCellIds(UniqueCells::AbstractVector{CartesianIndex{3}})
        minx, maxx = extrema(Cell -> Cell[1], UniqueCells)
        miny, maxy = extrema(Cell -> Cell[2], UniqueCells)
        minz = minimum(Cell -> Cell[3], UniqueCells)
        nx = maxx - minx + 1
        ny = maxy - miny + 1
        return [
            (Cell[3] - minz) * (nx * ny) + (Cell[2] - miny) * nx + (Cell[1] - minx) + 1
            for Cell in UniqueCells
        ]
    end

    @inline GridVTKType(::Val{2}) = UInt8(9)
    @inline GridVTKType(::Val{3}) = UInt8(12)

    function RequiredGridBufferBytes(SimKernel, MaximumCellCount, Dimensions)
        PointType = SVector{3, typeof(SimKernel.H)}
        return max(
            Base.checked_mul(GridPointCount(Dimensions, MaximumCellCount), sizeof(PointType)),
            1,
        )
    end

    @inline function ResolveGridBuffer(Buffer, SimKernel, CellCount, Dimensions)
        return Buffer === nothing ?
            AllocBuffer(RequiredGridBufferBytes(SimKernel, CellCount, Dimensions)) :
            Buffer
    end

    @inline function ResolveParticleBuffer(Buffer, Positions, OutputData, Dimensions)
        return Buffer === nothing ?
            AllocBuffer(RequiredVTKBufferBytes(Positions, OutputData, Dimensions)) :
            Buffer
    end

    function cell_data_payload(cell_ids, cell_particle_counts, cell_neighbor_counts)
        payload = Dict("CellData" => cell_ids)
        if cell_particle_counts !== nothing
            payload["ParticleCount"] = cell_particle_counts
        end
        if cell_neighbor_counts !== nothing
            payload["ParticleNeighborsPerCell"] = cell_neighbor_counts
        end
        return payload
    end

    @inline function IsVectorField(Source)
        return eltype(Source) <: StaticVector || eltype(Source) <: CartesianIndex
    end

    @inline function VectorElementType(Source)
        if eltype(Source) <: StaticVector
            return eltype(eltype(Source))
        elseif eltype(Source) <: CartesianIndex
            return eltype(eltype(Source))
        end
        return eltype(Source)
    end

    @inline VTKElementType(::VTKFieldSchema{T, N}) where {T, N} = T
    @inline VTKComponentCount(::VTKFieldSchema{T, N}) where {T, N} = N

    @inline function VTKElementType(Source)
        return IsVectorField(Source) ? VectorElementType(Source) : eltype(Source)
    end

    @inline function VTKComponentCount(Source)
        return IsVectorField(Source) ? 3 : 1
    end

    @inline OutputFieldSchema(::Val{:Type}, Source) = VTKFieldSchema{Int8, 1}()
    @inline OutputFieldSchema(::Val{:BoundaryBool}, Source) = VTKFieldSchema{UInt8, 1}()

    @inline function OutputFieldSchema(::Val{Name}, Source) where {Name}
        T = VTKElementType(Source)
        return IsVectorField(Source) ? VTKFieldSchema{T, 3}() : VTKFieldSchema{T, 1}()
    end

    function FillVectorBuffer!(Dest, Source, Dimensions)
        if eltype(Source) <: StaticVector
            if Dimensions == 2
                to_3d!(Dest, Source)
            else
                copy!(Dest, Source)
            end
            return Dest
        end
        if eltype(Source) <: CartesianIndex
            if Dimensions == 2
                @inbounds for i in eachindex(Source)
                    idx = Source[i]
                    Dest[i] = SVector(idx[1], idx[2], zero(VectorElementType(Source)))
                end
            else
                @inbounds for i in eachindex(Source)
                    idx = Source[i]
                    Dest[i] = SVector(idx[1], idx[2], idx[3])
                end
            end
            return Dest
        end
        copy!(Dest, Source)
        return Dest
    end

    @inline function FillTypeBuffer!(Dest::AbstractVector{Int8}, Source)
        @inbounds for i in eachindex(Source)
            Dest[i] = Int8(Source[i])
        end
        return Dest
    end

    @inline function FillBoundaryBoolBuffer!(Dest::AbstractVector{UInt8}, Source)
        @inbounds for i in eachindex(Source)
            Dest[i] = UInt8(Source[i] != Fluid)
        end
        return Dest
    end

    @inline function ResolveOutputSource(::Val{:BoundaryBool}, SimParticles)
        return SimParticles.Type
    end

    @inline function ResolveOutputSource(::Val{Name}, SimParticles) where {Name}
        return getproperty(SimParticles, Name)
    end

    @inline function AllocateSnapshotBuffer(::Val{:Type}, Source, n, Dimensions)
        return Vector{Int8}(undef, n)
    end

    @inline function AllocateSnapshotBuffer(::Val{:BoundaryBool}, Source, n, Dimensions)
        return Vector{UInt8}(undef, n)
    end

    @inline function AllocateSnapshotBuffer(::Val{Name}, Source, n, Dimensions) where {Name}
        return similar(Source, n)
    end

    @inline function FillOutputField!(::Val{:Type}, Dest, Source, Dimensions)
        return FillTypeBuffer!(Dest, Source)
    end

    @inline function FillOutputField!(::Val{:BoundaryBool}, Dest, Source, Dimensions)
        return FillBoundaryBoolBuffer!(Dest, Source)
    end

    @inline function FillOutputField!(::Val{Name}, Dest, Source, Dimensions) where {Name}
        copy!(Dest, Source)
        return Dest
    end

    function RequiredVTKBufferBytes(Positions, OutputSources, ::Val{2})
        PositionType = SVector{3, VectorElementType(Positions)}
        RequiredBytes = Base.checked_mul(length(Positions), sizeof(PositionType))
        for Source in OutputSources
            if IsVectorField(Source)
                BufferType = SVector{3, VectorElementType(Source)}
                RequiredBytes = max(
                    RequiredBytes,
                    Base.checked_mul(length(Source), sizeof(BufferType)),
                )
            end
        end
        return max(RequiredBytes, 1)
    end

    @inline RequiredVTKBufferBytes(Positions, OutputSources, ::Val{3}) = 1

    @inline function ThreeDimensionalView(Source)
        return reinterpret(reshape, VectorElementType(Source), Source)
    end

    function WriteStaticVectorField!(Group, Name, Source, ::Val{2}, Buffer)
        T = VectorElementType(Source)
        @no_escape Buffer begin
            Data3D = @alloc(SVector{3, T}, length(Source))
            FillVectorBuffer!(Data3D, Source, 2)
            Group[Name] = ThreeDimensionalView(Data3D)
            nothing
        end
        return nothing
    end

    function WriteStaticVectorField!(Group, Name, Source, ::Val{3}, Buffer)
        Group[Name] = ThreeDimensionalView(Source)
        return nothing
    end

    function WriteStaticField!(Group, Name, Source, Dimensions, Buffer)
        if IsVectorField(Source)
            WriteStaticVectorField!(Group, Name, Source, Dimensions, Buffer)
        else
            Group[Name] = Source
        end
        return nothing
    end

    function WriteTransientVectorField!(Dataset, Columns, Source, ::Val{2}, Buffer)
        T = VectorElementType(Source)
        @no_escape Buffer begin
            Data3D = @alloc(SVector{3, T}, length(Source))
            FillVectorBuffer!(Data3D, Source, 2)
            Dataset[:, Columns] = ThreeDimensionalView(Data3D)
            nothing
        end
        return nothing
    end

    function WriteTransientVectorField!(Dataset, Columns, Source, ::Val{3}, Buffer)
        Dataset[:, Columns] = ThreeDimensionalView(Source)
        return nothing
    end

    function WriteTransientField!(Dataset, Indices, Source, Dimensions, Buffer)
        if IsVectorField(Source)
            WriteTransientVectorField!(Dataset, Indices, Source, Dimensions, Buffer)
        else
            Dataset[Indices] = Source
        end
        return nothing
    end

    function SaveVTKHDF(fid_vector, index, filepath, points, variable_names = String[], args...;
                        Dimensions = Val(3), Buffer = nothing, CloseAfterWrite = false)
        @assert length(variable_names) == length(args) "Same number of variable_names as args is necessary"
        ParticleBuffer = ResolveParticleBuffer(Buffer, points, args, Dimensions)
        io = h5open(filepath, "w")
        try
            gtop = HDF5.create_group(io, "VTKHDF")

            HDF5.attrs(gtop)["Version"] = [2, 3]
            write_ascii_attribute(gtop, "Type", "PolyData")

            # Points
            np = length(points)
            gtop["NumberOfPoints"] = [np]
            WriteStaticVectorField!(gtop, "Points", points, Dimensions, ParticleBuffer)

            # Point data
            let g = HDF5.create_group(gtop, "PointData")
                for i ∈ eachindex(variable_names)
                    WriteStaticField!(g, variable_names[i], args[i], Dimensions, ParticleBuffer)
                end
                close(g)
            end

            # Vertices: 1 point per cell
            let g = HDF5.create_group(gtop, "Vertices")
                g["NumberOfCells"] = [np]
                g["NumberOfConnectivityIds"] = [np]
                g["Connectivity"] = collect(0:(np - 1))
                g["Offsets"] = collect(0:np)
                close(g)
            end

            # Empty groups for unused cell types
            for type ∈ ("Lines", "Polygons", "Strips")
                gempty = HDF5.create_group(gtop, type)
                gempty["NumberOfCells"] = [0]
                gempty["NumberOfConnectivityIds"] = [0]
                gempty["Connectivity"] = Int[]
                gempty["Offsets"] = [0]
                close(gempty)
            end
            close(gtop)

            if CloseAfterWrite
                # Static writer jobs are complete after this call. Closing immediately
                # avoids retaining one HDF5 handle for every requested frame.
                close(io)
            else
                # Preserve the public helper's existing handle-storage behavior.
                fid_vector[index] = io
            end
        catch
            isopen(io) && close(io)
            rethrow()
        end
    end


    function GenerateGeometryStructure(root, variable_names = String[], args...;
                                       chunk_size = 100,
                                       vtk_file_type = "PolyData",
                                       idType = Int64,
                                       fType = Float64,
                                       cell_data_names = ["CellData"])
        @assert length(variable_names) == length(args) "Same number of variable_names as args is necessary"
        # Write version of VTKHDF format as an attribute
        HDF5.attrs(root)["Version"] = Int32.([2, 3])
        
        write_ascii_attribute(root, "Type", vtk_file_type)
        
        NumberOfPoints = HDF5.create_dataset(root, "NumberOfPoints" , idType , ((0,),(-1,)), chunk=(chunk_size,))
        Points         = HDF5.create_dataset(root, "Points"         , fType  , ((3,0), (3,-1)), chunk=(3,chunk_size))

        if vtk_file_type == "PolyData"
            connectivities = ["Vertices", "Lines", "Polygons", "Strips"]
            for connect in connectivities
                group = HDF5.create_group(root, connect)

                NumberOfConnectivityIds = HDF5.create_dataset(group, "NumberOfConnectivityIds", idType, ((0,),(-1,)), chunk=(chunk_size,) )
                NumberOfCells           = HDF5.create_dataset(group, "NumberOfCells", idType, ((0,),(-1,)), chunk=(chunk_size,) )
                Offsets                 = HDF5.create_dataset(group, "Offsets", idType, ((0,),(-1,)), chunk=(chunk_size,) )
                Connectivity            = HDF5.create_dataset(group, "Connectivity", idType, ((0,),(-1,)), chunk=(chunk_size,) )
            end

            pData = HDF5.create_group(root, "PointData")

            for i ∈ eachindex(variable_names)
                var_name = variable_names[i]
                arg      = args[i]
                arg_val_type   = VTKElementType(arg)
                arg_val_length = VTKComponentCount(arg)

                if arg_val_length == 3
                    HDF5.create_dataset(pData, var_name, arg_val_type, ((3,0),(3,-1)), chunk=(3,chunk_size))
                else
                    HDF5.create_dataset(pData, var_name, arg_val_type, ((0,),(-1,)), chunk=(chunk_size,))
                end
            end
        elseif vtk_file_type == "UnstructuredGrid"
            HDF5.create_dataset(root, "Connectivity" , idType , ((0,),(-1,)), chunk=(chunk_size,))
            HDF5.create_dataset(root, "NumberOfCells" , idType , ((0,),(-1,)), chunk=(chunk_size,))
            HDF5.create_dataset(root, "NumberOfConnectivityIds" , idType , ((0,),(-1,)), chunk=(chunk_size,))
            HDF5.create_dataset(root, "Offsets" , idType , ((0,),(-1,)), chunk=(chunk_size,))
            HDF5.create_dataset(root, "Types" , UInt8 , ((0,),(-1,)), chunk=(chunk_size,)) #Must be UInt8
            
            FieldData = HDF5.create_group(root, "FieldData") #Currently just empty group

            CellData = HDF5.create_group(root, "CellData")
            for name in cell_data_names
                HDF5.create_dataset(CellData, name, idType, ((0,), (-1,)),
                                    chunk=(chunk_size,))
            end

        end

        return nothing
    end

    function GenerateStepStructure(root, variable_names = String[], args...;
                                   vtk_file_type = "PolyData",
                                   chunk_size = 1000,
                                   cell_data_names = ["CellData"])
        steps = HDF5.create_group(root, "Steps")
    
        NSteps, _ = HDF5.create_attribute(steps, "NSteps", Int32)
    
        Values = HDF5.create_dataset(steps, "Values", fType , ((0,),(-1,)), chunk=(chunk_size,))
    
        singleDSs = ["PartOffsets", "NumberOfParts", "PointOffsets"]
        for name in singleDSs
            HDF5.create_dataset(steps, name, idType, ((0,),(-1,)), chunk=(chunk_size,))
        end
    
        if vtk_file_type == "PolyData"
            nTopoDSs = ["CellOffsets", "ConnectivityIdOffsets"]
            for name in nTopoDSs
                HDF5.create_dataset(steps, name, idType, ((4,0),(4, -1)), chunk=(4, chunk_size))
            end
        elseif vtk_file_type == "UnstructuredGrid"
            nTopoDSs = ["CellOffsets", "ConnectivityIdOffsets"]
            for name in nTopoDSs
                HDF5.create_dataset(steps, name, idType, ((0,),(-1,)), chunk=(chunk_size,))
            end
            cData = HDF5.create_group(steps, "CellDataOffsets")
            for name in cell_data_names
                HDF5.create_dataset(cData, name, idType, ((0,),(-1,)),
                                    chunk=(chunk_size,))
            end
        end
            
        pData = HDF5.create_group(steps, "PointDataOffsets")

        for i ∈ eachindex(variable_names)
            var_name = variable_names[i]

            HDF5.create_dataset(pData, var_name, idType, ((0,),(-1,)),
                                chunk=(chunk_size,))
        end
    
    end

    function AppendVTKHDFData(root, newStep, Positions, variable_names, args...;
                              Dimensions = Val(3), Buffer = nothing)
        ParticleBuffer = ResolveParticleBuffer(Buffer, Positions, args, Dimensions)
        steps = root["Steps"]

        # To update attributes, this is the best way I've found so far
        old_NSteps = HDF5.read_attribute(steps, "NSteps")
        steps_attr_dict = attributes(steps)
        NSteps = steps_attr_dict["NSteps"]
        write_attribute(NSteps,HDF5.datatype(idType), old_NSteps + 1)

        HDF5.set_extent_dims(steps["Values"], (length(steps["Values"]) + 1,))
        steps["Values"][end] = newStep

        PointsStartIndex = size(root["Points"])[2] + 1
        PositionLength   = length(Positions)

        PointIndices = PointsStartIndex:(PointsStartIndex + PositionLength - 1)
        HDF5.set_extent_dims(root["Points"], (3, size(root["Points"])[2] + PositionLength))
        WriteTransientVectorField!(root["Points"], PointIndices, Positions, Dimensions, ParticleBuffer)

        HDF5.set_extent_dims(steps["PointOffsets"], (length(steps["PointOffsets"]) + 1,))
        steps["PointOffsets"][end] = PointsStartIndex - 1

        HDF5.set_extent_dims(root["NumberOfPoints"], (length(root["NumberOfPoints"]) + 1,))
        root["NumberOfPoints"][end] = PositionLength

        NumberOfPartsStartIndex = length(steps["NumberOfParts"]) + 1
        HDF5.set_extent_dims(steps["NumberOfParts"], (length(steps["NumberOfParts"]) + 1,))
        steps["NumberOfParts"][NumberOfPartsStartIndex] = 1

        PartOffsetsStartIndex = length(steps["PartOffsets"]) + 1
        PartOffsetsLength     = length(steps["PartOffsets"]) + 1
        HDF5.set_extent_dims(steps["PartOffsets"], (PartOffsetsLength,))
        steps["PartOffsets"][PartOffsetsStartIndex] = PartOffsetsLength - 1

        CellOffsetsStartIndex = size(steps["CellOffsets"])[2] + 1
        HDF5.set_extent_dims(steps["CellOffsets"], (4, CellOffsetsStartIndex))
        # steps["CellOffsets"][:, CellOffsetsStartIndex] = zeros(4) # When you extent dimensions, it autofills with zero values

        ConnectivityIdOffsetsStartIndex = size(steps["ConnectivityIdOffsets"])[2] + 1
        HDF5.set_extent_dims(steps["ConnectivityIdOffsets"], (4, ConnectivityIdOffsetsStartIndex))
        # steps["ConnectivityIdOffsets"][:, ConnectivityIdOffsetsStartIndex] = zeros(4) # When you extent dimensions, it autofills with zero values

        for point_data_name in keys(steps["PointDataOffsets"])
            HDF5.set_extent_dims(steps["PointDataOffsets"][point_data_name], (length(steps["PointDataOffsets"][point_data_name]) + 1,))
            steps["PointDataOffsets"][point_data_name][end] = Int(PointsStartIndex - 1)
        end

        for i ∈ eachindex(variable_names)
            var_name = variable_names[i]
            arg      = args[i]
            arg_val_length = VTKComponentCount(arg)

            if arg_val_length == 3
                HDF5.set_extent_dims(root["PointData"][var_name], (arg_val_length, size(root["PointData"][var_name], 2) + PositionLength))
            else
                HDF5.set_extent_dims(root["PointData"][var_name], (length(root["PointData"][var_name]) + PositionLength,))
            end
            WriteTransientField!(root["PointData"][var_name], PointIndices, arg, Dimensions, ParticleBuffer)
        end
        

        connectivities = ["Vertices", "Lines", "Polygons", "Strips"]
        for connect in connectivities
            for dataset in ["NumberOfCells", "NumberOfConnectivityIds", "Offsets", "Connectivity"]
                HDF5.set_extent_dims(root[connect][dataset], (length(root[connect][dataset]) + 1,))
                # root[connect][dataset][end] = idType(0) # When you extent dimensions, it autofills with zero values
            end
        end
    end

    function AppendVTKHDFGridData(root, newStep, SimKernel, UniqueCells,
                                  cell_particle_counts = nothing,
                                  cell_neighbor_counts = nothing;
                                  Dimensions = Val(length(first(UniqueCells))),
                                  Buffer = nothing)
        CellCount = length(UniqueCells)
        PointCount = GridPointCount(Dimensions, CellCount)
        vtk_type = GridVTKType(Dimensions)
        cell_ids = GridCellIds(UniqueCells)
        PointsPerCell = GridPointCount(Dimensions, 1)
        offsets = 0:PointsPerCell:PointCount
        GridBuffer = ResolveGridBuffer(Buffer, SimKernel, CellCount, Dimensions)

        steps = root["Steps"]

        # To update attributes, this is the best way I've found so far
        old_NSteps = HDF5.read_attribute(steps, "NSteps")
        steps_attr_dict = attributes(steps)
        NSteps = steps_attr_dict["NSteps"]
        write_attribute(NSteps,HDF5.datatype(idType), old_NSteps + 1)

        HDF5.set_extent_dims(steps["Values"], (length(steps["Values"]) + 1,))
        steps["Values"][end] = newStep

        PointsStartIndex = size(root["Points"])[2] + 1
        ExtendedPositionLength = size(root["Points"])[2] + PointCount
        HDF5.set_extent_dims(root["Points"], (3, ExtendedPositionLength))
        T = typeof(SimKernel.H)
        @no_escape GridBuffer begin
            Points = @alloc(SVector{3, T}, PointCount)
            FillGridPoints!(Points, SimKernel, UniqueCells, Dimensions)
            root["Points"][:, PointsStartIndex:ExtendedPositionLength] = ThreeDimensionalView(Points)
            nothing
        end

        HDF5.set_extent_dims(steps["PointOffsets"], (length(steps["PointOffsets"]) + 1,))
        steps["PointOffsets"][end] = PointsStartIndex - 1

        HDF5.set_extent_dims(steps["NumberOfParts"], (length(steps["NumberOfParts"]) + 1,))
        steps["NumberOfParts"][end] = 1

        PartOffsetsStartIndex = length(steps["PartOffsets"]) + 1
        PartOffsetsLength     = length(steps["PartOffsets"]) + 1
        HDF5.set_extent_dims(steps["PartOffsets"], (PartOffsetsLength,))
        steps["PartOffsets"][PartOffsetsStartIndex] = PartOffsetsLength - 1

        HDF5.set_extent_dims(steps["ConnectivityIdOffsets"], (length(steps["ConnectivityIdOffsets"]) + 1,))
        steps["ConnectivityIdOffsets"][end] = PointsStartIndex - 1

        ##             
        HDF5.set_extent_dims(root["NumberOfPoints"], (length(root["NumberOfPoints"]) + 1,))
        root["NumberOfPoints"][end] = PointCount

        HDF5.set_extent_dims(root["NumberOfCells"], (length(root["NumberOfCells"]) + 1,))
        root["NumberOfCells"][end] = length(UniqueCells)

        ## Update data
        HDF5.set_extent_dims(root["Connectivity"], (ExtendedPositionLength,))
        root["Connectivity"][PointsStartIndex:end] = (PointsStartIndex:ExtendedPositionLength) .- PointsStartIndex

        HDF5.set_extent_dims(root["NumberOfConnectivityIds"], (length(root["NumberOfConnectivityIds"]) + 1,))
        root["NumberOfConnectivityIds"][end] = PointCount



        HDF5.set_extent_dims(steps["CellOffsets"], (length(steps["CellOffsets"]) + 1,)) #For first value, it becomes 0

        if length(steps["CellOffsets"]) == 1
            LastCellOffset = 0
        else
            LastCellOffset = sum(root["NumberOfCells"][:]) - length(UniqueCells)
        end

        steps["CellOffsets"][end] = LastCellOffset

        OffsetStartIndex = length(root["Offsets"])  + 1
        OffsetsLength    = length(root["Offsets"]) + length(UniqueCells) + 1
        HDF5.set_extent_dims(root["Offsets"], (OffsetsLength,))
        root["Offsets"][OffsetStartIndex:end] = offsets

        TypesStartIndex = length(root["Types"]) + 1
        HDF5.set_extent_dims(root["Types"], (length(root["Types"]) + length(UniqueCells),))
        root["Types"][TypesStartIndex:end] = vtk_type

        cell_payload = cell_data_payload(
            cell_ids,
            cell_particle_counts,
            cell_neighbor_counts,
        )
        for (name, data) in cell_payload
            if !haskey(root["CellData"], name)
                HDF5.create_dataset(
                    root["CellData"],
                    name,
                    eltype(data),
                    ((0,), (-1,)),
                    chunk=(1024,),
                )
            end
            start_index = length(root["CellData"][name]) + 1
            HDF5.set_extent_dims(
                steps["CellDataOffsets"][name],
                (length(steps["CellDataOffsets"][name]) + 1,),
            )
            steps["CellDataOffsets"][name][end] = start_index - 1
            HDF5.set_extent_dims(
                root["CellData"][name],
                (length(root["CellData"][name]) + length(UniqueCells),),
            )
            root["CellData"][name][start_index:end] = data
        end

        
        return nothing
    end

    function SaveCellGridVTKHDF(FilePath, SimKernel, UniqueCells,
                                cell_particle_counts = nothing,
                                cell_neighbor_counts = nothing;
                                Dimensions = Val(length(first(UniqueCells))),
                                Buffer = nothing)
        CellCount = length(UniqueCells)
        PointCount = GridPointCount(Dimensions, CellCount)
        PointsPerCell = GridPointCount(Dimensions, 1)
        offsets = 0:PointsPerCell:PointCount
        cell_ids = GridCellIds(UniqueCells)
        vtk_type = GridVTKType(Dimensions)
        GridBuffer = ResolveGridBuffer(Buffer, SimKernel, CellCount, Dimensions)

        # Open HDF5 file for writing
        io = h5open(FilePath, "w")
        try
            # Create top-level group "VTKHDF"
            gtop = HDF5.create_group(io, "VTKHDF")

            HDF5.attrs(gtop)["Version"] = [2, 3]
            write_ascii_attribute(gtop, "Type", "UnstructuredGrid")

            # Write Number of Points, Number of Cells, and Number of Connectivity IDs
            gtop["NumberOfPoints"]          = [PointCount]
            gtop["NumberOfCells"]           = [CellCount]
            gtop["NumberOfConnectivityIds"] = [PointCount]

            # Write Points
            T = typeof(SimKernel.H)
            @no_escape GridBuffer begin
                Points = @alloc(SVector{3, T}, PointCount)
                FillGridPoints!(Points, SimKernel, UniqueCells, Dimensions)
                gtop["Points"] = ThreeDimensionalView(Points)
                nothing
            end

            # Write Connectivity, Offsets, and Types
            gtop["Connectivity"] = collect(0:(PointCount - 1))
            gtop["Offsets"] = collect(offsets)
            gtop["Types"] = fill(vtk_type, CellCount)

            # Write CellData (cell-level variables)
            let cell_group = HDF5.create_group(gtop, "CellData")
                for (name, data) in cell_data_payload(
                    cell_ids,
                    cell_particle_counts,
                    cell_neighbor_counts,
                )
                    cell_group[name] = data
                end
                close(cell_group)
            end

            # Write an empty FieldData group (placeholder for additional data)
            field_group = create_group(gtop, "FieldData")
            close(field_group)
            close(gtop)
        finally
            isopen(io) && close(io)
        end
        return nothing
    end

    """
        SetupVTKOutput(SimMetaData, SimParticles, SimKernel, Dimensions)

    Prepare VTK/HDF5 output. Returns a named tuple with `enqueue_particles`,
    `enqueue_grid`, `flush_output`, and `close_files` functions. Output is
    written asynchronously using a background task with double-buffered
    particle snapshots. Two-dimensional vectors stay two-dimensional in those
    snapshots; the writer expands them into a reusable, Bumper-backed 3D arena
    only for the duration of each synchronous HDF5 write. The queue must be
    flushed before closing files. Uses single or multi-file mode depending on
    `SimMetaData.ExportSingleVTKHDF`.
    """
    function SetupVTKOutput(SimMetaData, SimParticles, SimKernel, Dimensions)
        # Generate save locations
        particle_savepath = joinpath(SimMetaData.SaveLocation, SimMetaData.SimulationName)
        grid_savepath = joinpath(SimMetaData.SaveLocation, "CellGrid_$(SimMetaData.SimulationName)")
    
        # File naming functions
        particle_filename = (iter) -> "$(particle_savepath)_$(lpad(iter,6,"0")).vtkhdf"
        grid_filename = (iter) -> "$(grid_savepath)_$(lpad(iter,6,"0")).vtkhdf"
        
        output_fields = filter(
            Name -> !(Name in (:Cells, :Position)),
            propertynames(SimParticles),
        )
        output_var_names = collect(String.(output_fields))
        n_output_fields = length(output_fields)
        # Keep output sources as a typed tuple so per-field copy logic can be specialized.
        output_sources = ntuple(Val(n_output_fields)) do i
            Field = output_fields[i]
            ResolveOutputSource(Val(Field), SimParticles)
        end
        output_schemas = ntuple(Val(n_output_fields)) do i
            OutputFieldSchema(Val(output_fields[i]), output_sources[i])
        end
        DimensionValue = Val(Dimensions)
    
        # Initialize storage for file handles
        file_handles = if !SimMetaData.ExportSingleVTKHDF
            # Static files are opened, written, and closed by the writer task.
            (
                particle_files = HDF5.File[],
                grid_files = nothing,
            )
        else
            # Single-file mode: handles for both files
            OutputVTKHDF = h5open("$(particle_savepath).vtkhdf", "w")
            root = HDF5.create_group(OutputVTKHDF, "VTKHDF")
            
            GenerateGeometryStructure(
                root,
                output_var_names,
                output_schemas...;
                chunk_size=1024,
                fType=VectorElementType(SimParticles.Position),
            )
            GenerateStepStructure(root, output_var_names, output_schemas...)
    
            # Initialize grid file if needed
            if SimMetaData.ExportGridCells
                OutputVTKHDFGrid = h5open("$(particle_savepath)_GridCells.vtkhdf", "w")
                root_grid = HDF5.create_group(OutputVTKHDFGrid, "VTKHDF")
                cell_data_names = ["CellData"]
                if SimMetaData.ExportGridCellParticleCounts
                    push!(cell_data_names, "ParticleCount")
                    push!(cell_data_names, "ParticleNeighborsPerCell")
                end
                GenerateGeometryStructure(
                    root_grid;
                    vtk_file_type="UnstructuredGrid",
                    cell_data_names=cell_data_names,
                    fType=typeof(SimKernel.H),
                )
                GenerateStepStructure(
                    root_grid;
                    vtk_file_type="UnstructuredGrid",
                    cell_data_names=cell_data_names,
                )
                
                (particle_files = OutputVTKHDF, grid_files = OutputVTKHDFGrid)
            else
                (particle_files = OutputVTKHDF, grid_files = nothing)
            end
        end

        function allocate_particle_snapshot()
            n = length(SimParticles.Position)
            positions = similar(SimParticles.Position, n)
            output_data = ntuple(Val(n_output_fields)) do i
                Field = output_fields[i]
                Source = output_sources[i]
                AllocateSnapshotBuffer(Val(Field), Source, n, Dimensions)
            end
            return ParticleSnapshot(positions, output_data)
        end

        function fill_particle_snapshot!(snapshot)
            copy!(snapshot.positions, SimParticles.Position)

            # Iterate via `ntuple` to preserve compile-time field information.
            ntuple(Val(n_output_fields)) do i
                Field = output_fields[i]
                Source = output_sources[i]
                Dest = snapshot.output_data[i]
                FillOutputField!(Val(Field), Dest, Source, Dimensions)
            end
            return snapshot
        end

        snapshot1 = allocate_particle_snapshot()
        snapshot2 = allocate_particle_snapshot()
        buffer_pool = Channel{typeof(snapshot1)}(2)
        put!(buffer_pool, snapshot1)
        put!(buffer_pool, snapshot2)

        ParticleJobType = typeof(ParticleWriteJob(0, SimMetaData.TotalTime, snapshot1))
        GridJobType = GridWriteJob{Dimensions, typeof(SimMetaData.TotalTime)}
        job_channel = Channel{Union{ParticleJobType, GridJobType}}(8)
        vtk_buffer_size = RequiredVTKBufferBytes(
            SimParticles.Position,
            output_sources,
            DimensionValue,
        )
        if SimMetaData.ExportGridCells
            vtk_buffer_size = max(
                vtk_buffer_size,
                RequiredGridBufferBytes(
                    SimKernel,
                    length(SimParticles.Position),
                    DimensionValue,
                ),
            )
        end
        writer_task = Threads.@spawn begin
            # This buffer belongs exclusively to the writer task. Each @no_escape
            # write restores its checkpoint, so the same storage is reused safely.
            VTKBuffer = AllocBuffer(vtk_buffer_size)
            for job in job_channel
                if job isa ParticleJobType
                    snapshot = job.snapshot
                    try
                        if !SimMetaData.ExportSingleVTKHDF
                            SaveVTKHDF(
                                file_handles.particle_files,
                                job.iteration,
                                particle_filename(job.iteration),
                                snapshot.positions,
                                output_var_names,
                                snapshot.output_data...,
                                Dimensions=DimensionValue,
                                Buffer=VTKBuffer,
                                CloseAfterWrite=true,
                            )
                        else
                            AppendVTKHDFData(
                                root,
                                job.time,
                                snapshot.positions,
                                output_var_names,
                                snapshot.output_data...,
                                Dimensions=DimensionValue,
                                Buffer=VTKBuffer,
                            )
                        end
                    finally
                        # Return every borrowed buffer even when HDF5 fails. The
                        # channel bindings below then wake blocked producers and
                        # surface the writer-task failure instead of deadlocking.
                        put!(buffer_pool, snapshot)
                    end
                elseif job isa GridJobType
                    if !SimMetaData.ExportSingleVTKHDF
                        SaveCellGridVTKHDF(
                            grid_filename(job.iteration),
                            SimKernel,
                            job.cells,
                            job.cell_particle_counts,
                            job.cell_neighbor_counts,
                            Dimensions=DimensionValue,
                            Buffer=VTKBuffer,
                        )
                    else
                        AppendVTKHDFGridData(
                            root_grid,
                            job.time,
                            SimKernel,
                            job.cells,
                            job.cell_particle_counts,
                            job.cell_neighbor_counts,
                            Dimensions=DimensionValue,
                            Buffer=VTKBuffer,
                        )
                    end
                end
            end
        end
        bind(job_channel, writer_task)
        bind(buffer_pool, writer_task)

        function enqueue_particle_data(iteration)
            snapshot = take!(buffer_pool)
            Enqueued = false
            try
                fill_particle_snapshot!(snapshot)
                job = ParticleWriteJob(iteration, SimMetaData.TotalTime, snapshot)
                put!(job_channel, job)
                Enqueued = true
            finally
                if !Enqueued && isopen(buffer_pool)
                    put!(buffer_pool, snapshot)
                end
            end
            return nothing
        end

        function enqueue_cell_grid(iteration, cells;
                                   cell_particle_counts = nothing,
                                   cell_neighbor_counts = nothing)
            if !SimMetaData.ExportGridCells
                return nothing
            end
            cells_snapshot = copy(cells)
            counts_snapshot = cell_particle_counts === nothing ? nothing :
                copy(cell_particle_counts)
            neighbors_snapshot = cell_neighbor_counts === nothing ? nothing :
                copy(cell_neighbor_counts)
            job = GridWriteJob{Dimensions, typeof(SimMetaData.TotalTime)}(
                iteration,
                SimMetaData.TotalTime,
                cells_snapshot,
                counts_snapshot,
                neighbors_snapshot,
            )
            put!(job_channel, job)
        end

        function flush_output()
            if isopen(job_channel)
                close(job_channel)
            end
            wait(writer_task)
        end

        function close_files()
            try
                flush_output()
            finally
                if SimMetaData.ExportSingleVTKHDF
                    isopen(file_handles.particle_files) && close(file_handles.particle_files)
                    if file_handles.grid_files !== nothing
                        isopen(file_handles.grid_files) && close(file_handles.grid_files)
                    end
                end
            end
            return nothing
        end
    
        # Return interface functions and handles
        return (
            enqueue_particles = enqueue_particle_data,
            enqueue_grid = enqueue_cell_grid,
            flush_output = flush_output,
            close_files = close_files,
            # Single-file mode exposes live handles until close. Multi-file mode
            # closes every completed file immediately and therefore returns [].
            file_handles = file_handles,
            variable_names = output_var_names
        )
    end

end
