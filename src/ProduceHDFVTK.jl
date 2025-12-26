
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
    using HDF5
    using StaticArrays

    using ..AuxiliaryFunctions: to_3d!


    const idType = Int64
    const fType = Float64

    struct ParticleSnapshot{P, V}
        positions::P
        output_data::V
    end

    struct ParticleWriteJob{P, V}
        iteration::Int
        time::Float64
        snapshot::ParticleSnapshot{P, V}
    end

    struct GridWriteJob{N}
        iteration::Int
        time::Float64
        cells::Vector{CartesianIndex{N}}
        cell_particle_counts::Union{Nothing, Vector{Int}}
    end

    """Write an ASCII attribute `name => value` to `grp`."""
    function write_ascii_attribute(grp, name, value)
        dtype = HDF5.datatype(value)
        HDF5.API.h5t_set_cset(dtype.id, HDF5.API.H5T_CSET_ASCII)
        dspace = HDF5.dataspace(value)
        attr = HDF5.create_attribute(grp, name, dtype, dspace)
        HDF5.write_attribute(attr, dtype, value)
    end

    """
    Compute points and connectivity information for an unstructured grid given
    `UniqueCells` from the SPH cell list.  Returns `(points, connectivity,
    offsets, cell_types, cell_ids, dims)` where `dims` is 2 or 3.
    """
    function compute_grid_geometry(SimKernel, UniqueCells)
        ExtractDimensionality(::AbstractVector{CartesianIndex{N}}) where N = N

        dims = ExtractDimensionality(UniqueCells)

        dx = dy = dz = SimKernel.H

        if dims == 2
            minx, maxx = minimum(ci -> ci[1], UniqueCells), maximum(ci -> ci[1], UniqueCells)
            miny, maxy = minimum(ci -> ci[2], UniqueCells), maximum(ci -> ci[2], UniqueCells)
            nx = maxx - minx + 1
        elseif dims == 3
            minx, maxx = minimum(ci -> ci[1], UniqueCells), maximum(ci -> ci[1], UniqueCells)
            miny, maxy = minimum(ci -> ci[2], UniqueCells), maximum(ci -> ci[2], UniqueCells)
            minz, maxz = minimum(ci -> ci[3], UniqueCells), maximum(ci -> ci[3], UniqueCells)
            nx = maxx - minx + 1
            ny = maxy - miny + 1
        else
            error("Dimensionality of UniqueCells must be 2 or 3, got $dims")
        end

        points       = Vector{SVector{3, Float64}}()
        connectivity = Int[]
        offsets      = Int[0]
        cell_types   = UInt8[]
        cell_ids     = Int[]

        vtk_type = dims == 2 ? UInt8(9) : UInt8(12)  # QUAD or HEXADRON

        for cell in UniqueCells
            id = if dims == 2
                (cell[2] - miny) * nx + (cell[1] - minx) + 1
            else
                (cell[3] - minz) * (nx * ny) + (cell[2] - miny) * nx + (cell[1] - minx) + 1
            end

            corners = if dims == 2
                xi, yi = cell.I
                x_c = xi * dx
                y_c = yi * dy
                [
                    SVector(x_c - dx/2, y_c - dy/2, 0.0),
                    SVector(x_c + dx/2, y_c - dy/2, 0.0),
                    SVector(x_c + dx/2, y_c + dy/2, 0.0),
                    SVector(x_c - dx/2, y_c + dy/2, 0.0),
                ]
            else
                xi, yi, zi = cell.I
                x_c = xi * dx
                y_c = yi * dy
                z_c = zi * dz
                [
                    SVector(x_c - dx/2, y_c - dy/2, z_c - dz/2),
                    SVector(x_c + dx/2, y_c - dy/2, z_c - dz/2),
                    SVector(x_c + dx/2, y_c + dy/2, z_c - dz/2),
                    SVector(x_c - dx/2, y_c + dy/2, z_c - dz/2),
                    SVector(x_c - dx/2, y_c - dy/2, z_c + dz/2),
                    SVector(x_c + dx/2, y_c - dy/2, z_c + dz/2),
                    SVector(x_c + dx/2, y_c + dy/2, z_c + dz/2),
                    SVector(x_c - dx/2, y_c + dy/2, z_c + dz/2),
                ]
            end

            n = length(points)
            append!(points, corners)
            for j = 0:length(corners)-1
                push!(connectivity, n + j)
            end
            push!(offsets, length(connectivity))
            push!(cell_types, vtk_type)
            push!(cell_ids, id)
        end

        return points, connectivity, offsets, cell_types, cell_ids, dims
    end

    function cell_data_payload(cell_ids, cell_particle_counts)
        payload = Dict("CellData" => cell_ids)
        if cell_particle_counts !== nothing
            payload["ParticleCount"] = cell_particle_counts
        end
        return payload
    end

    function SaveVTKHDF(fid_vector, index, filepath, points, variable_names = String[], args...)
        @assert length(variable_names) == length(args) "Same number of variable_names as args is necessary"
        io = h5open(filepath, "w")
        gtop = HDF5.create_group(io, "VTKHDF")

        HDF5.attrs(gtop)["Version"] = [2, 3]
        write_ascii_attribute(gtop, "Type", "PolyData")

        # Points
        np = length(points)
        gtop["NumberOfPoints"] = [np]
        gtop["Points"] = reinterpret(reshape, eltype(eltype(points)), points)

        # Point data
        let g = HDF5.create_group(gtop, "PointData")
            for i ∈ eachindex(variable_names)
                g[variable_names[i]] = reinterpret(reshape, eltype(eltype(args[i])), args[i])
            end
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

        fid_vector[index] = io
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
                arg_val_type   = eltype(eltype(arg))
                arg_val_length = length(first(arg)) > 1 ? 3 : 1

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

    function AppendVTKHDFData(root, newStep, Positions, variable_names, args...)
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

        HDF5.set_extent_dims(root["Points"], (length(first(Positions)), size(root["Points"])[2] + PositionLength))
        root["Points"][:, PointsStartIndex:(PointsStartIndex+PositionLength-1)] = stack(Positions)

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

        NumberOfPartsStartIndex = length(steps["NumberOfParts"]) + 1
        HDF5.set_extent_dims(steps["NumberOfParts"], (length(steps["NumberOfParts"]) + 1,))
        steps["NumberOfParts"][NumberOfPartsStartIndex] = 1

        for point_data_name in keys(steps["PointDataOffsets"])
            HDF5.set_extent_dims(steps["PointDataOffsets"][point_data_name], (length(steps["PointDataOffsets"][point_data_name]) + 1,))
            steps["PointDataOffsets"][point_data_name][end] = Int(PointsStartIndex - 1)
        end

        for i ∈ eachindex(variable_names)
            var_name = variable_names[i]
            arg      = args[i]
            arg_val_type   = eltype(eltype(arg))
            arg_val_length = length(first(arg)) > 1 ? 3 : 1


            if arg_val_length == 3
                HDF5.set_extent_dims(root["PointData"][var_name], (arg_val_length, size(root["PointData"][var_name], 2) + PositionLength))
                root["PointData"][var_name][:, PointsStartIndex:(PointsStartIndex + PositionLength - 1)] = stack(arg)
            else
                HDF5.set_extent_dims(root["PointData"][var_name], (length(root["PointData"][var_name]) + PositionLength,))
                root["PointData"][var_name][PointsStartIndex:(PointsStartIndex + PositionLength - 1)] = arg
            end
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
                                  cell_particle_counts = nothing)
        points, connectivity, offsets, cell_types, cell_ids, _ =
            compute_grid_geometry(SimKernel, UniqueCells)
        vtk_type = first(cell_types)

        

        Positions = points

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

        ExtendedPositionLength = size(root["Points"])[2] + PositionLength
        HDF5.set_extent_dims(root["Points"], (length(first(Positions)), ExtendedPositionLength))
        root["Points"][:, PointsStartIndex:(PointsStartIndex+PositionLength-1)] = stack(Positions)

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
        root["NumberOfPoints"][end] = PositionLength

        HDF5.set_extent_dims(root["NumberOfCells"], (length(root["NumberOfCells"]) + 1,))
        root["NumberOfCells"][end] = length(UniqueCells)

        ## Update data
        HDF5.set_extent_dims(root["Connectivity"], (ExtendedPositionLength,))
        root["Connectivity"][PointsStartIndex:end] = (PointsStartIndex:ExtendedPositionLength) .- PointsStartIndex

        HDF5.set_extent_dims(root["NumberOfConnectivityIds"], (length(root["NumberOfConnectivityIds"]) + 1,))
        root["NumberOfConnectivityIds"][end] = PositionLength



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

        cell_payload = cell_data_payload(cell_ids, cell_particle_counts)
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
                                cell_particle_counts = nothing)
        points, connectivity, offsets, cell_types, cell_ids, _ =
            compute_grid_geometry(SimKernel, UniqueCells)

        # Open HDF5 file for writing
        io = h5open(FilePath, "w")

        # Create top-level group "VTKHDF"
        gtop = HDF5.create_group(io, "VTKHDF")

        HDF5.attrs(gtop)["Version"] = [2, 3]
        write_ascii_attribute(gtop, "Type", "UnstructuredGrid")

        # Write Number of Points, Number of Cells, and Number of Connectivity IDs
        gtop["NumberOfPoints"]          = [length(points)]
        gtop["NumberOfCells"]           = [length(cell_types)]
        gtop["NumberOfConnectivityIds"] = [length(connectivity)]

        # Write Points
        gtop["Points"] = reinterpret(reshape, eltype(eltype(points)), points)

        # Write Connectivity, Offsets, and Types
        gtop["Connectivity"] = connectivity
        gtop["Offsets"] = offsets
        gtop["Types"] = fill(first(cell_types), length(cell_types))

        # Write CellData (cell-level variables)
        let cell_group = HDF5.create_group(gtop, "CellData")
            for (name, data) in cell_data_payload(cell_ids, cell_particle_counts)
                cell_group[name] = data
            end
            close(cell_group)
        end

        # Write an empty FieldData group (placeholder for additional data)
        create_group(gtop, "FieldData")
        
        # Close file
        close(io)
    end

    """
        SetupVTKOutput(SimMetaData, SimParticles, SimKernel, Dimensions)

    Prepare VTK/HDF5 output. Returns a named tuple with `enqueue_particles`,
    `enqueue_grid`, `flush_output`, and `close_files` functions. Output is
    written asynchronously using a background task with double-buffered
    particle snapshots, and the queue must be flushed before closing files.
    Uses single or multi-file mode depending on `SimMetaData.ExportSingleVTKHDF`.
    """
    function SetupVTKOutput(SimMetaData, SimParticles, SimKernel, Dimensions)
        # Generate save locations
        particle_savepath = joinpath(SimMetaData.SaveLocation, SimMetaData.SimulationName)
        grid_savepath = joinpath(SimMetaData.SaveLocation, "CellGrid_$(SimMetaData.SimulationName)")
    
        # File naming functions
        particle_filename = (iter) -> "$(particle_savepath)_$(lpad(iter,6,"0")).vtkhdf"
        grid_filename = (iter) -> "$(grid_savepath)_$(lpad(iter,6,"0")).vtkhdf"
        
        output_vars = SimMetaData.OutputVariables
    
        # Initialize storage for file handles
        file_handles = if !SimMetaData.ExportSingleVTKHDF
            # Multi-file mode: vector for particle files
            n_outputs = if SimMetaData.OutputTimes isa AbstractVector
                length(SimMetaData.OutputTimes) + 1
            else
                Int(SimMetaData.SimulationTime/SimMetaData.OutputTimes + 1)
            end
            (
                particle_files = Vector{HDF5.File}(undef, n_outputs),
                grid_files = nothing,
            )
        else
            # Single-file mode: handles for both files
            OutputVTKHDF = h5open("$(particle_savepath).vtkhdf", "w")
            root = HDF5.create_group(OutputVTKHDF, "VTKHDF")
            
            available_init = Dict(
                "Kernel" => SimParticles.Kernel,
                "KernelGradient" => SimParticles.KernelGradient,
                "Density" => SimParticles.Density,
                "Pressure" => SimParticles.Pressure,
                "Velocity" => SimParticles.Velocity,
                "Acceleration" => SimParticles.Acceleration,
                "BoundaryBool" => SimParticles.BoundaryBool,
                "ID" => SimParticles.ID,
                "Type" => Int8.(SimParticles.Type),
                "GroupMarker" => SimParticles.GroupMarker,
                "GhostPoints" => SimParticles.GhostPoints,
                "GhostNormals" => SimParticles.GhostNormals,
            )
            output_data_init = [available_init[name] for name in output_vars]

            GenerateGeometryStructure(root, output_vars, output_data_init...; chunk_size=1024)
            GenerateStepStructure(root, output_vars, output_data_init...)
    
            # Initialize grid file if needed
            if SimMetaData.ExportGridCells
                OutputVTKHDFGrid = h5open("$(particle_savepath)_GridCells.vtkhdf", "w")
                root_grid = HDF5.create_group(OutputVTKHDFGrid, "VTKHDF")
                cell_data_names = ["CellData"]
                if SimMetaData.ExportGridCellParticleCounts
                    push!(cell_data_names, "ParticleCount")
                end
                GenerateGeometryStructure(
                    root_grid;
                    vtk_file_type="UnstructuredGrid",
                    cell_data_names=cell_data_names,
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

        vector_fields = Set([
            "KernelGradient",
            "Velocity",
            "Acceleration",
            "GhostPoints",
            "GhostNormals",
        ])
        field_map = Dict(
            "Kernel" => :Kernel,
            "KernelGradient" => :KernelGradient,
            "Density" => :Density,
            "Pressure" => :Pressure,
            "Velocity" => :Velocity,
            "Acceleration" => :Acceleration,
            "BoundaryBool" => :BoundaryBool,
            "ID" => :ID,
            "Type" => :Type,
            "GroupMarker" => :GroupMarker,
            "GhostPoints" => :GhostPoints,
            "GhostNormals" => :GhostNormals,
        )

        T = eltype(eltype(SimParticles.Position))
        function allocate_particle_snapshot()
            n = length(SimParticles.Position)
            positions = Vector{SVector{3, T}}(undef, n)
            output_data = Vector{Any}(undef, length(output_vars))
            for (i, name) in pairs(output_vars)
                if name in vector_fields
                    output_data[i] = Vector{SVector{3, T}}(undef, n)
                elseif name == "Type"
                    output_data[i] = Vector{Int8}(undef, n)
                else
                    src = getproperty(SimParticles, field_map[name])
                    output_data[i] = similar(src, n)
                end
            end
            return ParticleSnapshot(positions, output_data)
        end

        function fill_particle_snapshot!(snapshot)
            if Dimensions == 2
                to_3d!(snapshot.positions, SimParticles.Position)
            else
                copy!(snapshot.positions, SimParticles.Position)
            end

            for (i, name) in pairs(output_vars)
                buf = snapshot.output_data[i]
                if name == "Type"
                    src = SimParticles.Type
                    @inbounds for j in eachindex(src)
                        buf[j] = Int8(src[j])
                    end
                elseif name in vector_fields
                    src = getproperty(SimParticles, field_map[name])
                    if Dimensions == 2
                        to_3d!(buf, src)
                    else
                        copy!(buf, src)
                    end
                else
                    src = getproperty(SimParticles, field_map[name])
                    copy!(buf, src)
                end
            end
            return snapshot
        end

        buffer_pool = Channel{ParticleSnapshot}(2)
        put!(buffer_pool, allocate_particle_snapshot())
        put!(buffer_pool, allocate_particle_snapshot())

        job_channel = Channel{Any}(8)
        writer_task = Threads.@spawn begin
            for job in job_channel
                if job isa ParticleWriteJob
                    snapshot = job.snapshot
                    if !SimMetaData.ExportSingleVTKHDF
                        SaveVTKHDF(
                            file_handles.particle_files,
                            job.iteration,
                            particle_filename(job.iteration),
                            snapshot.positions,
                            output_vars,
                            snapshot.output_data...,
                        )
                    else
                        AppendVTKHDFData(
                            root,
                            job.time,
                            snapshot.positions,
                            output_vars,
                            snapshot.output_data...,
                        )
                    end
                    put!(buffer_pool, snapshot)
                elseif job isa GridWriteJob
                    if !SimMetaData.ExportSingleVTKHDF
                        SaveCellGridVTKHDF(
                            grid_filename(job.iteration),
                            SimKernel,
                            job.cells,
                            job.cell_particle_counts,
                        )
                    else
                        AppendVTKHDFGridData(
                            root_grid,
                            job.time,
                            SimKernel,
                            job.cells,
                            job.cell_particle_counts,
                        )
                    end
                end
            end
        end

        function enqueue_particle_data(iteration)
            snapshot = take!(buffer_pool)
            fill_particle_snapshot!(snapshot)
            job = ParticleWriteJob(iteration, SimMetaData.TotalTime, snapshot)
            put!(job_channel, job)
        end

        function enqueue_cell_grid(iteration, cells; cell_particle_counts = nothing)
            if !SimMetaData.ExportGridCells
                return nothing
            end
            cells_snapshot = copy(cells)
            counts_snapshot = cell_particle_counts === nothing ? nothing :
                copy(cell_particle_counts)
            job = GridWriteJob{Dimensions}(
                iteration,
                SimMetaData.TotalTime,
                cells_snapshot,
                counts_snapshot,
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
            flush_output()
            if !SimMetaData.ExportSingleVTKHDF
                # Close all particle files in multi-file mode
                for f in file_handles.particle_files
                    isopen(f) && close(f)
                end
            else
                # Close single-file handles
                isopen(file_handles.particle_files) && close(file_handles.particle_files)
                if file_handles.grid_files !== nothing
                    isopen(file_handles.grid_files) && close(file_handles.grid_files)
                end
            end
        end
    
        # Return interface functions and handles
        return (
            enqueue_particles = enqueue_particle_data,
            enqueue_grid = enqueue_cell_grid,
            flush_output = flush_output,
            close_files = close_files,
            file_handles = file_handles,  # For advanced access if needed
            variable_names = output_vars
        )
    end

end
