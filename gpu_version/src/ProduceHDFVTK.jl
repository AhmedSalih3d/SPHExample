
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
       AppendVTKHDFData, SavePolygonVTKHDF, SaveCellGridVTKHDF, AppendVTKHDFGridData,
       GridGeometryBuffers, fill_grid_geometry!, GridFrameWriter, append_grid_frame!,
       PolyDataFrameWriter, append_frame!, flush_frames!, frames_written, frames_pending,
       buffered_frames, MAX_BUFFERED_FRAMES,
       SetupVTKOutput

    using HDF5
    using Meshes
    using StaticArrays

    using ..AuxiliaryFunctions: to_3d, to_3d!, components!
    using ..PolygonDrawing: ExtrudedPolygon


    const idType = Int64
    const fType = Float64

    """Write an ASCII attribute `name => value` to `grp`."""
    function write_ascii_attribute(grp, name, value)
        dtype = HDF5.datatype(value)
        HDF5.API.h5t_set_cset(dtype.id, HDF5.API.H5T_CSET_ASCII)
        dspace = HDF5.dataspace(value)
        attr = HDF5.create_attribute(grp, name, dtype, dspace)
        HDF5.write_attribute(attr, dtype, value)
    end

    # Point data arguments are either vectors (of scalars or `SVector`s) or a
    # preallocated `3 × N` component matrix (see `components!`). These helpers
    # give the number of points, the components per point, the component type
    # and the data in the layout of the file for both forms; `component_view`
    # reinterprets a vector without copying (whole dataset writes),
    # `component_data` copies a vector of `SVector`s (hyperslab writes).
    npoints(arg::AbstractMatrix)        = size(arg, 2)
    npoints(arg::AbstractVector)        = length(arg)
    ncomponents(arg::AbstractMatrix)    = size(arg, 1)
    ncomponents(arg::AbstractVector)    = eltype(arg) <: StaticVector ? 3 : 1
    component_type(arg::AbstractArray)  = eltype(eltype(arg))
    component_view(arg::AbstractMatrix) = arg
    component_view(arg::AbstractVector) = reinterpret(reshape, component_type(arg), arg)
    component_data(arg::AbstractMatrix) = arg
    component_data(arg::AbstractVector) = ncomponents(arg) == 3 ? stack(arg) : arg

    """
    Compute points and connectivity information for an unstructured grid given
    `UniqueCells` from the SPH cell list.  Returns `(points, connectivity,
    offsets, cell_types, cell_data, dims)` where `dims` is 2 or 3.
    """
    # Edge length of the exported cells: `H` divided by the number of cells
    # per support radius of the GPU grid (`GPUCellSubdivision`).
    grid_cell_edge(SimKernel, SimMetaData) = SimKernel.H / SimMetaData.GPUCellSubdivision

    function compute_grid_geometry(cell_edge::Real, UniqueCells)
        ExtractDimensionality(::AbstractVector{CartesianIndex{N}}) where N = N

        dims = ExtractDimensionality(UniqueCells)

        dx = dy = dz = cell_edge

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
        cell_data    = Int[]

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
            push!(cell_data, id)
        end

        return points, connectivity, offsets, cell_types, cell_data, dims
    end

    function SaveVTKHDF(fid_vector, index, filepath, points, variable_names = String[], args...)
        @assert length(variable_names) == length(args) "Same number of variable_names as args is necessary"
        io = h5open(filepath, "w")
        gtop = HDF5.create_group(io, "VTKHDF")

        HDF5.attrs(gtop)["Version"] = [2, 3]
        write_ascii_attribute(gtop, "Type", "PolyData")

        # Points
        np = npoints(points)
        gtop["NumberOfPoints"] = [np]
        gtop["Points"] = component_view(points)

        # Point data
        let g = HDF5.create_group(gtop, "PointData")
            for i ∈ eachindex(variable_names)
                g[variable_names[i]] = component_view(args[i])
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

    """
        SaveVTKHDF(filepath, points, variable_names = String[], args...)

    Write one static `PolyData` file of vertices (one cell per point) with the
    point data `variable_names => args` and close it. Returns `filepath`.
    """
    function SaveVTKHDF(filepath::AbstractString, points, variable_names = String[], args...)
        files = Vector{HDF5.File}(undef, 1)
        SaveVTKHDF(files, 1, filepath, points, variable_names, args...)
        close(files[1])
        return filepath
    end

    """
        SavePolygonVTKHDF(filepath, regions)

    Write the polygons of the named tuple `regions` as one static VTKHDF
    `PolyData` file. Every region is a `PolyArea`, a `Multi` of `PolyArea`s, an
    `ExtrudedPolygon` (see `prism`) or a tuple or vector of those. The polygons
        without holes are written as polygon cells; polygons with holes are
        triangulated because a VTK polygon cell cannot encode interior rings.
        Prisms are written as closed surfaces, with polygonal caps and quad sides
        when possible. The cells carry the position of their region in the tuple
        in the cell data array `Region`. 2D coordinates are written in the XY
        plane with `z = 0`. Returns `filepath`.
        """
        function SavePolygonVTKHDF(filepath::AbstractString, regions::NamedTuple)
            isempty(regions) && throw(ArgumentError("at least one polygon region is required"))
            points, connectivity, offsets, region_ids = polygonal_regions(values(regions))

        mkpath(dirname(abspath(filepath)))
        h5open(filepath, "w") do io
            root = HDF5.create_group(io, "VTKHDF")
            HDF5.attrs(root)["Version"] = Int32[2, 3]
            write_ascii_attribute(root, "Type", "PolyData")
            root["NumberOfPoints"] = [length(points)]
            root["Points"]         = stack(points)

            write_polydata_cells(root, "Polygons", connectivity, offsets)
            for cell_type in ("Vertices", "Lines", "Strips")
                write_polydata_cells(root, cell_type, Int64[], Int64[0])
            end
            HDF5.create_group(root, "CellData")["Region"] = region_ids
        end
        return filepath
    end

    """
    Convert the polygons of all `regions` into one point list with VTK style
    (zero based) connectivity and offsets. Simple polygons are kept as single
    cells, while polygons with holes are triangulated. Prism caps are
    triangulated only when their base has holes; prism sides are quads.
    """
    function polygonal_regions(regions)
        points       = SVector{3, Float64}[]
        connectivity = Int64[]
        offsets      = Int64[0]
        region_ids   = Int32[]

        add_cell!(ids, id) = begin
            append!(connectivity, ids .- 1)
            push!(offsets, length(connectivity))
            push!(region_ids, id)
        end

        for (id, region) in enumerate(regions), (polygon, z) in surfaces_of(region)
            polygon_rings = collect(rings(polygon))
            base_z      = z === nothing ? 0.0 : z[1]

            if length(polygon_rings) == 1
                ring = only(polygon_rings)
                if z === nothing
                    ids = append_ring_points!(points, ring, base_z)
                    add_cell!(counter_clockwise(points, ids), id)
                    continue
                end

                bottom_ids = append_ring_points!(points, ring, z[1])
                top_ids = append_ring_points!(points, ring, z[2])
                add_cell!(reverse(counter_clockwise(points, bottom_ids)), id)
                add_cell!(counter_clockwise(points, top_ids), id)
                for i in eachindex(bottom_ids)
                    next = mod1(i + 1, length(bottom_ids))
                    add_cell!([bottom_ids[i], bottom_ids[next], top_ids[next],
                               top_ids[i]], id)
                end
                continue
            end

            # VTK polygon cells have a single ring, so preserve holes by
            # triangulating only these faces.
            mesh        = discretize(polygon)
            first_point = length(points)
            for vertex in vertices(mesh)
                x, y = Meshes.ustrip.(to(vertex))
                push!(points, SVector(x, y, base_z))
            end
            for triangle in elements(topology(mesh))
                ids = counter_clockwise(points, first_point .+ collect(indices(triangle)))
                # A prism's bottom lid faces down.
                add_cell!(z === nothing ? ids : reverse(ids), id)
            end
            z === nothing && continue

            # Prism: a triangulated top lid and one quad per outline edge.
            top_first = length(points)
            for k in 1:nvertices(mesh)
                bottom = points[first_point + k]
                push!(points, SVector(bottom[1], bottom[2], z[2]))
            end
            for triangle in elements(topology(mesh))
                ids = top_first .+ collect(indices(triangle))
                add_cell!(counter_clockwise(points, ids), id)
            end
            for ring in rings(polygon), segment in segments(ring)
                a, b = (Meshes.ustrip.(to(v)) for v in vertices(segment))
                corner = length(points)
                push!(points, SVector(a[1], a[2], z[1]), SVector(b[1], b[2], z[1]),
                              SVector(b[1], b[2], z[2]), SVector(a[1], a[2], z[2]))
                add_cell!(corner .+ [1, 2, 3, 4], id)
            end
        end
        isempty(points) && throw(ArgumentError("polygon regions have no points"))
        return points, connectivity, offsets, region_ids
    end

    function append_ring_points!(points, ring, z)
        first_point = length(points)
        for vertex in vertices(ring)
            x, y = Meshes.ustrip.(to(vertex))
            push!(points, SVector(x, y, z))
        end
        return collect((first_point + 1):length(points))
    end

    # `(polygon, nothing)` for flat polygons, `(polygon, (bottom, top))` for prisms.
    surfaces_of(region::PolyArea) = ((region, nothing),)
    surfaces_of(region::Multi) =
        ((p, nothing) for p in polygons_of_multi(parent(region)))
    surfaces_of(region::ExtrudedPolygon) =
        ((p, (region.bottom, region.top)) for (p, _) in surfaces_of(region.base))
    function surfaces_of(region::Union{Tuple, AbstractVector})
        isempty(region) && throw(ArgumentError("polygon regions cannot be empty"))
        return Iterators.flatten(map(surfaces_of, collect(region)))
    end
    surfaces_of(region) =
        throw(ArgumentError("each region must be a PolyArea, a Multi of PolyAreas, an " *
                            "ExtrudedPolygon or a tuple or vector of those, got $(typeof(region))"))

    function polygons_of_multi(polygons)
        isempty(polygons) && throw(ArgumentError("polygon regions cannot be empty"))
        all(p -> p isa PolyArea, polygons) ||
            throw(ArgumentError("Multi regions may only contain PolyAreas"))
        return polygons
    end

    # One based polygon `ids` into `points`, reversed if its XY area is clockwise.
    function counter_clockwise(points, ids)
        twice_area = sum(
            points[ids[i]][1] * points[ids[mod1(i + 1, length(ids))]][2] -
            points[ids[mod1(i + 1, length(ids))]][1] * points[ids[i]][2]
            for i in eachindex(ids)
        )
        return twice_area < 0 ? reverse(ids) : ids
    end

    """Write one PolyData connectivity group (`Vertices`, `Lines`, `Polygons` or `Strips`)."""
    function write_polydata_cells(root, name, connectivity, offsets)
        group = HDF5.create_group(root, name)
        group["NumberOfCells"]           = [length(offsets) - 1]
        group["NumberOfConnectivityIds"] = [length(connectivity)]
        group["Connectivity"]            = connectivity
        group["Offsets"]                 = offsets
        return group
    end


    function GenerateGeometryStructure(root, variable_names = String[], args...; chunk_size = 100, vtk_file_type = "PolyData", idType = Int64, fType = Float64,
                                       count_chunk_size = 1000)
        @assert length(variable_names) == length(args) "Same number of variable_names as args is necessary"
        # Write version of VTKHDF format as an attribute
        HDF5.attrs(root)["Version"] = Int32.([2, 3])

        write_ascii_attribute(root, "Type", vtk_file_type)

        # `chunk_size` chunks the per point datasets; the datasets with one
        # entry per frame (the point counts and the connectivity groups of the
        # PolyData) take `count_chunk_size`, since a chunk is stored whole even
        # when only a few entries are used.
        NumberOfPoints = HDF5.create_dataset(root, "NumberOfPoints" , idType , ((0,),(-1,)), chunk=(count_chunk_size,))
        Points         = HDF5.create_dataset(root, "Points"         , fType  , ((3,0), (3,-1)), chunk=(3,chunk_size))

        if vtk_file_type == "PolyData"
            connectivities = ["Vertices", "Lines", "Polygons", "Strips"]
            for connect in connectivities
                group = HDF5.create_group(root, connect)

                NumberOfConnectivityIds = HDF5.create_dataset(group, "NumberOfConnectivityIds", idType, ((0,),(-1,)), chunk=(count_chunk_size,) )
                NumberOfCells           = HDF5.create_dataset(group, "NumberOfCells", idType, ((0,),(-1,)), chunk=(count_chunk_size,) )
                Offsets                 = HDF5.create_dataset(group, "Offsets", idType, ((0,),(-1,)), chunk=(count_chunk_size,) )
                Connectivity            = HDF5.create_dataset(group, "Connectivity", idType, ((0,),(-1,)), chunk=(count_chunk_size,) )
            end

            pData = HDF5.create_group(root, "PointData")

            for i ∈ eachindex(variable_names)
                var_name = variable_names[i]
                arg      = args[i]
                arg_val_type   = component_type(arg)
                arg_val_length = ncomponents(arg)

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
            HDF5.create_dataset(CellData, "CellData" , idType , ((0,),(-1,)), chunk=(chunk_size,))
        end

        return nothing
    end

    function GenerateStepStructure(root,  variable_names = String[], args...; vtk_file_type = "PolyData", chunk_size = 1000)
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
        end
            
        pData = HDF5.create_group(steps, "PointDataOffsets")


        for i ∈ eachindex(variable_names)
            var_name = variable_names[i]

            HDF5.create_dataset(pData, var_name, idType, ((0,),(-1,)), chunk=(chunk_size,))
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
        PositionLength   = npoints(Positions)

        HDF5.set_extent_dims(root["Points"], (ncomponents(Positions), size(root["Points"])[2] + PositionLength))
        root["Points"][:, PointsStartIndex:(PointsStartIndex+PositionLength-1)] = component_data(Positions)

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
            arg_val_type   = component_type(arg)
            arg_val_length = ncomponents(arg)


            if arg_val_length == 3
                HDF5.set_extent_dims(root["PointData"][var_name], (arg_val_length, size(root["PointData"][var_name], 2) + PositionLength))
                root["PointData"][var_name][:, PointsStartIndex:(PointsStartIndex + PositionLength - 1)] = component_data(arg)
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

    #---------------------------------------------------------------
    # Buffered transient writers
    #---------------------------------------------------------------
    #
    # Appending one frame to a transient file with `AppendVTKHDFData` or
    # `AppendVTKHDFGridData` costs some 35 tiny HDF5 operations next to the
    # writes of the frame data: every bookkeeping dataset (step values,
    # offsets, counts, the four empty connectivity groups) is looked up by
    # name, extended by one element and written, and the step count
    # attribute is read and rewritten. At about 20 µs each this bookkeeping
    # was most of the cost of a small frame, and with an output interval of
    # a few milliseconds of GPU time the writer could not keep up. The
    # buffered writers below keep the dataset handles open, collect
    # `capacity` frames in host memory and write them in one go: one extent
    # and one hyperslab write per dataset, whole slices for the bookkeeping.
    # Frames still pending at the end are written by `flush_frames!`, which
    # `SetupVTKOutput`'s `close_files` calls.

    """
    Upper bound on the frames a buffered writer holds before a flush. The
    flush blocks the writer, and the simulation thread once every set of
    staging buffers is in flight: on the RTX A1000 a flush of 16 frames of the
    2D still wedge (3 027 particles) hides behind an output interval of
    2.3 ms of GPU time, a flush of 39 frames stalls the simulation. Larger
    batches hardly reduce the cost per frame any further.
    """
    const MAX_BUFFERED_FRAMES = 16

    """
        buffered_frames(bytes_per_frame, budget_bytes) -> Int

    Frames to hold in memory for a host memory budget: the budget divided by
    the size of a frame, rounded down to a power of two, at least 1 and at
    most `MAX_BUFFERED_FRAMES`. A power of two lets the dataset chunks hold a
    divisor of the frames per flush, so that a flush writes whole chunks.
    """
    buffered_frames(bytes_per_frame::Integer, budget_bytes::Integer) =
        prevpow(2, clamp(budget_bytes ÷ max(bytes_per_frame, 1), 1, MAX_BUFFERED_FRAMES))

    # Append `n` entries to the one dimensional dataset `ds` and return the
    # (one based) index of the first new entry.
    function extend!(ds, n::Integer)
        len = length(ds)
        HDF5.set_extent_dims(ds, (len + n,))
        return len + 1
    end

    # Append `n` columns to the `k × M` dataset `ds` and return the (one
    # based) index of the first new column.
    function extend_columns!(ds, n::Integer)
        rows, cols = size(ds)
        HDF5.set_extent_dims(ds, (rows, cols + n))
        return cols + 1
    end

    # Append the entries of `data` to the one dimensional dataset `ds`.
    function append_slice!(ds, data::AbstractVector)
        m = length(data)
        m == 0 && return ds
        i = extend!(ds, m)
        ds[i:(i + m - 1)] = data
        return ds
    end

    # Append the columns of `data` to the `k × M` dataset `ds`.
    function append_columns!(ds, data::AbstractMatrix)
        m = size(data, 2)
        m == 0 && return ds
        i = extend_columns!(ds, m)
        ds[:, i:(i + m - 1)] = data
        return ds
    end

    set_nsteps!(attr::HDF5.Attribute, n::Integer) = HDF5.write_attribute(attr, HDF5.datatype(idType), idType(n))

    # Fill a frame slot of a buffer from a particle field: the `3 × N`
    # component layout for vectors of `SVector`s, an element-wise conversion
    # (`Type` to `Int8`, or the identity) for scalars.
    fill_field!(dest::AbstractMatrix, src::AbstractVector{<:SVector}) = components!(dest, src)
    fill_field!(dest::AbstractMatrix, src::AbstractMatrix)            = copyto!(dest, src)
    fill_field!(dest::AbstractVector{T}, src::AbstractVector) where {T} = map!(T, dest, src)

    """
    Buffered writer of particle frames to a transient PolyData file created
    with `GenerateGeometryStructure` and `GenerateStepStructure`. Holds the
    dataset handles and up to `capacity` frames of `n` points each in host
    memory (`positions` and one array per point data variable, laid out
    frame after frame so that a flush writes one hyperslab per dataset). The
    buffers have the element types of the file (the positions `fType`), so
    that a full flush hands the arrays to HDF5 without a conversion or copy.
    Produces the same file as `AppendVTKHDFData` called once per frame.
    """
    mutable struct PolyDataFrameWriter
        const root::HDF5.Group
        const nsteps::HDF5.Attribute
        const points::HDF5.Dataset
        const number_of_points::HDF5.Dataset
        const values::HDF5.Dataset
        const point_offsets::HDF5.Dataset
        const number_of_parts::HDF5.Dataset
        const part_offsets::HDF5.Dataset
        const topology_offsets::Vector{HDF5.Dataset}   # `4 × NSteps` CellOffsets and ConnectivityIdOffsets, all zero
        const zero_counts::Vector{HDF5.Dataset}        # the datasets of the four connectivity groups, all zero
        const point_data::Vector{HDF5.Dataset}         # one per variable
        const point_data_offsets::Vector{HDF5.Dataset} # one per variable
        const n::Int                                    # points per frame
        const capacity::Int                             # frames held before a flush
        const times::Vector{Float64}                    # step value of every pending frame
        const positions::Matrix{fType}                  # `3 × (capacity · n)`
        const data::Vector{Array}                       # per variable `3 × (capacity · n)` or `capacity · n`
        pending::Int                                    # frames in the buffers
        nframes::Int                                    # frames in the file
    end

    """
        PolyDataFrameWriter(root, Positions, variable_names, args...; capacity = 1)

    Writer for the file structure below `root`; `Positions` and `args` are
    the arrays of the first frame and only provide the point count, the
    component counts and the element types.
    """
    function PolyDataFrameWriter(root::HDF5.Group, Positions, variable_names, args...; capacity::Integer = 1)
        length(variable_names) == length(args) || throw(ArgumentError("one array per variable name is required"))
        capacity >= 1 || throw(ArgumentError("capacity must be at least 1, got $capacity"))
        n     = npoints(Positions)
        steps = root["Steps"]
        pdata = root["PointData"]
        poffs = steps["PointDataOffsets"]
        data  = Array[ncomponents(a) == 3 ? Matrix{component_type(a)}(undef, 3, capacity * n) :
                                            Vector{component_type(a)}(undef, capacity * n) for a in args]
        zero_counts = HDF5.Dataset[root[c][d] for c in ("Vertices", "Lines", "Polygons", "Strips")
                                             for d in ("NumberOfCells", "NumberOfConnectivityIds", "Offsets", "Connectivity")]
        return PolyDataFrameWriter(root, attributes(steps)["NSteps"], root["Points"], root["NumberOfPoints"],
                                   steps["Values"], steps["PointOffsets"], steps["NumberOfParts"], steps["PartOffsets"],
                                   HDF5.Dataset[steps["CellOffsets"], steps["ConnectivityIdOffsets"]], zero_counts,
                                   HDF5.Dataset[pdata[v] for v in variable_names], HDF5.Dataset[poffs[v] for v in variable_names],
                                   n, capacity, Float64[], Matrix{fType}(undef, 3, capacity * n), data, 0, 0)
    end

    # Buffer columns of the frame that is appended next.
    next_columns(w::PolyDataFrameWriter) = (w.pending * w.n + 1):((w.pending + 1) * w.n)

    """
        append_frame!(writer, newStep, Positions, args...)

    Buffer the frame at time `newStep` (`Positions` and one array per variable,
    in the order of the writer's variable names) and write the buffered frames
    to the file once `capacity` frames are pending.
    """
    function append_frame!(w::PolyDataFrameWriter, newStep, Positions, args...)
        length(args) == length(w.data) || throw(ArgumentError("expected $(length(w.data)) point data arrays, got $(length(args))"))
        npoints(Positions) == w.n || throw(DimensionMismatch("frame of $(npoints(Positions)) points for a writer of $(w.n) points"))
        cols = next_columns(w)
        fill_field!(view(w.positions, :, cols), Positions)
        for (buf, arg) in zip(w.data, args)
            fill_field!(buf isa AbstractMatrix ? view(buf, :, cols) : view(buf, cols), arg)
        end
        push!(w.times, Float64(newStep))
        w.pending += 1
        w.pending == w.capacity && flush_frames!(w)
        return w
    end

    """
        flush_frames!(writer)

    Write the pending frames to the file: one hyperslab write per point
    dataset and whole slices of the bookkeeping datasets. A full buffer is
    handed to HDF5 as is (HDF5.jl copies any other array type); the partial
    buffer of a final flush is copied once.
    """
    function flush_frames!(w::PolyDataFrameWriter)
        k = w.pending
        k == 0 && return w
        n, f0   = w.n, w.nframes
        full    = k == w.capacity
        cols    = 1:(k * n)
        frames  = f0:(f0 + k - 1)
        offsets = collect(frames .* n)
        append_columns!(w.points, full ? w.positions : w.positions[:, cols])
        for (ds, buf) in zip(w.point_data, w.data)
            if buf isa AbstractMatrix
                append_columns!(ds, full ? buf : buf[:, cols])
            else
                append_slice!(ds, full ? buf : buf[cols])
            end
        end
        append_slice!(w.values, w.times)
        append_slice!(w.point_offsets, offsets)
        append_slice!(w.number_of_points, fill(idType(n), k))
        append_slice!(w.number_of_parts, fill(idType(1), k))
        append_slice!(w.part_offsets, collect(frames))
        foreach(ds -> append_slice!(ds, offsets), w.point_data_offsets)
        foreach(ds -> extend_columns!(ds, k), w.topology_offsets)  # new columns are zero
        foreach(ds -> extend!(ds, k), w.zero_counts)               # new entries are zero
        set_nsteps!(w.nsteps, f0 + k)
        w.nframes += k
        w.pending  = 0
        empty!(w.times)
        return w
    end

    frames_written(w::PolyDataFrameWriter) = w.nframes
    frames_pending(w::PolyDataFrameWriter) = w.pending

    """
    Preallocated SPHGeometry of one cell grid frame in the layout of the file
    (see `fill_grid_geometry!`): the corner points as a `3 × (corners · cells)`
    matrix, the connectivity, the cell offsets, the VTK cell types and the
    cell ids. The arrays are resized per frame and only grow.
    """
    mutable struct GridGeometryBuffers
        points::Matrix{Float64}
        connectivity::Vector{Int}
        offsets::Vector{Int}
        types::Vector{UInt8}
        cell_data::Vector{Int}
    end
    GridGeometryBuffers() = GridGeometryBuffers(Matrix{Float64}(undef, 3, 0), Int[], Int[], UInt8[], Int[])

    """
        fill_grid_geometry!(buf, cell_edge, UniqueCells) -> npoints

    Fill `buf` with the SPHGeometry of the cells `UniqueCells` (edge length
    `cell_edge`): the same corners, connectivity, offsets, types and ids as
    `compute_grid_geometry`, without allocating per cell. Returns the number
    of corner points; `buf.points` has exactly that many columns.
    """
    function fill_grid_geometry!(buf::GridGeometryBuffers, cell_edge::Real, UniqueCells::AbstractVector{CartesianIndex{D}}) where {D}
        D == 2 || D == 3 || error("Dimensionality of UniqueCells must be 2 or 3, got $D")
        ncells   = length(UniqueCells)
        ncorners = 2^D
        npts     = ncorners * ncells
        size(buf.points, 2) == npts || (buf.points = Matrix{Float64}(undef, 3, npts))
        resize!(buf.connectivity, npts)
        resize!(buf.offsets, ncells + 1)
        resize!(buf.types, ncells)
        resize!(buf.cell_data, ncells)
        buf.offsets[1] = 0
        ncells == 0 && return 0

        mins = ntuple(d -> minimum(ci -> ci[d], UniqueCells), Val(D))
        maxs = ntuple(d -> maximum(ci -> ci[d], UniqueCells), Val(D))
        nx   = maxs[1] - mins[1] + 1
        ny   = D == 3 ? maxs[2] - mins[2] + 1 : 1
        edge = Float64(cell_edge)
        h    = edge / 2
        vtk_type = D == 2 ? UInt8(9) : UInt8(12)   # QUAD or HEXAHEDRON

        @inbounds for c in 1:ncells
            cell = UniqueCells[c]
            base = (c - 1) * ncorners
            id = if D == 2
                (cell[2] - mins[2]) * nx + (cell[1] - mins[1]) + 1
            else
                (cell[3] - mins[3]) * (nx * ny) + (cell[2] - mins[2]) * nx + (cell[1] - mins[1]) + 1
            end
            xc = cell[1] * edge
            yc = cell[2] * edge
            zc = D == 3 ? cell[3] * edge : 0.0
            # corner order per layer: (-,-), (+,-), (+,+), (-,+); the second layer (3D) is at z+
            for k in 0:ncorners - 1
                kk = k & 3
                sx = (kk == 1 || kk == 2) ? h : -h
                sy = kk >= 2 ? h : -h
                pt = base + k + 1
                buf.points[1, pt] = xc + sx
                buf.points[2, pt] = yc + sy
                buf.points[3, pt] = D == 3 ? zc + (k >= 4 ? h : -h) : 0.0
                buf.connectivity[pt] = pt - 1
            end
            buf.offsets[c + 1] = base + ncorners
            buf.types[c]       = vtk_type
            buf.cell_data[c]   = id
        end
        return npts
    end

    """
    Buffered writer of cell grid frames to the transient (single file) grid
    output. Holds the dataset handles, the SPHGeometry buffers of the frame being
    appended and up to `capacity` pending frames (their SPHGeometry appended to
    growing host arrays, since the cell count varies per frame) together with
    the totals of the file (the `PointOffsets` and `CellOffsets` of the next
    frame). Produces the same file as `AppendVTKHDFGridData` called once per
    frame, without allocating per cell or re-reading previous frames.
    """
    mutable struct GridFrameWriter
        const root::HDF5.Group
        const nsteps::HDF5.Attribute
        const points::HDF5.Dataset
        const number_of_points::HDF5.Dataset
        const number_of_cells::HDF5.Dataset
        const number_of_connectivity_ids::HDF5.Dataset
        const connectivity::HDF5.Dataset
        const offsets::HDF5.Dataset
        const types::HDF5.Dataset
        const cell_data::HDF5.Dataset
        const values::HDF5.Dataset
        const point_offsets::HDF5.Dataset
        const number_of_parts::HDF5.Dataset
        const part_offsets::HDF5.Dataset
        const cell_offsets::HDF5.Dataset
        const connectivity_id_offsets::HDF5.Dataset
        const buf::GridGeometryBuffers      # SPHGeometry of the frame being appended
        const capacity::Int                 # frames held before a flush
        # pending frames: per frame values and the concatenated SPHGeometry
        const times::Vector{Float64}
        const frame_npts::Vector{Int}
        const frame_ncells::Vector{Int}
        const pts::Vector{Float64}          # `3 × Σ npts` column major
        const conn::Vector{Int}
        const offs::Vector{Int}
        const typ::Vector{UInt8}
        const cdata::Vector{Int}
        pending::Int                        # frames in the buffers
        nframes::Int                        # frames in the file
        npts_total::Int                     # points in the file
        ncells_total::Int                   # cells in the file
    end

    function GridFrameWriter(root::HDF5.Group; capacity::Integer = 1)
        capacity >= 1 || throw(ArgumentError("capacity must be at least 1, got $capacity"))
        steps = root["Steps"]
        return GridFrameWriter(root, attributes(steps)["NSteps"], root["Points"], root["NumberOfPoints"], root["NumberOfCells"],
                               root["NumberOfConnectivityIds"], root["Connectivity"], root["Offsets"], root["Types"],
                               root["CellData"]["CellData"], steps["Values"], steps["PointOffsets"], steps["NumberOfParts"],
                               steps["PartOffsets"], steps["CellOffsets"], steps["ConnectivityIdOffsets"],
                               GridGeometryBuffers(), capacity,
                               Float64[], Int[], Int[], Float64[], Int[], Int[], UInt8[], Int[], 0, 0, 0, 0)
    end

    """
        append_grid_frame!(writer, newStep, cell_edge, UniqueCells)

    Buffer the cells `UniqueCells` at time `newStep` as one frame of the
    transient grid file (same layout as `AppendVTKHDFGridData`) and write the
    buffered frames once `capacity` frames are pending.
    """
    function append_grid_frame!(w::GridFrameWriter, newStep, cell_edge::Real, UniqueCells)
        npts = fill_grid_geometry!(w.buf, cell_edge, UniqueCells)
        append!(w.pts,   vec(w.buf.points))
        append!(w.conn,  w.buf.connectivity)
        append!(w.offs,  w.buf.offsets)
        append!(w.typ,   w.buf.types)
        append!(w.cdata, w.buf.cell_data)
        push!(w.times, Float64(newStep))
        push!(w.frame_npts, npts)
        push!(w.frame_ncells, length(UniqueCells))
        w.pending += 1
        w.pending == w.capacity && flush_frames!(w)
        return nothing
    end

    """
        flush_frames!(writer)

    Write the pending grid frames to the file.
    """
    function flush_frames!(w::GridFrameWriter)
        k = w.pending
        k == 0 && return w
        f0     = w.nframes
        frames = f0:(f0 + k - 1)
        # offsets of every pending frame into the point and cell datasets
        point_offsets = w.npts_total   .+ [0; cumsum(w.frame_npts)[1:end - 1]]
        cell_offsets  = w.ncells_total .+ [0; cumsum(w.frame_ncells)[1:end - 1]]
        append_columns!(w.points, reshape(w.pts, 3, :))
        append_slice!(w.connectivity, w.conn)
        append_slice!(w.offsets, w.offs)
        append_slice!(w.types, w.typ)
        append_slice!(w.cell_data, w.cdata)
        append_slice!(w.number_of_points, w.frame_npts)
        append_slice!(w.number_of_cells, w.frame_ncells)
        append_slice!(w.number_of_connectivity_ids, w.frame_npts)
        append_slice!(w.values, w.times)
        append_slice!(w.point_offsets, point_offsets)
        append_slice!(w.number_of_parts, fill(idType(1), k))
        append_slice!(w.part_offsets, collect(frames))
        append_slice!(w.connectivity_id_offsets, point_offsets)
        append_slice!(w.cell_offsets, cell_offsets)
        set_nsteps!(w.nsteps, f0 + k)
        w.nframes      += k
        w.npts_total   += sum(w.frame_npts)
        w.ncells_total += sum(w.frame_ncells)
        w.pending = 0
        foreach(empty!, (w.times, w.frame_npts, w.frame_ncells, w.pts, w.conn, w.offs, w.typ, w.cdata))
        return w
    end

    frames_written(w::GridFrameWriter) = w.nframes
    frames_pending(w::GridFrameWriter) = w.pending

    function AppendVTKHDFGridData(root, newStep, cell_edge::Real, UniqueCells, SimParticles)
        points, connectivity, offsets, cell_types, cell_data, _ = compute_grid_geometry(cell_edge, UniqueCells)
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

        CellDataStartIndex = length(root["CellData"]["CellData"]) + 1
        HDF5.set_extent_dims(root["CellData"]["CellData"], (length(root["CellData"]["CellData"]) + length(UniqueCells),))
        root["CellData"]["CellData"][CellDataStartIndex:end] = cell_data

        return nothing
    end

    function SaveCellGridVTKHDF(FilePath, cell_edge::Real, UniqueCells)
        points, connectivity, offsets, cell_types, cell_data, _ = compute_grid_geometry(cell_edge, UniqueCells)

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
            cell_group["CellData"] = cell_data
            close(cell_group)
        end

        # Write an empty FieldData group (placeholder for additional data)
        create_group(gtop, "FieldData")
        
        # Close file
        close(io)
    end

    """
        SetupVTKOutput(SimMetaData, SimParticles, SimKernel, Dimensions)

    Prepare VTK/HDF5 output. Returns a named tuple with `save_particles`,
    `save_grid` and `close_files` functions. Uses single or multi-file mode
    depending on `SimMetaData.ExportSingleVTKHDF`.

    In single file mode the frames pass through the buffered writers
    (`PolyDataFrameWriter`, `GridFrameWriter`): `frames_per_flush` frames are
    held in host memory and written together, the number following from
    `SimMetaData.GPUOutputBufferBytes` and the size of a frame (see
    `buffered_frames`). `close_files` writes the frames still pending.
    """
    function SetupVTKOutput(SimMetaData, SimParticles, SimKernel, Dimensions)
        # Generate save locations
        particle_savepath = joinpath(SimMetaData.SaveLocation, SimMetaData.SimulationName)
        grid_savepath = joinpath(SimMetaData.SaveLocation, "CellGrid_$(SimMetaData.SimulationName)")

        # File naming functions
        particle_filename = (iter) -> "$(particle_savepath)_$(lpad(iter,6,"0")).vtkhdf"
        grid_filename = (iter) -> "$(grid_savepath)_$(lpad(iter,6,"0")).vtkhdf"

        output_vars = SimMetaData.OutputVariables
        multi_file  = !SimMetaData.ExportSingleVTKHDF
        n = length(SimParticles.Position)

        # The particle field behind an output variable, and an array with the
        # element type and component count of the variable as it is written
        # (`Type` is written as `Int8`, vector fields as `3 × N` components in
        # the precision of the field) for creating the datasets.
        field_source(name) = getproperty(SimParticles, Symbol(name))
        function field_descriptor(name)
            name == "Type" && return Int8[]
            src = field_source(name)
            return ncomponents(src) == 3 ? Matrix{component_type(src)}(undef, 3, 0) : src
        end

        # Multi file mode: host arrays written for the output variables.
        # Vector fields are written from a preallocated `3 × N` component
        # matrix in the precision of the field (the positions and ghost nodes
        # may be `Float64` with `GPUDoublePosition` while the other vector
        # fields are `Float32`): in 2D the third row is zero, in 3D the matrix
        # replaces the copy that `stack` made per frame. `Type` is written as
        # `Int8` from a buffer as well. The single file mode fills the frame
        # buffers of the `PolyDataFrameWriter` directly and needs none of these.
        component_buffer(src) = Matrix{component_type(src)}(undef, 3, n)
        pos_buf  = multi_file ? component_buffer(SimParticles.Position) : nothing
        vec_bufs = Dict{String, Matrix}(name => component_buffer(field_source(name))
                                        for name in output_vars if multi_file && ncomponents(field_source(name)) == 3)
        type_buf = multi_file && "Type" in output_vars ? Vector{Int8}(undef, n) : nothing
        function output_field(name)
            name == "Type" && return map!(Int8, type_buf, SimParticles.Type)
            src = field_source(name)
            buf = get(vec_bufs, name, nothing)
            buf === nothing && return src
            return components!(buf, src)
        end
        output_position() = components!(pos_buf, SimParticles.Position)

        # Initialize storage for file handles and the frame writers of the single file mode
        frame_writer = nothing
        grid_writer  = nothing
        frames_per_flush = 1
        file_handles = if multi_file
            # Multi-file mode: vector for particle files
            # Frames written: the initial state plus one per output deadline.
            # Deadlines are clamped to `SimulationTime` (see `next_output_time`),
            # so a final frame lands on the end time even when it is not a
            # multiple of the interval or not listed in the schedule. One spare
            # slot guards against floating point rounding of the ratio;
            # `close_files` skips unassigned slots.
            n_outputs = if SimMetaData.OutputTimes isa AbstractVector
                count(t -> t < SimMetaData.SimulationTime, SimMetaData.OutputTimes) + 2
            else
                ceil(Int, SimMetaData.SimulationTime / SimMetaData.OutputTimes) + 2
            end
            (
                particle_files = Vector{HDF5.File}(undef, n_outputs),
                grid_files = nothing,
            )
        else
            descriptors = [field_descriptor(name) for name in output_vars]

            # Frames held in host memory before a flush: the memory budget
            # divided by the size of a frame. A particle frame holds the
            # positions and the output variables; a grid frame is bounded by
            # every particle sitting in its own cell (corner points and
            # connectivity per corner, offset, type and id per cell).
            frame_bytes = n * (3 * sizeof(component_type(SimParticles.Position)) +
                               sum(d -> ncomponents(d) * sizeof(component_type(d)), descriptors; init = 0))
            if SimMetaData.ExportGridCells
                frame_bytes += n * (2^Dimensions * (3 * sizeof(Float64) + sizeof(Int)) + 2 * sizeof(Int) + sizeof(UInt8))
            end
            frames_per_flush = buffered_frames(frame_bytes, SimMetaData.GPUOutputBufferBytes)

            # Single-file mode: handles for both files
            OutputVTKHDF = h5open("$(particle_savepath).vtkhdf", "w")
            root = HDF5.create_group(OutputVTKHDF, "VTKHDF")

            # The point datasets are chunked by whole flushes (the frames per
            # flush are a power of two, so smaller batches would still align):
            # every flush writes complete chunks and the file holds one chunk
            # per dataset and flush instead of one per frame, which makes the
            # writes and the close (which writes the chunk indices) cheaper.
            GenerateGeometryStructure(root, output_vars, descriptors...; chunk_size = frames_per_flush * n)
            GenerateStepStructure(root, output_vars, descriptors...)
            frame_writer = PolyDataFrameWriter(root, SimParticles.Position, output_vars, descriptors...; capacity = frames_per_flush)

            # Initialize grid file if needed
            if SimMetaData.ExportGridCells
                OutputVTKHDFGrid = h5open("$(particle_savepath)_GridCells.vtkhdf", "w")
                root_grid = HDF5.create_group(OutputVTKHDFGrid, "VTKHDF")
                GenerateGeometryStructure(root_grid; vtk_file_type="UnstructuredGrid", chunk_size = 2^13)
                GenerateStepStructure(root_grid; vtk_file_type="UnstructuredGrid")
                grid_writer = GridFrameWriter(root_grid; capacity = frames_per_flush)

                (particle_files = OutputVTKHDF, grid_files = OutputVTKHDFGrid)
            else
                (particle_files = OutputVTKHDF, grid_files = nothing)
            end
        end

        # Main saving functions. `time` is passed explicitly so that a writer
        # running asynchronously stamps the step with the time of the download.
        function save_particle_data(iteration, time = SimMetaData.TotalTime)
            if multi_file
                output_data = [output_field(name) for name in output_vars]
                SaveVTKHDF(file_handles.particle_files, iteration, particle_filename(iteration),
                           output_position(), output_vars, output_data...)
            else
                append_frame!(frame_writer, time, SimParticles.Position, (field_source(name) for name in output_vars)...)
            end
        end

        function save_cell_grid(iteration, cells, SimParticles, time = SimMetaData.TotalTime)
            if SimMetaData.ExportGridCells
                isempty(cells) && return nothing
                cell_edge = grid_cell_edge(SimKernel, SimMetaData)
                if multi_file
                    SaveCellGridVTKHDF(grid_filename(iteration), cell_edge, cells)
                else
                    append_grid_frame!(grid_writer, time, cell_edge, cells)
                end
            end
        end

        function close_files()
            if multi_file
                # Close all particle files in multi-file mode
                for i in eachindex(file_handles.particle_files)
                    isassigned(file_handles.particle_files, i) || continue
                    f = file_handles.particle_files[i]
                    isopen(f) && close(f)
                end
            else
                # Write the pending frames and close the single-file handles
                if isopen(file_handles.particle_files)
                    flush_frames!(frame_writer)
                    close(file_handles.particle_files)
                end
                if file_handles.grid_files !== nothing && isopen(file_handles.grid_files)
                    flush_frames!(grid_writer)
                    close(file_handles.grid_files)
                end
            end
        end

        # Return interface functions and handles
        return (
            save_particles = save_particle_data,
            save_grid = save_cell_grid,
            close_files = close_files,
            file_handles = file_handles,  # For advanced access if needed
            variable_names = output_vars,
            frames_per_flush = frames_per_flush,
        )
    end
end
