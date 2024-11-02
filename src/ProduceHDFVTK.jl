module ProduceHDFVTK     

    export SaveVTKHDF, SaveCellGrid, SaveCellGridVTKHDF

    using StaticArrays
    using WriteVTK
    using HDF5

    function SaveVTKHDF(fid_vector, index, filepath,points, variable_names, args...)
        @assert length(variable_names) == length(args) "Same number of variable_names as args is necessary"
            io = h5open(filepath, "w")
            # Create toplevel group /VTKHDF
            gtop = HDF5.create_group(io, "VTKHDF")

            # Write version of VTKHDF format as an attribute
            HDF5.attrs(gtop)["Version"] = [2, 1]

            # Write type of dataset ("PolyData") as an ASCII string to a "Type" attribute.
            # This is a bit tricky because VTK/ParaView don't support UTF-8 strings here, which is
            # the default in HDF5.jl.
            let s = "PolyData"
                dtype = HDF5.datatype(s)
                HDF5.API.h5t_set_cset(dtype.id, HDF5.API.H5T_CSET_ASCII)
                dspace = HDF5.dataspace(s)
                attr = HDF5.create_attribute(gtop, "Type", dtype, dspace)
                HDF5.write_attribute(attr, dtype, s)
            end

            # Write points + number of points.
            # Note that we need to reinterpret the vector of SVector onto a 3×Np matrix.
            Np = length(points)
            gtop["NumberOfPoints"] = [Np]
            gtop["Points"] = reinterpret(reshape, eltype(eltype(points)), points)

            # Write velocities as point data.
            let g = HDF5.create_group(gtop, "PointData")
                for i ∈ eachindex(variable_names)
                    var_name = variable_names[i]
                    arg      = args[i]
                    g[var_name] = reinterpret(reshape, eltype(eltype(arg)), arg)
                end
            end

            # Create and fill Vertices group.
            let g = HDF5.create_group(gtop, "Vertices")
                # In our case 1 point == 1 cell.
                g["NumberOfCells"] = [Np]
                g["NumberOfConnectivityIds"] = [Np]
                g["Connectivity"] = collect(0:(Np - 1))
                g["Offsets"] = collect(0:Np)
                close(g)
            end

            # Add unused PolyData types. ParaView expects this, even if they're empty.
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

    function SaveCellGrid(FilePath, SimConstants, UniqueCells)
        # Cell dimensions
        dx = SimConstants.H
        dy = SimConstants.H

        # Initialise lists for storing points and cells
        points   = Vector{SVector{3, Float64}}()    # List to store unique SVector points
        cells    = empty!([MeshCell(VTKCellTypes.VTK_QUAD, 1:4)])  # empty vector of cells (with the right element type)
        cells_id = Int[]  # not needed, but useful for visualisation purposes

        # Loop through each CartesianIndex cell
        for (id, cell) in enumerate(UniqueCells)
            # Get x and y from the CartesianIndex and calculate cell center
            xi, yi = cell.I
            x_center = (xi - 0.5) * dx
            y_center = (yi - 0.5) * dy

            # Define corners in counterclockwise order
            x0, y0 = x_center - dx / 2, y_center - dy / 2
            x1, y1 = x_center + dx / 2, y_center - dy / 2
            x2, y2 = x_center + dx / 2, y_center + dy / 2
            x3, y3 = x_center - dx / 2, y_center + dy / 2

            # Create quad cell connecting the 4 new points
            n = length(points)
            cell = MeshCell(VTKCellTypes.VTK_QUAD, (n + 1):(n + 4))
            push!(cells, cell)
            push!(cells_id, id)

            # Add corners as individual points
            push!(points, SVector(x0, y0, 0.0))
            push!(points, SVector(x1, y1, 0.0))
            push!(points, SVector(x2, y2, 0.0))
            push!(points, SVector(x3, y3, 0.0))
        end

        # Write to VTK as unstructured grid
        vtk_grid(FilePath, points, cells) do vtk
            vtk["cell_id"] = cells_id
        end
    end

    function SaveCellGridVTKHDF(FilePathVector, Index, FilePath, SimConstants, UniqueCells)
        # Cell dimensions
        dx = SimConstants.H
        dy = SimConstants.H
    
        # Initialize lists for storing points and cells
        points = Vector{SVector{3, Float64}}()  # List to store unique SVector points
        connectivity = Int[]                    # Connectivity for each cell
        offsets      = Int[]                    # Offsets for each cell
        cell_types   = Int[]                    # Cell types (for VTK_QUAD)
        cell_data    = Int[]
    
        push!(offsets, 0)
        # Loop through each CartesianIndex cell
        for (id, cell) in enumerate(UniqueCells)
            # Get x and y from the CartesianIndex and calculate cell center
            xi, yi = cell.I
            x_center = (xi - 0.5) * dx
            y_center = (yi - 0.5) * dy
    
            # Define corners individually
            x0, y0 = x_center - dx / 2, y_center - dy / 2
            x1, y1 = x_center + dx / 2, y_center - dy / 2
            x2, y2 = x_center + dx / 2, y_center + dy / 2
            x3, y3 = x_center - dx / 2, y_center + dy / 2
    
            # Add each corner point individually and update connectivity
            n = length(points)
            push!(points, SVector(x0, y0, 0.0))
            push!(connectivity, n)
            n += 1
    
            push!(points, SVector(x1, y1, 0.0))
            push!(connectivity, n)
            n += 1
    
            push!(points, SVector(x2, y2, 0.0))
            push!(connectivity, n)
            n += 1
    
            push!(points, SVector(x3, y3, 0.0))
            push!(connectivity, n)
    
            # Define cell type and offsets
            push!(offsets, length(connectivity))
            push!(cell_types, VTKCellTypes.VTK_QUAD.vtk_id)  # Convert to Int

            push!(cell_data, id)
        end
    
        # Open HDF5 file for writing
        io = h5open(FilePath, "w")

        # Create top-level group "VTKHDF"
        gtop = HDF5.create_group(io, "VTKHDF")

        # Set the Version attribute
        HDF5.attrs(gtop)["Version"] = [2, 1]

        # Write Type attribute as ASCII string
        let s = "UnstructuredGrid"
            dtype = HDF5.datatype(s)
            HDF5.API.h5t_set_cset(dtype.id, HDF5.API.H5T_CSET_ASCII)
            dspace = HDF5.dataspace(s)
            attr = HDF5.create_attribute(gtop, "Type", dtype, dspace)
            HDF5.write_attribute(attr, dtype, s)
        end

        # Write Number of Points, Number of Cells, and Number of Connectivity IDs
        gtop["NumberOfPoints"]          = [length(points)]
        gtop["NumberOfCells"]           = [length(cell_types)]
        gtop["NumberOfConnectivityIds"] = [length(connectivity)]

        # Write Points
        gtop["Points"] = reinterpret(reshape, eltype(eltype(points)), points)

        # Write Connectivity, Offsets, and Types
        gtop["Connectivity"] = connectivity
        gtop["Offsets"] = offsets
        gtop["Types"] = [VTKCellTypes.VTK_QUAD.vtk_id for _ in cell_types]  # Use VTKCellTypes Quad vtk_id for all types

        # Write CellData (cell-level variables)
        let cell_group = HDF5.create_group(gtop, "CellData")
            cell_group["CellData"] = cell_data
            close(cell_group)
        end

        # Write PointData (point-level variables)
        # let point_group = HDF5.create_group(gtop, "PointData")
        #     for (name, data) in point_data
        #         point_group[name] = data
        #     end
        #     close(point_group)
        # end

        # Write an empty FieldData group (placeholder for additional data)
        create_group(gtop, "FieldData")
        
        # Close file
        FilePathVector[Index] = io
    end
    
    

end