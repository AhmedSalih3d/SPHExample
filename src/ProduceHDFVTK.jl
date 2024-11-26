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
            HDF5.attrs(gtop)["Version"] = [2, 3]

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

    function SaveCellGridVTKHDF(FilePathVector, Index, FilePath, SimConstants, UniqueCells::SubArray{CartesianIndex{2}, 1, Vector{CartesianIndex{2}}, Tuple{UnitRange{Int64}}, true}, SimParticles)
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
            if cell == zero(eltype(UniqueCells))
                break
            end

            xi, yi = cell.I


            # Calculate cell center
            x_center = (xi - 1.5) * dx
            y_center = (yi - 1.5) * dy
 
            corners = [
                SVector(x_center - dx / 2, y_center - dy / 2, 0.0),
                SVector(x_center + dx / 2, y_center - dy / 2, 0.0),
                SVector(x_center + dx / 2, y_center + dy / 2, 0.0),
                SVector(x_center - dx / 2, y_center + dy / 2, 0.0)
            ]
        
            # Add each corner point and update connectivity
            n = length(points)
            for corner in corners
                push!(points, corner)
                push!(connectivity, n)
                n += 1
            end
        
            # Define cell type and offsets
            push!(offsets, length(connectivity))
            push!(cell_types, VTKCellTypes.VTK_QUAD.vtk_id)

            push!(cell_data, id)
        end
    
        # Open HDF5 file for writing
        io = h5open(FilePath, "w")

        # Create top-level group "VTKHDF"
        gtop = HDF5.create_group(io, "VTKHDF")

        # Set the Version attribute
        HDF5.attrs(gtop)["Version"] = [2, 3]

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

    function SaveCellGridVTKHDF(FilePathVector, Index, FilePath, SimConstants, UniqueCells::SubArray{CartesianIndex{3}, 1, Vector{CartesianIndex{3}}, Tuple{UnitRange{Int64}}, true}, SimParticles)
        # Cell dimensions
        dx = SimConstants.H
        dy = SimConstants.H
        dz = SimConstants.H
    
        # Initialize arrays for storing points and cells
        points = Vector{SVector{3, Float64}}()  # List of unique SVector points
        connectivity = Int[]                    # Connectivity for each cell
        offsets      = Int[]                    # Offsets for each cell
        cell_types   = Int[]                    # Cell types (for VTK_HEXAHEDRON)
        cell_data    = Int[]                    # Cell-level data
    
        push!(offsets, 0)
    
        # Iterate over UniqueCells to create hexahedrons
        for (id, cell) in enumerate(UniqueCells)
            if cell == zero(eltype(UniqueCells))
                break
            end
    
            # Get cell indices
            xi, yi, zi = cell.I
    
            # Calculate cell center
            x_center = (xi - 1.5) * dx
            y_center = (yi - 1.5) * dy
            z_center = (zi - 1.5) * dz
    
            # Calculate the 8 corners of the cell relative to the center
            corners = [
                SVector(x_center - dx / 2, y_center - dy / 2, z_center - dz / 2),  # Bottom-front-left
                SVector(x_center + dx / 2, y_center - dy / 2, z_center - dz / 2),  # Bottom-front-right
                SVector(x_center + dx / 2, y_center + dy / 2, z_center - dz / 2),  # Bottom-back-right
                SVector(x_center - dx / 2, y_center + dy / 2, z_center - dz / 2),  # Bottom-back-left
                SVector(x_center - dx / 2, y_center - dy / 2, z_center + dz / 2),  # Top-front-left
                SVector(x_center + dx / 2, y_center - dy / 2, z_center + dz / 2),  # Top-front-right
                SVector(x_center + dx / 2, y_center + dy / 2, z_center + dz / 2),  # Top-back-right
                SVector(x_center - dx / 2, y_center + dy / 2, z_center + dz / 2)   # Top-back-left
            ]
    
            # Add each corner point and update connectivity
            n = length(points)
            for corner in corners
                push!(points, corner)
                push!(connectivity, n)
                n += 1
            end
    
            # Define cell type and offsets
            push!(offsets, length(connectivity))
            push!(cell_types, VTKCellTypes.VTK_HEXAHEDRON.vtk_id)
    
            push!(cell_data, id)
        end
    
        # Open HDF5 file for writing
        io = h5open(FilePath, "w")
    
        # Create top-level group "VTKHDF"
        gtop = HDF5.create_group(io, "VTKHDF")
    
        # Set the Version attribute
        HDF5.attrs(gtop)["Version"] = [2, 3]
    
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
        gtop["Types"] = [VTKCellTypes.VTK_HEXAHEDRON.vtk_id for _ in cell_types]  # Use VTK_HEXAHEDRON type for all
    
        # Write CellData (cell-level variables)
        let cell_group = HDF5.create_group(gtop, "CellData")
            cell_group["CellData"] = cell_data
            close(cell_group)
        end
    
        # Write an empty FieldData group (placeholder for additional data)
        create_group(gtop, "FieldData")
    
        # Close file
        FilePathVector[Index] = io
    end    
    
    

end