module PostProcess

export create_vtp_file

using WriteVTK


function create_vtp_file(filename,points,Wi,Wg,ρ,dvdt,v)
    # Convert the particle positions and densities into the format required by the vtk_grid function:
    points = reduce(hcat,points)  # Concatenate the particle positions into a single matrix
    polys = empty(MeshCell{WriteVTK.PolyData.Polys,UnitRange{Int64}}[])
    verts = empty(MeshCell{WriteVTK.PolyData.Verts,UnitRange{Int64}}[])

    # Note: the order of verts, lines, polys and strips is not important.
    # One doesn't even need to pass all of them.
    all_cells = (verts, polys)

    # Create a .vtp file with the particle positions and densities:
    vtk_grid(filename, points, all_cells..., compress = true, append = false) do vtk

        # Add the particle densities as a point data array:
        vtk_point_data(vtk, Wi, "Wi")
        vtk_point_data(vtk, Wg, "Wg")
        vtk_point_data(vtk, ρ, "Density")
        vtk_point_data(vtk, dvdt, "Acceleration")
        vtk_point_data(vtk, v, "Velocity")
    end
end

end