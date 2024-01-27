module PostProcess

export create_vtp_file, OutputVTP

    using WriteVTK
    using Parameters
    using Printf
    include("SimulationEquations.jl");  using .SimulationEquations

    # This function uses WriteVTK to produce a simple ParaView file for visualization
    # Make sure to use "Point Gaussian" and select something other than "Solid Color" to see the particles!
    function create_vtp_file(SimulationMetaData,InputData,SimulationData)
        @unpack SaveLocation,SimulationName,Iteration = SimulationMetaData
        @unpack ρ₀,c₀,γ = InputData

        # Convert the particle positions and densities into the format required by the vtk_grid function:
        points = reduce(hcat,SimulationData.Position)  # Concatenate the particle positions into a single matrix
        polys = empty(MeshCell{WriteVTK.PolyData.Polys,UnitRange{Int64}}[])
        verts = empty(MeshCell{WriteVTK.PolyData.Verts,UnitRange{Int64}}[])

        # Note: the order of verts, lines, polys and strips is not important.
        # One doesn't even need to pass all of them.
        all_cells = (verts, polys)

        filename  = SaveLocation*"/"*SimulationName*"_"*lpad(Iteration,4,"0")
        # Create a .vtp file with the particle positions and densities:
        vtk_grid(filename, points, all_cells..., compress = true, append = false) do vtk

            # Add the particle densities as a point data array:
            vtk_point_data(vtk, SimulationData.Kernel, "Wi")
            vtk_point_data(vtk, SimulationData.KernelGradient, "Wg")
            vtk_point_data(vtk, SimulationData.Density, "Density")
            vtk_point_data(vtk, Pressure.(SimulationData.Density,c₀,γ,ρ₀), "Pressure")
            vtk_point_data(vtk, SimulationData.Acceleration, "Acceleration")
            vtk_point_data(vtk, SimulationData.Velocity, "Velocity")
        end
    end

    function OutputVTP(SimulationMetaData,SimulationConstants,FinalResults,dt)
        @printf "Iteration %i | dt = %.5e \n" SimulationMetaData.Iteration dt
        if SimulationMetaData.Iteration % SimulationMetaData.OutputIteration == 0
            create_vtp_file(SimulationMetaData,SimulationConstants,FinalResults)
        end
    end

end