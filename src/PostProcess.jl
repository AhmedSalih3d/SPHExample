module PostProcess

export create_vtp_file, OutputVTP

    using WriteVTK
    using Parameters
    using Printf
    include("SimulationEquations.jl");  using .SimulationEquations

    # This function uses WriteVTK to produce a simple ParaView file for visualization
    # Make sure to use "Point Gaussian" and select something other than "Solid Color" to see the particles!
    function create_vtp_file(SimulationMetaData, InputData, Positions; kwargs...)
        @unpack SaveLocation, SimulationName, Iteration = SimulationMetaData
        @unpack ρ₀, c₀, γ = InputData
    
        # Initialize containers for VTK data structure
        polys = empty(MeshCell{WriteVTK.PolyData.Polys, UnitRange{Int64}}[])
        verts = empty(MeshCell{WriteVTK.PolyData.Verts, UnitRange{Int64}}[])
        all_cells = (verts, polys)
    
        filename = SaveLocation * "/" * SimulationName * "_" * lpad(Iteration, 4, "0")
    
        # Create a .vtp file with the specified positions
        vtk_grid(filename, Positions, all_cells..., compress = false) do vtk
            # Add additional arrays passed via keyword arguments
            for (name, array) in kwargs
                vtk_point_data(vtk, array, string(name))
            end
        end
    end
    
    function OutputVTP(SimulationMetaData, SimulationConstants, Positions; kwargs...)
        if SimulationMetaData.Iteration % SimulationMetaData.OutputIteration == 0
            create_vtp_file(SimulationMetaData, SimulationConstants, Positions; kwargs...)
        end
    end

end