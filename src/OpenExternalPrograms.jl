"""
Utility wrappers for launching external programs such as a text editor or
ParaView. These helpers are optional conveniences used at the end of a
simulation run to quickly inspect the produced output.
"""
module OpenExternalPrograms

export AutoOpenLogFile, AutoOpenParaview

using ..SimulationLoggerConfiguration
using ..SimulationMetaDataConfiguration

"""
    _default_open_command(path)

Return a platform specific [`Cmd`] used to open `path` with the default
application.
"""
function _default_open_command(path::AbstractString)
    if Sys.iswindows()
        return `notepad $(path)`
    elseif Sys.isapple()
        return `open $(path)`
    else
        return `xdg-open $(path)`
    end
end


"""
    AutoOpenLogFile(logger, metadata; editor_cmd=nothing)

Open the simulation log file using an external editor. If `editor_cmd` is not
provided, a platform specific default is used. Setting `editor_cmd = nothing`
disables the automatic opening entirely.
"""
function AutoOpenLogFile(SimLogger::SimulationLogger,
                         SimMetaData::SimulationMetaData;
                         editor_cmd::Union{String,Nothing}=nothing)
    log_file = replace(strip(SimLogger.LoggerIo.name, ['<', '>']), "file " => "")
    if SimMetaData.OpenLogFile && !isempty(log_file)
        cmd = editor_cmd === nothing ? _default_open_command(log_file) :
              `$(editor_cmd) $(log_file)`
        try
            run(cmd; wait=false)
        catch e
            @warn("Unable to open log file automatically", e)
        end
    end

    return nothing
end

"""
    AutoOpenParaview(metadata, variable_names;
                     paraview_cmd="paraview",
                     representation="Point Gaussian",
                     color_variable="Density")

Write a ParaView state file for the given simulation and optionally
launch ParaView to visualise the results. `variable_names` should contain the
point arrays stored in the output files. Pass `paraview_cmd = nothing` to skip
launching ParaView automatically.
"""
function AutoOpenParaview(SimMetaData::SimulationMetaData, OutputVariableNames;
                          paraview_cmd::Union{String,Nothing}="paraview",
                          representation::String="Point Gaussian",
                          color_variable::String="Density")
    ## Generate auto paraview py

    if SimMetaData.ExportSingleVTKHDF
        ParaViewStateFileName = joinpath(SimMetaData.SaveLocation, SimMetaData.SimulationName) * "_SingleVTKHDFStateFile.py"
        py_regex = "$(SimMetaData.SimulationName).vtkhdf"
    else
        ParaViewStateFileName = joinpath(SimMetaData.SaveLocation, SimMetaData.SimulationName) * "_StateFile.py"
        py_regex = "^$(SimMetaData.SimulationName)_(\\d+).vtk" #^ means to anchor the regex to the start of the string
    end

    ExtractDimensionalityMetaData(::SimulationMetaData{N, FloatType, SMode}) where {N, FloatType, SMode} = N
    ViewDimension = ExtractDimensionalityMetaData(SimMetaData) == 2 ? "2D" : "3D"

    ParaViewStateFile     = open(ParaViewStateFileName, "w")

    ParaViewConfig    = 
                            """
                            # import regex library
                            import re

                            # state file generated using paraview version 5.12.0
                            import paraview
                            paraview.compatibility.major = 5
                            paraview.compatibility.minor = 12
                            
                            # Directory containing the .vtkhdf files
                            directory = "$(SimMetaData.SaveLocation)"

                            # List all .vtkhdf files in the directory
                            import os
                            regex = r"$(py_regex)" # Regular expression to match the .vtkhdf files
                            file_list = [os.path.join(directory, f) for f in os.listdir(directory) if re.search(regex,f)]

                            #### import the simple module from the paraview
                            from paraview.simple import *
                            #### disable automatic camera reset on 'Show'
                            paraview.simple._DisableFirstRenderCameraReset()
                            
                            # ----------------------------------------------------------------
                            # setup views used in the visualization
                            # ----------------------------------------------------------------

                            # get the material library
                            materialLibrary1 = GetMaterialLibrary()

                            # Create a new 'Render View'
                            renderView1 = CreateView('RenderView')
                            
                            # init the 'Grid Axes 3D Actor' selected for 'AxesGrid'
                            renderView1.AxesGrid.Visibility = 1

                            # set dimensionality of rendered view
                            renderView1.InteractionMode = "$(ViewDimension)"
                            
                            SetActiveView(None)

                            # create new layout object 'Layout #1'
                            layout1 = CreateLayout(name='Layout #1')
                            layout1.AssignView(0, renderView1)
                            #layout1.SetSize(2252, 794)
                            
                            # ----------------------------------------------------------------
                            # restore active view
                            SetActiveView(renderView1)
                            # ----------------------------------------------------------------

                            # ----------------------------------------------------------------
                            # setup the data processing pipelines
                            # ----------------------------------------------------------------

                            # create a new 'VTKHDF Reader'
                            
                            Simulation_vtkhdf = VTKHDFReader(registrationName='$(SimMetaData.SimulationName).vtkhdf*', FileName=file_list)

                            Simulation_vtkhdf.PointArrayStatus = $("['" * join(OutputVariableNames, "', '") * "']")
                            
                            # ----------------------------------------------------------------
                            # setup the visualization in view 'renderView1'
                            # ----------------------------------------------------------------

                            # show data from Simulation_vtkhdf
                            Simulation_vtkhdfDisplay = Show(Simulation_vtkhdf, renderView1, 'GeometryRepresentation')

                            Simulation_vtkhdfDisplay.SetRepresentationType('$(representation)')

                            # To always load in at correct position
                            Simulation_vtkhdfDisplay.Position = [0.0, 0.0, 0.0]

                            # set scalar coloring
                            ColorBy(Simulation_vtkhdfDisplay, ('POINTS', '$(color_variable)'))

                            # rescale color and/or opacity maps used to include current data range
                            Simulation_vtkhdfDisplay.RescaleTransferFunctionToDataRange(True, False)

                            # show color bar/color legend
                            Simulation_vtkhdfDisplay.SetScalarBarVisibility(renderView1, True)
                            
                            # Focus the camera on the dataset
                            renderView1.ResetCamera()

                            Render()
                            """

    write(ParaViewStateFile, ParaViewConfig) 

    close(ParaViewStateFile)

    if SimMetaData.VisualizeInParaview && paraview_cmd !== nothing
        try
            OpenInParaview = `$(paraview_cmd) --state="$(ParaViewStateFileName)"`
            run(OpenInParaview; wait=false)
        catch e
            @error("You must add Paraview to path as $(paraview_cmd) and use at minimum version 5.12", e)
        end
    end

    return nothing
end

end
