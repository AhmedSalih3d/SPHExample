module OpenExternalPrograms

export AutoOpenLogFile, AutoOpenParaview

using ..SimulationLoggerConfiguration
using ..SimulationMetaDataConfiguration


function AutoOpenLogFile(SimLogger::SimulationLogger, SimMetaData::SimulationMetaData)
    LogFileName = replace(strip(SimLogger.LoggerIo.name,['<', '>']), "file " => "")
    if SimMetaData.OpenLogFile
        try
            OpenLogFileCommand = `notepad "$(LogFileName)"`
            run(OpenLogFileCommand; wait = false)
        catch e
            @warn("Unable to open log file automatically. It uses notepad for Windows by default.", e)
        end
    end

    return nothing
end

function AutoOpenParaview(SimMetaData::SimulationMetaData, OutputVariableNames)
    ## Generate auto paraview py

    if SimMetaData.ExportSingleVTKHDF
        ParaViewStateFileName = SimMetaData.SaveLocation * "_SingleVTKHDFStateFile.py"
        py_regex = "$(SimMetaData.SimulationName).vtkhdf"
    else
        ParaViewStateFileName = SimMetaData.SaveLocation * "_StateFile.py"
        py_regex = "^$(SimMetaData.SimulationName)_(\\d+).vtk" #^ means to anchor the regex to the start of the string
    end

    ExtractDimensionalityMetaData(::SimulationMetaData{N, FloatType}) where {N, FloatType} = N
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

                            Simulation_vtkhdfDisplay.SetRepresentationType('Point Gaussian')

                            # To always load in at correct position
                            Simulation_vtkhdfDisplay.Position = [0.0, 0.0, 0.0]

                            # set scalar coloring
                            ColorBy(Simulation_vtkhdfDisplay, ('POINTS', 'Density'))

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

    if SimMetaData.VisualizeInParaview
        try
            OpenInParaview = `paraview --state="$(ParaViewStateFileName)"`
            run(OpenInParaview; wait = false)
        catch
            @error("You must add Paraview to path as `paraview` and use at minimum version 5.12")
        end
    end

    return nothing
end

end