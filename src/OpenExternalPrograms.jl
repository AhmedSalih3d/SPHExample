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

function AutoOpenParaview(SaveLocation_, SimMetaData::SimulationMetaData, OutputVariableNames)
    ## Generate auto paraview py
    ParaViewStateFileName = SaveLocation_ * "_StateFile.py"
    ParaViewStateFile     = open(ParaViewStateFileName, "w")

    ParaViewConfig    = 
                            """
                            # state file generated using paraview version 5.12.0
                            import paraview
                            paraview.compatibility.major = 5
                            paraview.compatibility.minor = 12
                            
                            # Directory containing the .vtkhdf files
                            directory = "$(SimMetaData.SaveLocation)"

                            # List all .vtkhdf files in the directory
                            import os
                            file_list = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.vtkhdf')]

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