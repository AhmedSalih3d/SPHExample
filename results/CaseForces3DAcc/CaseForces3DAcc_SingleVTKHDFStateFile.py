# import regex library
import re

# state file generated using paraview version 6.0.0
import paraview
paraview.compatibility.major = 6
paraview.compatibility.minor = 0

# Directory containing the .vtkhdf files
directory = "./results/CaseForces3DAcc"

# List all .vtkhdf files in the directory
import os
regex = r"CaseForces3DAcc.vtkhdf" # Regular expression to match the .vtkhdf files
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
renderView1.InteractionMode = "3D"

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

Simulation_vtkhdf = VTKHDFReader(registrationName='CaseForces3DAcc.vtkhdf*', FileName=file_list)

Simulation_vtkhdf.PointArrayStatus = ['Acceleration', 'Velocity', 'Density', 'Pressure', 'Type', 'GroupMarker', 'ID']

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from Simulation_vtkhdf
Simulation_vtkhdfDisplay = Show(Simulation_vtkhdf, renderView1, 'GeometryRepresentation')

Simulation_vtkhdfDisplay.SetRepresentationType('Point Gaussian')

# To always load in at correct position
# Simulation_vtkhdfDisplay.Position = [0.0, 0.0, 0.0]

# set scalar coloring
ColorBy(Simulation_vtkhdfDisplay, ('POINTS', 'Density'))

# rescale color and/or opacity maps used to include current data range
Simulation_vtkhdfDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
Simulation_vtkhdfDisplay.SetScalarBarVisibility(renderView1, True)

# Focus the camera on the dataset
renderView1.ResetCamera()

Render()
