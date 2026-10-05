# import regex library
import re

# state file generated using paraview version 6.1.0
import paraview
paraview.compatibility.major = 6
paraview.compatibility.minor = 1

# Directory containing the .vtkhdf files
directory = "__SAVE_LOCATION__"

# List all .vtkhdf files in the directory
import os
single_file = __SINGLE_FILE__
export_grid = __EXPORT_GRID__
simulation_name = '__SIM_NAME__'
if single_file:
    regex = re.escape(simulation_name + '.vtkhdf')
    grid_regex = re.escape(simulation_name + '_GridCells.vtkhdf')
else:
    regex = re.escape(simulation_name) + r'_\d+\.vtkhdf'
    grid_regex = re.escape('CellGrid_' + simulation_name) + r'_\d+\.vtkhdf'
file_list = sorted(os.path.join(directory, f) for f in os.listdir(directory)
                   if re.fullmatch(regex, f))
grid_files = sorted(os.path.join(directory, f) for f in os.listdir(directory)
                    if re.fullmatch(grid_regex, f)) if export_grid else []

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
renderView1.InteractionMode = "__VIEW_DIMENSION__"

SetActiveView(None)

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'VTKHDF Reader'
Simulation_vtkhdf = VTKHDFReader(registrationName='__SIM_NAME__.vtkhdf*', FileName=file_list)

Simulation_vtkhdf.PointArrayStatus = __OUTPUT_VARIABLES__

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from Simulation_vtkhdf
Simulation_vtkhdfDisplay = Show(Simulation_vtkhdf, renderView1, 'GeometryRepresentation')

Simulation_vtkhdfDisplay.SetRepresentationType('__REPRESENTATION__')

# To always load in at correct position
# Simulation_vtkhdfDisplay.Position = [0.0, 0.0, 0.0]

# set scalar coloring
ColorBy(Simulation_vtkhdfDisplay, ('POINTS', '__COLOR_VAR__'))

# rescale color and/or opacity maps used to include current data range
Simulation_vtkhdfDisplay.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
Simulation_vtkhdfDisplay.SetScalarBarVisibility(renderView1, True)

# set the Gaussian radius for the point representation
Simulation_vtkhdfDisplay.GaussianRadius = float(__GAUSSIAN_RADIUS__)

# Load the exported grid in the same session without obscuring the particles.
if grid_files:
    grid_reader = VTKHDFReader(registrationName='Cell grid', FileName=grid_files)
    grid_display = Show(grid_reader, renderView1, 'GeometryRepresentation')
    grid_display.SetRepresentationType('Wireframe')
    grid_display.ColorArrayName = ['CELLS', '']
    grid_display.DiffuseColor = [0.35, 0.35, 0.35]
elif export_grid:
    print('No exported cell grid files found for ' + simulation_name)

GetAnimationScene().UpdateAnimationUsingDataTimeSteps()
SetActiveSource(Simulation_vtkhdf)

# ----------------------------------------------------------------
# reset view to fit data bounds
renderView1.ResetCamera()
# ----------------------------------------------------------------

# Update the view to ensure updated data information
renderView1.Update()
