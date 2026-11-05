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

point_array_status = list(__OUTPUT_VARIABLES__)
# Body data is saved independently of the requested particle arrays.
has_floating_forces = 'Force' in Simulation_vtkhdf.PointArrayStatus.Available
has_native_forces = 'ConstraintForce' in Simulation_vtkhdf.PointArrayStatus.Available
if has_floating_forces:
    point_array_status = list(dict.fromkeys(point_array_status +
                             ['Body', 'GroupMarker', 'Velocity', 'Force', 'Torque', 'ForceMagnitude']))
if has_native_forces:
    point_array_status = list(dict.fromkeys(point_array_status +
        ['ConstraintForce', 'ConstraintTorque', 'GravityForce', 'TotalForce',
         'ConstraintForceMagnitude', 'GravityForceMagnitude', 'TotalForceMagnitude']))
if __SHOW_DAMPING__:
    point_array_status.append('NormalizedDamping')
Simulation_vtkhdf.PointArrayStatus = point_array_status

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from Simulation_vtkhdf
Simulation_vtkhdfDisplay = Show(Simulation_vtkhdf, renderView1, 'GeometryRepresentation')

Simulation_vtkhdfDisplay.SetRepresentationType('__REPRESENTATION__')

# Show only particles in composite measurement files. Measurement blocks stay
# available in the composite block selectors and can be enabled as needed.
Simulation_vtkhdf.UpdatePipeline()
data_type = Simulation_vtkhdf.GetDataInformation().GetDataSetTypeAsString()
if data_type == 'vtkMultiBlockDataSet':
    Simulation_vtkhdfDisplay.BlockSelectors = ['/Root/Particles']

if has_floating_forces and not has_native_forces and data_type == 'vtkMultiBlockDataSet':
    FloatingBodies = ExtractBlock(registrationName='Floating body loads', Input=Simulation_vtkhdf)
    FloatingBodies.Selectors = ['/Root/FloatingBodies']
    # Suppress zero loads without making arrow length depend on force units.
    FloatingForceMask = Calculator(registrationName='Nonzero floating loads', Input=FloatingBodies)
    FloatingForceMask.ResultArrayName = 'NonzeroForce'
    FloatingForceMask.Function = 'if(ForceMagnitude > 0, 1, 0)'
    FloatingForceGlyphs = Glyph(registrationName='Floating force arrows', Input=FloatingForceMask,
                               GlyphType='Arrow')
    FloatingForceGlyphs.OrientationArray = ['POINTS', 'Force']
    FloatingForceGlyphs.ScaleArray = ['POINTS', 'NonzeroForce']
    # Particle spacing can be tiny compared with the scene in refined runs.
    floating_bounds = Simulation_vtkhdf.GetDataInformation().GetBounds()
    floating_scene_size = max(floating_bounds[2*d+1] - floating_bounds[2*d] for d in range(3))
    FloatingForceGlyphs.ScaleFactor = max(12.0 * float(__GAUSSIAN_RADIUS__), 0.08 * floating_scene_size)
    FloatingForceGlyphs.GlyphMode = 'All Points'
    FloatingForceDisplay = Show(FloatingForceGlyphs, renderView1, 'GeometryRepresentation')
    FloatingForceDisplay.SetRepresentationType('Surface')
    ColorBy(FloatingForceDisplay, ('POINTS', 'ForceMagnitude'), separate=True)
    FloatingForceDisplay.RescaleTransferFunctionToDataRange(True, False)
    floating_force_lut = GetColorTransferFunction('ForceMagnitude', FloatingForceDisplay, separate=True)
    floating_force_lut.AutomaticRescaleRangeMode = 'Grow and update every timestep'
    FloatingForceDisplay.SetScalarBarVisibility(renderView1, True)
    floating_force_bar = GetScalarBar(floating_force_lut, renderView1)
    floating_force_bar.Title = 'Floating body force [N]'
    floating_force_bar.ComponentTitle = ''
    floating_force_bar.WindowLocation = 'Lower Left Corner'

if has_native_forces and data_type == 'vtkMultiBlockDataSet':
    FloatingBodies = ExtractBlock(registrationName='Body forces [N]', Input=Simulation_vtkhdf)
    FloatingBodies.Selectors = ['/Root/FloatingBodies']
    FloatingBodies.UpdatePipeline()
    body_bounds = FloatingBodies.GetDataInformation().GetBounds()
    body_size = max(body_bounds[2*d+1] - body_bounds[2*d] for d in range(3))
    force_arrow_length = max(8.0 * float(__GAUSSIAN_RADIUS__), 0.25 * body_size)
    native_force_arrows = []
    for title, vector, magnitude, colour in (
        ('Gravity force (blue)', 'GravityForce', 'GravityForceMagnitude', [0.10, 0.35, 0.95]),
        ('Joint and contact reaction (red)', 'ConstraintForce', 'ConstraintForceMagnitude', [0.90, 0.12, 0.08]),
        ('Net force (hidden)', 'TotalForce', 'TotalForceMagnitude', [0.15, 0.65, 0.25]),
    ):
        mask = Calculator(registrationName=title + ' mask', Input=FloatingBodies)
        mask.ResultArrayName = 'VisibleForce'
        mask.Function = 'if(' + magnitude + ' > 0, 1, 0)'
        arrows = Glyph(registrationName=title, Input=mask, GlyphType='Arrow')
        arrows.OrientationArray = ['POINTS', vector]
        arrows.ScaleArray = ['POINTS', 'VisibleForce']
        arrows.ScaleFactor = force_arrow_length
        arrows.GlyphMode = 'All Points'
        display = Show(arrows, renderView1, 'GeometryRepresentation')
        display.SetRepresentationType('Surface')
        display.ColorArrayName = ['POINTS', '']
        display.DiffuseColor = colour
        native_force_arrows.append((vector, arrows, colour))
        if vector == 'TotalForce':
            Hide(arrows, renderView1)
    ForceKey = Text(registrationName='Force arrow key')
    ForceKey.Text = ('Blue: gravity; red: joints + contact.\n'
                     'Arrows show direction (equal length); values in Body forces [N].\n'
                     'Net force = gravity + reaction + SPH/spring force (green, hidden).')
    ForceKeyDisplay = Show(ForceKey, renderView1, 'TextSourceRepresentation')
    ForceKeyDisplay.WindowLocation = 'Upper Left Corner'
    ForceKeyDisplay.FontSize = 13
    ForceKeyDisplay.Color = [0.12, 0.12, 0.12]

if data_type == 'vtkMultiBlockDataSet' and 'ForceMagnitude' in Simulation_vtkhdf.CellArrayStatus.Available:
    Simulation_vtkhdf.CellArrayStatus = list(Simulation_vtkhdf.CellArrayStatus.Available)
    SpringBlock = ExtractBlock(registrationName='Force-coloured springs', Input=Simulation_vtkhdf)
    SpringBlock.Selectors = ['/Root/Springs']
    SpringBlock.UpdatePipeline()
    SpringDisplay = Show(SpringBlock, renderView1, 'GeometryRepresentation')
    SpringDisplay.SetRepresentationType('Surface')
    ColorBy(SpringDisplay, ('CELLS', 'ForceMagnitude'))
    spring_lut = GetColorTransferFunction('ForceMagnitude')
    spring_lut.RGBPoints = [0.0, 0.267, 0.005, 0.329,
                           0.25, 0.230, 0.322, 0.546,
                           0.5, 0.128, 0.567, 0.551,
                           0.75, 0.369, 0.789, 0.383,
                           1.0, 0.993, 0.906, 0.144]
    spring_force_range = __SPRING_FORCE_RANGE__
    if spring_force_range is None:
        SpringDisplay.RescaleTransferFunctionToDataRangeOverTime()
    else:
        spring_lut.RescaleTransferFunction(*spring_force_range)
    spring_lut.AutomaticRescaleRangeMode = 'Never'
    SpringDisplay.SetScalarBarVisibility(renderView1, True)
    spring_bar = GetScalarBar(spring_lut, renderView1)
    spring_bar.Title = 'Spring + damper force [N]'
    spring_bar.ComponentTitle = ''
    spring_bar.WindowLocation = 'Upper Left Corner'
    GeometryBlock = ExtractBlock(registrationName='Body and refinement outlines', Input=Simulation_vtkhdf)
    GeometryBlock.Selectors = ['/Root/Geometry']
    GeometryDisplay = Show(GeometryBlock, renderView1, 'GeometryRepresentation')
    GeometryDisplay.SetRepresentationType('Wireframe')
    GeometryDisplay.ColorArrayName = ['CELLS', '']
    GeometryDisplay.DiffuseColor = [0.3, 0.3, 0.35]

if __SHOW_DAMPING__ and data_type == 'vtkMultiBlockDataSet':
    DampingBlocks = ExtractBlock(registrationName='Damping zones and profile',
                                 Input=Simulation_vtkhdf)
    DampingBlocks.Selectors = ['/Root/DampingZones', '/Root/DampingProfiles']
    DampingBlocks.UpdatePipeline()
    DampingBlocksDisplay = Show(DampingBlocks, renderView1, 'GeometryRepresentation')
    DampingBlocksDisplay.SetRepresentationType('Wireframe')
    ColorBy(DampingBlocksDisplay, ('POINTS', 'NormalizedDamping'))
    DampingBlocksDisplay.LineWidth = 2.0
    damping_lut = GetColorTransferFunction('NormalizedDamping')
    damping_lut.RescaleTransferFunction(0.0, 1.0)
    damping_lut.AutomaticRescaleRangeMode = 'Never'
    DampingBlocksDisplay.SetScalarBarVisibility(renderView1, True)
    damping_bar = GetScalarBar(damping_lut, renderView1)
    damping_bar.Title = 'Normalized damping weight (dimensionless)'
    damping_bar.ComponentTitle = ''

# To always load in at correct position
# Simulation_vtkhdfDisplay.Position = [0.0, 0.0, 0.0]

# set scalar coloring
ColorBy(Simulation_vtkhdfDisplay, ('POINTS', '__COLOR_VAR__'))

# rescale color and/or opacity maps used to include current data range
Simulation_vtkhdfDisplay.RescaleTransferFunctionToDataRange(True, False)

# Keep the pressure scale fixed so colors remain comparable across timesteps.
pressure_range = __PRESSURE_RANGE__
if 'Pressure' in __OUTPUT_VARIABLES__:
    Simulation_vtkhdf.UpdatePipeline()
    if pressure_range is None:
        pressure_range = list(Simulation_vtkhdf.PointData['Pressure'].GetRange())
        if pressure_range[0] == pressure_range[1]:
            pressure_range = [min(0.0, pressure_range[0]),
                              max(1.0, pressure_range[1])]
    pressure_lut = GetColorTransferFunction('Pressure')
    pressure_lut.RescaleTransferFunction(*pressure_range)
    pressure_lut.AutomaticRescaleRangeMode = 'Never'
    GetOpacityTransferFunction('Pressure').RescaleTransferFunction(*pressure_range)
    pressure_bar = GetScalarBar(pressure_lut, renderView1)
    pressure_bar.Title = 'Pressure [Pa]'
    pressure_bar.ComponentTitle = ''
    pressure_bar.Visibility = 0

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
