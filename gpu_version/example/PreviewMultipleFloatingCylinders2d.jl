# Usage: julia --project=. example/PreviewMultipleFloatingCylinders2d.jl output_dir
# Uses ParaView's bundled Python, Pillow and matplotlib; no Julia dependencies added.
"""
    preview_multiple_floating_cylinders(output_dir)

Render the saved four-cylinder simulation as PNG snapshots, an animated GIF and
a displacement plot. Also write a ParaView preview state with colored cylinders.
"""
function preview_multiple_floating_cylinders(output_dir)
    pvpython = Sys.which("pvpython")
    pvpython === nothing && error("pvpython must be available on PATH")
    state = joinpath(output_dir, "MultipleFloatingCylinders2D_SingleVTKHDFStateFile.py")
    isfile(state) || error("Simulation state not found: $state")
    # Extend the generated Python state through ParaView's rendering API.
    styling = raw"""

if 'grid_reader' in globals():
    Hide(grid_reader, renderView1)
Hide(Simulation_vtkhdf, renderView1)
# VTKHDF stores particles without cells; thresholds need one vertex per point.
vertices = ProgrammableFilter(Input=Simulation_vtkhdf)
vertices.OutputDataSetType = 'vtkPolyData'
vertices.Script = '''from vtkmodules.vtkFiltersGeneral import vtkVertexGlyphFilter
glyph = vtkVertexGlyphFilter()
glyph.SetInputData(self.GetInputDataObject(0, 0))
glyph.Update()
self.GetOutputDataObject(0).ShallowCopy(glyph.GetOutput())
'''
base = Threshold(registrationName='Fluid and walls', Input=vertices)
base.Scalars = ['POINTS', 'Type']
base.LowerThreshold = 1
base.UpperThreshold = 3
base.ThresholdMethod = 'Between'
base.UpdatePipeline()
base_display = Show(base, renderView1, 'GeometryRepresentation')
base_display.SetRepresentationType('Point Gaussian')
base_display.GaussianRadius = 0.025
ColorBy(base_display, ('POINTS', 'Pressure'))
GetScalarBar(GetColorTransferFunction('Density'), renderView1).Visibility = 0
base_display.SetScalarBarVisibility(renderView1, True)
renderView1.ViewSize = [1120, 840]
layout1.SetSize(1120, 840)
renderView1.BackgroundColorMode = 'Single Color'
renderView1.Background = [1.0, 1.0, 1.0]
renderView1.OrientationAxesVisibility = 0
renderView1.AxesGrid.Visibility = 0
renderView1.CameraParallelProjection = 1
renderView1.CameraPosition = [0.0, 3.8, 30.0]
renderView1.CameraFocalPoint = [0.0, 3.8, 0.0]
renderView1.CameraViewUp = [0.0, 1.0, 0.0]
renderView1.CameraParallelScale = 4.6
pressure_bar.TitleColor = [0.15, 0.15, 0.15]
pressure_bar.LabelColor = [0.15, 0.15, 0.15]
pressure_bar.TitleFontSize = 16
pressure_bar.LabelFontSize = 14
pressure_bar.WindowLocation = 'Lower Right Corner'
pressure_bar.ScalarBarLength = 0.4
colors = [[0.12, 0.65, 0.35], [0.98, 0.70, 0.08],
          [0.95, 0.32, 0.16], [0.56, 0.22, 0.75]]
weights = [0.7, 1.0, 1.2, 1.5]
for b, color in enumerate(colors, 1):
    body = Threshold(registrationName='Cylinder ' + str(b), Input=vertices)
    body.Scalars = ['POINTS', 'GroupMarker']
    body.LowerThreshold = b
    body.UpperThreshold = b
    body.ThresholdMethod = 'Between'
    display = Show(body, renderView1, 'GeometryRepresentation')
    display.SetRepresentationType('Point Gaussian')
    display.GaussianRadius = 0.03
    display.ColorArrayName = ['POINTS', '']
    display.DiffuseColor = color
    display.AmbientColor = color
    display.Ambient = 1.0
    display.Diffuse = 0.0
header = Text(Text='Four independent floating cylinders\n'
                   'Relative density, left to right: 0.7 / 1.0 / 1.2 / 1.5')
header_display = Show(header, renderView1)
header_display.Color = [0.1, 0.1, 0.1]
header_display.FontSize = 16
header_display.WindowLocation = 'Upper Left Corner'
clock = Text(Text='t = 0.00 s')
clock_display = Show(clock, renderView1)
clock_display.Color = [0.1, 0.1, 0.1]
clock_display.FontSize = 18
clock_display.WindowLocation = 'Upper Right Corner'
times = list(Simulation_vtkhdf.TimestepValues)
scene = GetAnimationScene()
"""
    rendering = raw"""
from PIL import Image
frames = []
frames_dir = os.path.join(directory, 'preview_frames')
os.makedirs(frames_dir, exist_ok=True)
for i in range(0, len(times), 2):
    scene.AnimationTime = times[i]
    renderView1.ViewTime = times[i]
    clock.Text = 't = %.2f s' % times[i]
    Render(renderView1)
    path = os.path.join(frames_dir, 'frame_%03d.png' % i)
    SaveScreenshot(path, renderView1, ImageResolution=[1120, 840])
    with Image.open(path) as frame:
        frames.append(frame.convert('RGB'))
    if i == 0:
        SaveScreenshot(os.path.join(directory, 'initial.png'), renderView1,
                       ImageResolution=[1120, 840])
SaveScreenshot(os.path.join(directory, 'final.png'), renderView1,
               ImageResolution=[1120, 840])
frames[0].save(os.path.join(directory, 'motion.gif'), save_all=True,
               append_images=frames[1:], duration=130, loop=0)

import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
with open(os.path.join(directory, 'MultipleFloatingCylinders2D_Floating.csv')) as f:
    rows = list(csv.DictReader(f))
fig, ax = plt.subplots(figsize=(9, 4.5), dpi=160)
for b, color in enumerate(colors, 1):
    data = [row for row in rows if int(row['Body']) == b]
    initial_height = float(data[0]['Center:1'])
    ax.plot([float(row['Time']) for row in data],
            [float(row['Center:1']) - initial_height for row in data],
            color=color, linewidth=2.5, label='Relative density %.1f' % weights[b-1])
    print('Body %d (relative density %.1f): vertical displacement %.3f m' %
          (b, weights[b-1], float(data[-1]['Center:1']) - initial_height))
ax.axhline(0, color='0.5', linewidth=0.8)
ax.axvline(0.25, color='0.5', linestyle='--', linewidth=0.8, label='Release')
ax.set(xlabel='Simulation time [s]', ylabel='Vertical displacement [m]',
       title='Four cylinders: rise and sinking after release at 0.25 s')
ax.grid(alpha=0.2)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(directory, 'displacement.png'))
print('Saved animation, snapshots and displacement plot to ' + directory)
"""
    preview_state = read(state, String) * styling
    write(joinpath(output_dir, "PreviewState.py"), preview_state * raw"""
scene.AnimationTime = times[-1]
renderView1.ViewTime = times[-1]
clock.Text = 't = %.2f s' % times[-1]
Render(renderView1)
""")
    render_script = joinpath(output_dir, "RenderPreview.py")
    write(render_script, preview_state * rendering)
    run(`$pvpython $render_script`)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    preview_multiple_floating_cylinders(only(ARGS))
end
