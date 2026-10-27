import argparse
import csv
from pathlib import Path
import numpy as np
from paraview.simple import (OpenDataFile, CreateView, Show, Hide, ColorBy, GetColorTransferFunction,
    CreateLayout, AssignViewToLayout, Text, SaveScreenshot, Delete, TrivialProducer)
from vtkmodules.vtkIOLegacy import vtkPolyDataReader
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

root = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description='Render Julia and released DualSPHysics scenes at matching saved frames.')
parser.add_argument('--reference-root', type=Path, required=True, help='Directory containing MotionXX/Case__Actual.vtk and transforms_mk*.csv')
parser.add_argument('--animations-root', type=Path, default=root/'output'/'Motion')
parser.add_argument('--output', type=Path, default=root/'output'/'Motion'/'comparison')
args = parser.parse_args()
output = args.output
output.mkdir(parents=True,exist_ok=True)
times = {1:.75,2:1.25,3:1.4,4:1.8,5:5.6,6:4.,7:3.,8:5.,9:5.}
palette = [(.72,.75,.79),(.82,.2,.18),(.17,.55,.31),(.2,.4,.83),(.92,.57,.13),(.62,.28,.73),(.15,.65,.7)]
for number,time in times.items():
    name = f'Motion{number:02}'
    path = args.animations_root/name/(name+'.vtkhdf')
    native = OpenDataFile(str(path))
    saved_time = min(native.TimestepValues,key=lambda t:abs(t-time))
    native.UpdatePipeline(saved_time)
    reader = vtkPolyDataReader()
    reader.SetFileName(str(args.reference_root/name/'Case__Actual.vtk'))
    reader.Update()
    data = reader.GetOutput()
    vertices = vtk_to_numpy(data.GetPoints().GetData()).astype(np.float64)
    mk = vtk_to_numpy(data.GetCellData().GetArray('Mk'))
    owners = np.zeros(len(vertices),dtype=int)
    for cell in range(data.GetNumberOfCells()):
        for i in range(data.GetCell(cell).GetNumberOfPoints()):
            owners[data.GetCell(cell).GetPointId(i)] = mk[cell]
    for marker in sorted(set(mk)):
        file = args.reference_root/name/f'transforms_mk{marker:04}.csv'
        if not file.exists():
            continue
        with file.open() as inp:
            row = min(csv.DictReader(inp,delimiter=';'),key=lambda r:abs(float(r['TimeStep [s]'])-time))
        matrix = np.array([[float(row[f'm{i}{j}']) for j in range(1,5)] for i in range(1,4)])
        ids = owners==marker
        vertices[ids] = vertices[ids] @ matrix[:,:3].T + matrix[:,3]
    vtkpoints = vtkPoints()
    vtkpoints.SetData(numpy_to_vtk(vertices,deep=True))
    data.SetPoints(vtkpoints)
    source = TrivialProducer()
    source.GetClientSideObject().SetOutput(data)
    view = CreateView('RenderView')
    view.ViewSize = [600,420]
    view.UseColorPaletteForBackground = 0
    view.Background = [1,1,1]
    view.OrientationAxesVisibility = 0
    display = Show(native,view)
    display.Representation = 'Surface'
    ColorBy(display,('CELLS','Mk'))
    lut = GetColorTransferFunction('Mk')
    markers = sorted(set(int(x) for x in mk))
    lut.InterpretValuesAsCategories = 1
    lut.Annotations = [s for m in markers for s in (str(m),str(m))]
    lut.IndexedColors = [x for i in range(len(markers)) for x in palette[i%len(palette)]]
    display.SetScalarBarVisibility(view,False)
    view.ViewTime = saved_time
    view.CameraParallelProjection = 1
    if number <= 5:
        view.CameraPosition = [0,0,20]
        view.CameraFocalPoint = [0,0,0]
        view.CameraViewUp = [0,1,0]
    else:
        view.CameraPosition = [6,-10,5] if number!=8 else [10,-9,8]
        view.CameraFocalPoint = [0,0,1.5] if number!=8 else [3,1.7,1]
        view.CameraViewUp = [0,0,1]
    view.ResetCamera()
    caption = Text()
    label = Show(caption,view)
    label.Color = [0.1,0.1,0.1]
    label.FontSize = 15
    caption.Text = f'{name} | Julia | t = {time:g} s'
    SaveScreenshot(str(output/f'{name}_Julia.png'),view,ImageResolution=[600,420])
    Hide(native,view)
    reference = Show(source,view)
    reference.Representation = 'Surface'
    ColorBy(reference,('CELLS','Mk'))
    reference.SetScalarBarVisibility(view,False)
    caption.Text = f'{name} | DualSPHysics | t = {time:g} s'
    SaveScreenshot(str(output/f'{name}_Reference.png'),view,ImageResolution=[600,420])
    Delete(caption); Delete(native); Delete(source); Delete(view)
    print(name,'rendered',flush=True)

# A technical comparison sheet of the rendered data, not an artistic image.
from PIL import Image
sheet = Image.new('RGB',(1800,630),'white')
for index in range(9):
    number = index+1
    row,col = divmod(index,3)
    for side,kind in enumerate(('Julia','Reference')):
        image = Image.open(output/f'Motion{number:02}_{kind}.png').convert('RGB').resize((300,210))
        sheet.paste(image,(col*600+side*300,row*210))
sheet.save(output/'Motion01-09_Comparison.png')
