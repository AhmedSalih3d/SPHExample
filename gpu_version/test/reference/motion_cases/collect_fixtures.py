"""Collect the released tools' observations; run with ParaView's pvpython.

GenCase/BoundaryVTK output must already exist under --reference-root. This
utility reads their output data; it does not implement any motion equations.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree

from vtkmodules.vtkIOLegacy import vtkPolyDataReader
from vtkmodules.util.numpy_support import vtk_to_numpy

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reference-root', type=Path, required=True)
parser.add_argument('--case-source', type=Path, required=True)
parser.add_argument('--tools', type=Path, required=True)
parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)

with (args.output/'transforms.csv').open('w', newline='') as transforms_file, \
     (args.output/'geometry_bounds.csv').open('w', newline='') as bounds_file, \
     (args.output/'geometry_vertices.csv').open('w', newline='') as vertices_file:
    transforms = csv.writer(transforms_file)
    bounds = csv.writer(bounds_file)
    vertices_writer = csv.writer(vertices_file)
    transforms.writerow(['case','id','time']+[f'm{i}{j}' for i in range(1,4) for j in range(1,5)])
    bounds.writerow(['case','id','xmin','xmax','ymin','ymax','zmin','zmax','points','cells'])
    vertices_writer.writerow(['case','id','x','y','z'])
    for number in range(1,10):
        folder = args.reference_root/f'Motion{number:02}'
        case_xml = ElementTree.parse(folder/'Case.xml')
        mk_offset = int(case_xml.find('./execution/particles').get('mkboundfirst'))
        for file in sorted(folder.glob('transforms_mk*.csv')):
            marker = int(file.stem.split('mk')[1])-mk_offset
            with file.open() as inp:
                for row in csv.DictReader(inp, delimiter=';'):
                    time = float(row['TimeStep [s]'])
                    if abs(time*10-round(time*10)) < 1e-8:
                        transforms.writerow([number,marker,row['TimeStep [s]']]+[
                            row[f'm{i}{j}'].strip() for i in range(1,4) for j in range(1,5)])
        reader = vtkPolyDataReader()
        reader.SetFileName(str(folder/'Case__Actual.vtk'))
        reader.Update()
        data = reader.GetOutput()
        points = vtk_to_numpy(data.GetPoints().GetData())
        markers = vtk_to_numpy(data.GetCellData().GetArray('Mk'))
        for marker in sorted(set(markers)):
            ids = set()
            cells = 0
            for cell in range(data.GetNumberOfCells()):
                if markers[cell] == marker:
                    cells += 1
                    shape = data.GetCell(cell)
                    ids.update(shape.GetPointId(i) for i in range(shape.GetNumberOfPoints()))
            vertices = points[sorted(ids)]
            limits = [float(x) for pair in zip(vertices.min(axis=0),vertices.max(axis=0)) for x in pair]
            bounds.writerow([number,int(marker)-mk_offset]+limits+[len(vertices),cells])
            for point in vertices:
                vertices_writer.writerow([number,int(marker)-mk_offset]+[float(x) for x in point])

inputs = [args.case_source/f'Motion{number:02}.xml' for number in range(1,10)]
inputs += [args.case_source/name for name in ('Motion08mov_f3.txt','Motion09mov_deg.csv','Motion09mov_rad.csv')]
inputs += [args.tools/name for name in ('GenCase_win64.exe','BoundaryVTK_win64.exe')]
(args.output/'source_hashes.json').write_text(json.dumps({
    file.name:hashlib.sha256(file.read_bytes()).hexdigest() for file in inputs},indent=2)+'\n')
print(args.output)
