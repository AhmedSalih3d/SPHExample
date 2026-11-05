"""Export GenCase -save:all output to portable CSVs using pvpython.

Usage: pvpython export_chrono_fixture.py CasePrefix fixture_directory [2|3]
SPH 2D coordinates use world X,Z; display coordinates use X,Z,-Y.
"""
import csv
import hashlib
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import vtk

prefix = Path(sys.argv[1])
target = Path(sys.argv[2])
dimension = int(sys.argv[3]) if len(sys.argv) > 3 else 3
target.mkdir(parents=True, exist_ok=True)
root = ET.parse(str(prefix) + '.xml').getroot()
definition = root.find('casedef/geometry/definition')
dx = float(definition.attrib['dp'])
origin = [float(definition.find('pointmin').attrib[d]) for d in ('x','y','z')]
reader = vtk.vtkPolyDataReader()
reader.SetFileName(str(prefix) + '_All.vtk')
reader.Update()
data = reader.GetOutput()
if data.GetNumberOfPoints() == 0:
    raise RuntimeError('GenCase particle output is empty or missing')
mk = data.GetPointData().GetArray('Mk')
rho = data.GetPointData().GetArray('Rhop')
kind = data.GetPointData().GetArray('Type')
rows = []
for p in range(data.GetNumberOfPoints()):
    position = data.GetPoint(p)
    index = [round((position[d]-origin[d])/dx) for d in range(3)]
    if max(abs(position[d]-origin[d]-dx*index[d]) for d in range(3)) > 1e-3*dx:
        raise RuntimeError(f'Non-lattice point {position}: preserve its coordinates explicitly instead of quantizing')
    rows.append((int(mk.GetTuple1(p)), int(kind.GetTuple1(p)), *index, rho.GetTuple1(p)))
with (target/'lattice.csv').open('w',newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['group','type','i','j','k','density'])
    writer.writerows(sorted(rows))
digest = hashlib.sha256(''.join(','.join(map(str,row[:5]))+'\n' for row in sorted(rows)).encode()).hexdigest()
surface_files = sorted(prefix.parent.glob(prefix.name+'*_Actual.vtk'))
if not surface_files:
    raise RuntimeError('No GenCase Actual surfaces found')
append = vtk.vtkAppendPolyData()
for path in surface_files:
    surface = vtk.vtkPolyDataReader()
    surface.SetFileName(str(path))
    surface.Update()
    append.AddInputData(surface.GetOutput())
append.Update()
data = append.GetOutput()
mk = data.GetCellData().GetArray('Mk')
with (target/'surface_points.csv').open('w',newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['id','x','y','z'])
    for p in range(data.GetNumberOfPoints()):
        x,y,z = data.GetPoint(p)
        writer.writerow([p+1,x,z,-y] if dimension == 2 else [p+1,x,y,z])
with (target/'surface_cells.csv').open('w',newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['group','n','v1','v2','v3','v4'])
    for i in range(data.GetNumberOfCells()):
        cell = data.GetCell(i)
        ids = [cell.GetPointId(j)+1 for j in range(cell.GetNumberOfPoints())]
        assert 1 <= len(ids) <= 4
        writer.writerow([int(mk.GetTuple1(i)),len(ids),*ids,*([0]*(4-len(ids)))])
print('dimension',dimension,'dx',dx,'origin',origin,'particles',len(rows),'sha256',digest)
