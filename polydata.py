import numpy as np
import h5py as h5

from vtkmodules.vtkCommonCore import vtkDoubleArray
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkFiltersCore import vtkAppendFilter
from vtkmodules.vtkFiltersGeometry import vtkGeometryFilter
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter

import vtkmodules.numpy_interface.dataset_adapter as dsa
from vtkmodules.util.numpy_support import (numpy_to_vtk as npvtk, vtk_to_numpy as vtknp)

fType = 'f'
idType = 'i8'

connectivities = ['Vertices', 'Lines', 'Polygons', 'Strips']

# -----------------------------------------------------------------
def generate_geometry_structure(root):
    root.attrs['Version'] = (2,0)
    ascii_type = 'PolyData'.encode('ascii')
    root.attrs.create('Type', ascii_type, dtype=h5.string_dtype('ascii', len(ascii_type))) 

    root.create_dataset('NumberOfPoints', (0,), maxshape=(None,), dtype=idType)
    root.create_dataset('Points', (0,3), maxshape=(None,3), dtype=fType)

    for connect in connectivities:
        group = root.create_group(connect)
        group.create_dataset('NumberOfConnectivityIds', (0,), maxshape=(None,), dtype=idType)
        group.create_dataset('NumberOfCells', (0,), maxshape=(None,), dtype=idType)
        group.create_dataset('Offsets', (0,), maxshape=(None,), dtype=idType)
        group.create_dataset('Connectivity', (0,), maxshape=(None,), dtype=idType)

    pData = root.create_group('PointData')
    pData.create_dataset('Warping', (0,3), maxshape=(None,3), dtype=fType)
    pData.create_dataset('Normals', (0,3), maxshape=(None,3), dtype=fType)

    cData = root.create_group('CellData')
    cData.create_dataset('Materials', (0,), maxshape=(None,), dtype=idType)

# -----------------------------------------------------------------
def generate_step_structure(root):
    steps = root.create_group('Steps')
    steps.attrs['NSteps'] = 0

    steps.create_dataset('Values', (0,), maxshape=(None,), dtype=fType)

    singleDSs = ['PartOffsets', 'NumberOfParts', 'PointOffsets']
    for name in singleDSs:
        steps.create_dataset(name, (0,), maxshape=(None,), dtype=idType)

    nTopoDSs = ['CellOffsets', 'ConnectivityIdOffsets']
    for name in nTopoDSs:
        steps.create_dataset(name, (0,4), maxshape=(None,4), dtype=idType)

    pData = steps.create_group('PointDataOffsets')
    pData.create_dataset('Warping', (0,), maxshape=(None,), dtype=idType)
    pData.create_dataset('Normals', (0,), maxshape=(None,), dtype=idType)

    cData = steps.create_group('CellDataOffsets')
    cData.create_dataset('Materials', (0,), maxshape=(None,), dtype=idType)

# -----------------------------------------------------------------
def append_data(root, poly_data, newStep=None, geometryOffset=None):

    if not newStep == None:
        steps = root['Steps']
        steps.attrs['NSteps'] = steps.attrs['NSteps'] + 1
        append_dataset(steps['Values'], np.array([newStep]))

        geomOffs = []
        if geometryOffset == None:
            geomOffs.append(root['NumberOfPoints'].shape[0])
            geomOffs.append(1)
            geomOffs.append(root['Points'].shape[0])
            for connect in connectivities:
                geomOffs.append(root[connect + '/Offsets'].shape[0] - geomOffs[0])
            for connect in connectivities:
                geomOffs.append(root[connect + '/Connectivity'].shape[0])
        else:
            geomOffs.append(steps['PartOffsets'][geometryOffset])
            geomOffs.append(1)
            geomOffs.append(steps['PointOffsets'][geometryOffset])
            for iC in range(len(connectivities)):
                geomOffs.append(steps['CellOffsets'][geometryOffset, iC])
            for connect in connectivities:
                geomOffs.append(steps['ConnectivityIdOffsets'][geometryOffset, iC])

        append_dataset(steps['PartOffsets'], np.array([geomOffs[0]]))
        append_dataset(steps['NumberOfParts'], np.array([geomOffs[1]]))
        append_dataset(steps['PointOffsets'], np.array([geomOffs[2]]))
        append_dataset(steps['CellOffsets'], np.array(geomOffs[3:3+len(connectivities)]).reshape(1, len(connectivities)))
        append_dataset(steps['ConnectivityIdOffsets'], np.array(geomOffs[3+len(connectivities):3+2*len(connectivities)]).reshape(1, len(connectivities)))

        append_dataset(steps['PointDataOffsets/Warping'], np.array([root['PointData/Warping'].shape[0]]))
        append_dataset(steps['PointDataOffsets/Normals'], np.array([root['PointData/Normals'].shape[0]]))
        append_dataset(steps['CellDataOffsets/Materials'], np.array([root['CellData/Materials'].shape[0]]))
    else:
        steps = root['Steps']
        steps['NumberOfParts'][-1] += 1


    if geometryOffset == None:
        append_dataset(root['NumberOfPoints'], np.array([poly_data.GetNumberOfPoints()]))
        append_dataset(root['Points'], poly_data.Points)

        cellArrays = [poly_data.GetVerts(), poly_data.GetLines(), poly_data.GetPolys(), poly_data.GetStrips()]
        for name, connect in zip(connectivities, cellArrays):
            ca = vtknp(connect.GetConnectivityArray())
            append_dataset(root[name]['NumberOfConnectivityIds'], np.array([ca.shape[0]]))
            append_dataset(root[name]['Connectivity'], ca)
            oa = vtknp(connect.GetOffsetsArray())
            append_dataset(root[name]['NumberOfCells'], np.array([oa.shape[0]-1]))
            append_dataset(root[name]['Offsets'], oa)

    append_dataset(root['PointData']['Warping'], poly_data.PointData['Warping'])
    append_dataset(root['PointData']['Normals'], poly_data.PointData['Normals'])
    append_dataset(root['CellData']['Materials'], poly_data.CellData['Materials'])

# -----------------------------------------------------------------
def append_dataset(dset, array):
    origLen = dset.shape[0]
    dset.resize(origLen + array.shape[0], axis=0)
    dset[origLen:] = array


# -----------------------------------------------------------------
def generate_static_spheres():
    sphereSrc0 = vtkSphereSource()
    sphereSrc0.Update()
    sphere0 = dsa.WrapDataObject(sphereSrc0.GetOutput())

    sphereSrc1 = vtkSphereSource()
    sphereSrc1.SetCenter(2.0, 2.0, 2.0)
    sphereSrc1.SetThetaResolution(20)
    sphereSrc1.SetPhiResolution(20)
    sphereSrc1.Update()
    sphere1 = dsa.WrapDataObject(sphereSrc1.GetOutput())

    warping0 = npvtk(np.cross(sphere0.Points, [0, 0, 1]).astype(fType))
    warping0.SetName('Warping')
    sphere0.GetPointData().AddArray(warping0)
    warping1 = npvtk(np.cross(sphere1.Points - [2, 2, 2], [0, 0, 1]).astype(fType))
    warping1.SetName('Warping')
    sphere1.GetPointData().AddArray(warping1)

    mats0 = npvtk(np.full((sphere0.GetNumberOfCells(),), 0, dtype=idType))
    mats0.SetName('Materials')
    sphere0.GetCellData().AddArray(mats0)
    mats1 = npvtk(np.full((sphere1.GetNumberOfCells(),), 1, dtype=idType))
    mats1.SetName('Materials')
    sphere1.GetCellData().AddArray(mats1)

    return [sphere0]

# -----------------------------------------------------------------
def generate_morphing_spheres(t, morphGeometry=False):
    pds = generate_static_spheres()

    for pd in pds:
        mod = (np.sin(pd.Points[:, 0]) + np.sin(pd.Points[:,1])+1)*np.cos(np.pi*t)/2.0
        if morphGeometry:
            pd.Points = pd.Points * mod
        warped = npvtk(pd.PointData['Warping'] * mod)
        warped.SetName('Warping')
        pd.GetPointData().AddArray(warped)

    return pds

# -----------------------------------------------------------------
def write_vtp(filepath, pds):
    appender = vtkAppendFilter()
    for iPD, pd in enumerate(pds):
        appender.AddInputData(pd.VTKObject)
    appender.Update()
    surface = vtkGeometryFilter()
    surface.SetInputConnection(appender.GetOutputPort())
    surface.Update()
    writer = vtkXMLPolyDataWriter()
    writer.SetFileName(filepath)
    writer.SetInputConnection(surface.GetOutputPort())
    writer.Update()

# -----------------------------------------------------------------
def generate_data(root):
    generate_geometry_structure(root)
    generate_step_structure(root)

    ts = np.linspace(0.0, 0.5, 6)
    for iT, t in enumerate(ts):
        pds = generate_morphing_spheres(t)
        append_data(root, pds[0], newStep=t, geometryOffset=(None if t == 0.0 else 0))
        write_vtp('hdf_transient_poly_data_' + str(iT).zfill(3) + '.vtp', pds)

    ts = np.linspace(0.6, 0.9, 4)
    for iT, t in enumerate(ts):
        pds = generate_morphing_spheres(t, True)
        append_data(root, pds[0], newStep=t, geometryOffset=None)
        write_vtp('hdf_transient_poly_data_' + str(iT + 6).zfill(3) + '.vtp', pds)

# -----------------------------------------------------------------
if __name__ == "__main__":
    f = h5.File('polydata_transient.vtkhdf', 'w')
    root = f.create_group('VTKHDF')
    generate_data(root)
    f.close()
