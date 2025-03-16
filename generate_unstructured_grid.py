import numpy as np
import h5py as h5

# -----------------------------------------------------------------
# Global metadata
fType = 'f'
idType = 'i8'
charType = 'uint8'

# -----------------------------------------------------------------
# Convenient method to fill h5py node with a numpy array
def append_dataset(dset, array):
  origLen = dset.shape[0]
  dset.resize(origLen + array.shape[0], axis=0)
  dset[origLen:] = array

# -----------------------------------------------------------------
# Generate a UnstructuredGrid VTKHDF structure based on the spec
def generate_structure_for_unstructured(root):
  root.attrs['Version'] = (2,1)
  ascii_type = 'UnstructuredGrid'.encode('ascii')
  root.attrs.create('Type', ascii_type, dtype=h5.string_dtype('ascii', len(ascii_type)))

  root.create_dataset('NumberOfPoints', (0,), maxshape=(None,), dtype=idType)
  root.create_dataset('Types', (0,), maxshape=(None,), dtype=charType)
  root.create_dataset('Points', (0,3), maxshape=(None,3), dtype=fType)

  root.create_dataset('NumberOfConnectivityIds', (0,), maxshape=(None,), dtype=idType)
  root.create_dataset('NumberOfCells', (0,), maxshape=(None,), dtype=idType)
  root.create_dataset('Offsets', (0,), maxshape=(None,), dtype=idType)
  root.create_dataset('Connectivity', (0,), maxshape=(None,), dtype=idType)

# -----------------------------------------------------------------
# Fill VTKHDF file with cubes
def fill_with_dummy_unstructured_grid(root, nCubePerDim = 1):
  # Additional meta information to generate a valid VTKHDF file
  fullDim = nCubePerDim * nCubePerDim * nCubePerDim
  cubePoints = 8
  numberOfConnectivity = 8
  numberOfPts = fullDim * cubePoints
  numberOfOffset = fullDim + 1

  points = np.array(np.empty([numberOfPts,3]))
  connectivity = np.array(np.empty([numberOfConnectivity * fullDim]))
  offsets = np.array(np.empty([fullDim+1]))
  types = np.array(np.empty([numberOfOffset-1]))

  [fillGeometry(idx, points, connectivity, offsets, types, numberOfConnectivity, nCubePerDim) for idx in range(0,fullDim)]

  offsets[fullDim] = fullDim * 8

  append_dataset(root['NumberOfPoints'], np.array([numberOfPts]))
  append_dataset(root['Points'], points)

  append_dataset(root['NumberOfConnectivityIds'], np.array([numberOfConnectivity*fullDim]))
  append_dataset(root['Connectivity'], connectivity)

  append_dataset(root['NumberOfCells'], np.array([numberOfOffset-1]))
  append_dataset(root['Offsets'], offsets)

  append_dataset(root['Types'], types)

# -----------------------------------------------------------------
def fillGeometry(cubeIdx, points, connectivity,offsets, types, numberOfConnectivity, nCubePerDim):

  XIdx = cubeIdx % nCubePerDim
  XQuot = cubeIdx // nCubePerDim

  YIdx = XQuot % nCubePerDim
  YQuot = XQuot // nCubePerDim

  ZIdx = YQuot % nCubePerDim

  points[0 + cubeIdx * 8] = [2 + (XIdx + 0) * 0.25 ,(YIdx + 0) * 0.25, (ZIdx + 0) * 0.25]
  points[1 + cubeIdx * 8] = [2 + (XIdx + 1) * 0.25 ,(YIdx + 0) * 0.25, (ZIdx + 0) * 0.25]
  points[2 + cubeIdx * 8] = [2 + (XIdx + 0) * 0.25 ,(YIdx + 1) * 0.25, (ZIdx + 0) * 0.25]
  points[3 + cubeIdx * 8] = [2 + (XIdx + 1) * 0.25 ,(YIdx + 1) * 0.25, (ZIdx + 0) * 0.25]
  points[4 + cubeIdx * 8] = [2 + (XIdx + 0) * 0.25 ,(YIdx + 0) * 0.25, (ZIdx + 1) * 0.25]
  points[5 + cubeIdx * 8] = [2 + (XIdx + 1) * 0.25 ,(YIdx + 0) * 0.25, (ZIdx + 1) * 0.25]
  points[6 + cubeIdx * 8] = [2 + (XIdx + 0) * 0.25 ,(YIdx + 1) * 0.25, (ZIdx + 1) * 0.25]
  points[7 + cubeIdx * 8] = [2 + (XIdx + 1) * 0.25 ,(YIdx + 1) * 0.25, (ZIdx + 1) * 0.25]


  connectivity[0 + cubeIdx * 8] = 0 + numberOfConnectivity * cubeIdx
  connectivity[1 + cubeIdx * 8] = 1 + numberOfConnectivity * cubeIdx
  connectivity[2 + cubeIdx * 8] = 3 + numberOfConnectivity * cubeIdx
  connectivity[3 + cubeIdx * 8] = 2 + numberOfConnectivity * cubeIdx
  connectivity[4 + cubeIdx * 8] = 4 + numberOfConnectivity * cubeIdx
  connectivity[5 + cubeIdx * 8] = 5 + numberOfConnectivity * cubeIdx
  connectivity[6 + cubeIdx * 8] = 7 + numberOfConnectivity * cubeIdx
  connectivity[7 + cubeIdx * 8] = 6 + numberOfConnectivity * cubeIdx

  cubePoints = 8
  offsets[cubeIdx] = cubeIdx * cubePoints

  # Code recognized by VTK as HEXAHEDRON
  types[cubeIdx] = 12

# -----------------------------------------------------------------
def generate_data(root):
  generate_structure_for_unstructured(root)

  fill_with_dummy_unstructured_grid(root, 4)

  return root

# -----------------------------------------------------------------
def generate_unstructured_grid(name):
  f = h5.File('unstructured.hdf', 'w')

  root = f.create_group(name)
  generate_data(root)

# -----------------------------------------------------------------
if __name__ == "__main__":
  generate_unstructured_grid('VTKHDF')
