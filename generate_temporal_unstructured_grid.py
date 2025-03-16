import numpy as np
import h5py as h5
import random
# -----------------------------------------------------------------
# Global metadata
fType = 'f'
idType = 'i8'
charType = 'uint8'

# -----------------------------------------------------------------
# Convenient method to append data to an HDF5 dataset
def append_dataset(dset, array):
    origLen = dset.shape[0]
    dset.resize(origLen + array.shape[0], axis=0)
    dset[origLen:] = array

# -----------------------------------------------------------------
# Generate the VTK HDF Unstructured Grid structure with time dependency
def generate_structure_for_unstructured(root):
    root.attrs['Version'] = (2,1)
    ascii_type = 'UnstructuredGrid'.encode('ascii')
    root.attrs.create('Type', ascii_type, dtype=h5.string_dtype('ascii', len(ascii_type)))

    # Create the "Steps" group to manage time-dependent data
    steps = root.create_group('Steps')
    steps.attrs.create('NSteps', 0, dtype=idType)  # Number of time steps

    steps.create_dataset('Values', (0,), maxshape=(None,), dtype=fType)  # Time values
    steps.create_dataset('PointOffsets', (0,), maxshape=(None,), dtype=idType)
    steps.create_dataset('ConnectivityIdOffsets', (0,), maxshape=(None,), dtype=idType)
    steps.create_dataset('PartOffsets', (0,), maxshape=(None,), dtype=idType)
    steps.create_dataset('CellOffsets', (0,), maxshape=(None,), dtype=idType)
    steps.create_dataset('NumberOfParts', (0,), maxshape=(None,), dtype=idType)

    # Geometry-related datasets
    root.create_dataset('NumberOfPoints', (0,), maxshape=(None,), dtype=idType)
    root.create_dataset('Types', (0,), maxshape=(None,), dtype=charType)
    root.create_dataset('Points', (0,3), maxshape=(None,3), dtype=fType)

    root.create_dataset('NumberOfConnectivityIds', (0,), maxshape=(None,), dtype=idType)
    root.create_dataset('NumberOfCells', (0,), maxshape=(None,), dtype=idType)
    root.create_dataset('Offsets', (0,), maxshape=(None,), dtype=idType)
    root.create_dataset('Connectivity', (0,), maxshape=(None,), dtype=idType)

# -----------------------------------------------------------------
# Add a new timestep of data
def add_time_step(root, time, nCubePerDim=1):
    steps = root['Steps']

    # Store the new time value
    append_dataset(steps['Values'], np.array([time]))

    # Get the number of existing steps
    nSteps = steps.attrs['NSteps']

    # Generate the grid data for this timestep
    fullDim = nCubePerDim ** 3
    cubePoints = 8
    numberOfConnectivity = 8
    numberOfPts = fullDim * cubePoints
    numberOfOffset = fullDim + 1

    points = np.empty([numberOfPts,3], dtype=fType)
    connectivity = np.empty([numberOfConnectivity * fullDim], dtype=idType)
    offsets = np.empty([fullDim+1], dtype=idType)
    types = np.empty([numberOfOffset-1], dtype=charType)

    [fillGeometry(idx, points, connectivity, offsets, types, numberOfConnectivity, nCubePerDim) for idx in range(fullDim)]

    offsets[fullDim] = fullDim * 8  # Set last offset correctly

    # Append data to corresponding datasets
    append_dataset(root['NumberOfPoints'], np.array([numberOfPts]))
    append_dataset(root['Points'], points)

    append_dataset(root['NumberOfConnectivityIds'], np.array([numberOfConnectivity * fullDim]))
    append_dataset(root['Connectivity'], connectivity)

    append_dataset(root['NumberOfCells'], np.array([numberOfOffset-1]))
    append_dataset(root['Offsets'], offsets)

    append_dataset(root['Types'], types)

    # Track step-specific offsets
    PointsStartIndex = np.sum(root['NumberOfPoints'][:nSteps]) if nSteps > 0 else 0
    ConnectivityStartIndex = np.sum(root['NumberOfConnectivityIds'][:nSteps]) if nSteps > 0 else 0
    CellStartIndex = np.sum(root['NumberOfCells'][:nSteps]) if nSteps > 0 else 0

    append_dataset(steps['PointOffsets'], np.array([PointsStartIndex]))
    append_dataset(steps['ConnectivityIdOffsets'], np.array([ConnectivityStartIndex]))
    append_dataset(steps['CellOffsets'], np.array([CellStartIndex]))

    append_dataset(steps['NumberOfParts'], np.array([1]))  # Single part for now
    append_dataset(steps['PartOffsets'], np.array([nSteps]))  # Each step is a new part

    # Update NSteps
    steps.attrs.modify('NSteps', nSteps + 1)

# -----------------------------------------------------------------
def fillGeometry(cubeIdx, points, connectivity, offsets, types, numberOfConnectivity, nCubePerDim):
    XIdx = cubeIdx % nCubePerDim
    XQuot = cubeIdx // nCubePerDim

    YIdx = XQuot % nCubePerDim
    YQuot = XQuot // nCubePerDim

    ZIdx = YQuot % nCubePerDim

    points[0 + cubeIdx * 8] = [2 + (XIdx + 0) * 0.25 * random.random(),(YIdx + 0) * 0.25 * random.random(), (ZIdx + 0) * 0.25 * random.random()]
    points[1 + cubeIdx * 8] = [2 + (XIdx + 1) * 0.25 * random.random(),(YIdx + 0) * 0.25 * random.random(), (ZIdx + 0) * 0.25 * random.random()]
    points[2 + cubeIdx * 8] = [2 + (XIdx + 0) * 0.25 * random.random(),(YIdx + 1) * 0.25 * random.random(), (ZIdx + 0) * 0.25 * random.random()]
    points[3 + cubeIdx * 8] = [2 + (XIdx + 1) * 0.25 * random.random(),(YIdx + 1) * 0.25 * random.random(), (ZIdx + 0) * 0.25 * random.random()]
    points[4 + cubeIdx * 8] = [2 + (XIdx + 0) * 0.25 * random.random(),(YIdx + 0) * 0.25 * random.random(), (ZIdx + 1) * 0.25 * random.random()]
    points[5 + cubeIdx * 8] = [2 + (XIdx + 1) * 0.25 * random.random(),(YIdx + 0) * 0.25 * random.random(), (ZIdx + 1) * 0.25 * random.random()]
    points[6 + cubeIdx * 8] = [2 + (XIdx + 0) * 0.25 * random.random(),(YIdx + 1) * 0.25 * random.random(), (ZIdx + 1) * 0.25 * random.random()]
    points[7 + cubeIdx * 8] = [2 + (XIdx + 1) * 0.25 * random.random(),(YIdx + 1) * 0.25 * random.random(), (ZIdx + 1) * 0.25 * random.random()]

    connectivity[cubeIdx * 8:(cubeIdx+1) * 8] = np.arange(8) + cubeIdx * numberOfConnectivity
    offsets[cubeIdx] = cubeIdx * 8
    types[cubeIdx] = 12  # VTK_HEXAHEDRON

# -----------------------------------------------------------------
# Create the HDF file and generate the initial data structure
def generate_unstructured_grid(name):
    f = h5.File('unstructured_temporal.hdf', 'w')

    root = f.create_group(name)
    generate_structure_for_unstructured(root)

    # Add multiple time steps
    time_steps = [0.0, 0.1, 0.2, 0.3]  # Example time steps
    nCubePerDim_ = [1, 2, 3]
    for (t,v) in zip(time_steps, nCubePerDim_):
        add_time_step(root, t, nCubePerDim=v)

    f.close()

# -----------------------------------------------------------------
if __name__ == "__main__":
    generate_unstructured_grid('VTKHDF')
