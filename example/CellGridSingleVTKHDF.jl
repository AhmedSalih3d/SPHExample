using SPHExample
using HDF5
using StaticArrays

UniqueCells = [
    CartesianIndex(2, 1)
    CartesianIndex(3, 1)
    CartesianIndex(4, 1)
    CartesianIndex(5, 1)
    CartesianIndex(6, 1)
    CartesianIndex(7, 1)
    CartesianIndex(8, 1)
    CartesianIndex(9, 1)
    CartesianIndex(10, 1)
    CartesianIndex(11, 1)
    CartesianIndex(12, 1)
    CartesianIndex(13, 1)
    CartesianIndex(22, 1)
    CartesianIndex(23, 1)
    CartesianIndex(24, 1)
]


SimConstants = SimulationConstants{Float64}(dx=0.02,c₀=42.48576250492629, δᵩ = 0.1, CFL=0.2)

# SaveCellGridVTKHDF("test1.vtkhdf", SimConstants, UniqueCells)

OutputCellGridVTKHDF = h5open("single_test1" * ".vtkhdf", "w")
root         = HDF5.create_group(OutputCellGridVTKHDF, "VTKHDF")
GenerateGeometryStructure(root; vtk_file_type = "UnstructuredGrid")
GenerateStepStructure(root; vtk_file_type = "UnstructuredGrid")



AppendVTKHDFGridData(root, 0, SimConstants, UniqueCells)

UniqueCells = [
    CartesianIndex(2, 1)
]

AppendVTKHDFGridData(root, 1, SimConstants, UniqueCells)
close(OutputCellGridVTKHDF)