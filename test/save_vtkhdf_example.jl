using HDF5

# Open an HDF5 file
filename = "it_only_took_a_few_hours_sigh.vtkhdf"
file = h5open(filename, "w") 

# Create the top-level group 'VTKHDF' and set its attributes
vtkhdf_group = create_group(file, "VTKHDF")
write_attribute(vtkhdf_group, "Version", [1, 0])

# write_attribute(vtkhdf_group,"Type", "ImageData")
# Create the ASCII string datatype with fixed length and null padding
strtype = HDF5.h5t_copy(HDF5.H5T_C_S1)
HDF5.h5t_set_size(strtype, 9)  # Length of the string "ImageData"
HDF5.h5t_set_strpad(strtype, HDF5.H5T_STR_NULLPAD)  # Null padding
HDF5.h5t_set_cset(strtype, HDF5.H5T_CSET_ASCII)  # Set character set to ASCII
# Define a scalar dataspace for the attribute
dataspace_id = HDF5.dataspace(1)
# Create the 'Type' attribute with the specific string type and scalar dataspace
attr_id = HDF5.h5a_create(vtkhdf_group.id, "Type", strtype, dataspace_id, HDF5.H5P_DEFAULT, HDF5.H5P_DEFAULT)
# Write the string "ImageData" to the 'Type' attribute
HDF5.h5a_write(attr_id, strtype, codeunits("ImageData"))

whole_extent = [0, 3, 0, 1, 0, 0]
write_attribute(vtkhdf_group,"WholeExtent", whole_extent)

write_attribute(vtkhdf_group, "Origin",    [0.0, 0.0, 0.0])
write_attribute(vtkhdf_group, "Spacing", [1.0, 1.0, 1.0])
write_attribute(vtkhdf_group, "Direction", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

# Create the PointData group and the dataset inside it
field_data_group = create_group(vtkhdf_group, "PointData")
attr = create_attribute(field_data_group, "Scalars", datatype(String), HDF5.dataspace(1))
HDF5.API.h5a_write(attr, datatype(String), Ref{Cstring}(["PNGImage"]))
field_data_group["PNGImage"] = Array(UInt8[1 2 3 4; 5 6 7 8]')
close(file)

# Path to the file for reference
filename