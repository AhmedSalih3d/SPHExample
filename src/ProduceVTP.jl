#https://www.analytech-solutions.com/analytech-solutions/blog/binary-io.html
using XML
using XML: Document, Declaration, Element, Text
using StaticArrays

### Functions=================================================
# Function to create a DataArray element for VTK files
function create_data_array_element(name::String, data::AbstractVector{T}) where T
    # Create the DataArray elements
    dataarray = Element("DataArray")
    
    # Set attributes based on the input vector's type
    dataarray.attributes["type"] = String(eltype(first(data)))
    dataarray.attributes["Name"] = name  
    dataarray.attributes["NumberOfComponents"] = String(Int(sizeof(first(s))/sizeof(eltype(first(data)))))
    dataarray.attributes["format"] = "appended"
    dataarray.attributes["offset"] = "nan"  # Placeholder, to be replaced later
    
    return dataarray
end

# Function to write a single SVector to a buffer in binary format
function write_svector(io, vec)
   for element in vec
        write(io, element)
   end
end
    

###===========================================================

# Example usage with a Vector of SVector{3, Float64}
using StaticArrays: SVector
data = [SVector{3,Float64}(1,2,3)]  # Example data

points_element = create_data_array_element(data)

# To view the generated XML structure
println(XML.to_string(points_element))


N              = 1
Points = [SVector{3,Float64}(1,2,3)]
#Points         = rand(SVector{3,Float64},N)
Kernel         = Float64.(100) #rand(Float64,N)
KernelGradient = rand(SVector{3,Float64},N)

xml_doc = Document(Declaration(version=1.0,encoding="utf-8"))
vtk_file = Element("VTKFile")
vtk_file.attributes["type"]        = "PolyData"
vtk_file.attributes["version"]     = "1.0"
vtk_file.attributes["byte_order"]  = "LittleEndian"
vtk_file.attributes["header_type"] = "UInt64"
push!(xml_doc,vtk_file)

polydata  = Element("PolyData")
piece     = Element("Piece")
piece.attributes["NumberOfPoints"] = string(N)

points    = Element("Points")
dataarray = Element("DataArray")
dataarray.attributes["type"]                = "Float64"
dataarray.attributes["Name"]                = "Points"
dataarray.attributes["NumberOfComponents"]  = "3"
dataarray.attributes["format"]              = "appended"
dataarray.attributes["offset"]              = "0"

push!(points,dataarray)
push!(piece,points)
push!(polydata,piece)
push!(vtk_file,polydata)


pointdata  = Element("PointData")
dataarray1 = Element("DataArray")
dataarray1.attributes["type"]                = "Float64"
dataarray1.attributes["Name"]                = "Kernel"
dataarray1.attributes["NumberOfComponents"]  = "1"
dataarray1.attributes["format"]              = "appended"


# dataarray2 = Element("DataArray")
# dataarray2.attributes["type"]                = "Float64"
# dataarray2.attributes["Name"]                = "KernelGradient"
# dataarray2.attributes["NumberOfComponents"]  = "3"
# dataarray2.attributes["format"]              = "appended"
# dataarray2.attributes["offset"]              = "0"

push!(pointdata, dataarray1)
push!(piece,pointdata)

appendeddata = Element("AppendedData")
appendeddata.attributes["encoding"] = "raw"

push!(vtk_file,appendeddata)

# Open a file in binary write mode. Change 'yourfile.bin' to your desired file name.
        
io = IOBuffer()
write(io,"\n")
write(io,"_")
write(io,Char(24))
write(io,Char(0)^7)
# Write the data to the buffer
for vec in Points
   write_svector(io, vec)
end
write(io,8)
write(io,Kernel)
dataarray1.attributes["offset"]              = string(8 + sizeof(Float64)*N*3)

v = take!(io)
t = Text(String(v))
push!(appendeddata,t)
write(io,"\n")


XML.write(raw"E:\SPH\TestOfFile.vtp",xml_doc)
