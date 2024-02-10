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
    dataarray.attributes["type"] = string(eltype(first(data)))
    dataarray.attributes["Name"] = name  
    dataarray.attributes["NumberOfComponents"] = string(Int(sizeof(first(data))/sizeof(eltype(first(data)))))
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
N              = 1
Points         = [SVector{3,Float64}(1,2,3)]
Kernel         = [Float64.(100)] #rand(Float64,N)
KernelGradient = [SVector{3,Float64}(-1,1,0)]

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

points_element    = Element("Points")

dataarray = create_data_array_element("Points",Points)
dataarray["offset"] = 0
push!(points_element,dataarray)

push!(points,dataarray)
push!(piece,points_element)
push!(polydata,piece)
push!(vtk_file,polydata)


pointdata  = Element("PointData")
dataarray1 = create_data_array_element("Kernel", Kernel)
dataarray2 = create_data_array_element("KernelGradient", KernelGradient)

push!(pointdata, dataarray1)
push!(pointdata, dataarray2)
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
write(io,24)
write(io,KernelGradient)
dataarray1.attributes["offset"]              = string(8 + sizeof(Float64)*N*3)
dataarray2.attributes["offset"]              = string(3 * 8 + sizeof(Float64)*N*3)

v = take!(io)
t = Text(String(v))
write(io,"\n")
push!(appendeddata,t)
close(io)



XML.write(raw"E:\SPH\TestOfFile.vtp",xml_doc)
