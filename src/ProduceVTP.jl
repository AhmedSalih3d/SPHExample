#https://www.analytech-solutions.com/analytech-solutions/blog/binary-io.html
using XML
using XML: Document, Declaration, Element, Text
using Base64: base64encode
using StaticArrays
using WriteVTK
const IS_LITTLE_ENDIAN = ENDIAN_BOM == 0x04030201

N              = 1
Points = [SVector{3,Float64}(1,2,3)]
#Points         = rand(SVector{3,Float64},N)
# Kernel         = rand(Float64,N)
# KernelGradient = rand(SVector{3,Float64},N)

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
# piece.attributes["NumberOfVerts"] = "0"
# piece.attributes["NumberOfPolys"] = "0"
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
dataarray1.attributes["offset"]              = "0"

dataarray2 = Element("DataArray")
dataarray2.attributes["type"]                = "Float64"
dataarray2.attributes["Name"]                = "KernelGradient"
dataarray2.attributes["NumberOfComponents"]  = "3"
dataarray2.attributes["format"]              = "appended"
dataarray2.attributes["offset"]              = "0"

# push!(pointdata, dataarray1, dataarray2)
# push!(piece,pointdata)

appendeddata = Element("AppendedData")
appendeddata.attributes["encoding"] = "raw"

push!(vtk_file,appendeddata)

        # Initialize containers for VTK data structure
        polys = empty(MeshCell{WriteVTK.PolyData.Polys, UnitRange{Int64}}[])
        verts = empty(MeshCell{WriteVTK.PolyData.Verts, UnitRange{Int64}}[])
        all_cells = (verts, polys)



# Function to write a single SVector to a buffer in binary format
function write_svector(io, vec)
    for element in vec
        write(io, element)
    end
end



# Open a file in binary write mode. Change 'yourfile.bin' to your desired file name.
        
io = IOBuffer()
write(io,"\n")
write(io,"_")
write(io,Char(24))
write(io,Char(0)^7)
# write_svector(io, Points)

# Write the data to the buffer
for vec in Points
   write_svector(io, vec)
end

write(io,"\n")
v = take!(io)
t = Text(String(v))
print("This is the result from custom writer:    ", v[2:end]) #Skip newline
push!(appendeddata,t)

XML.write(raw"E:\SPH\TestOfFile.vtp",xml_doc)

# Overwrite a bit of WriteVTK to produce simplified working file

function WriteVTK.vtk_grid(dtype::WriteVTK.VTKPolyData, filename::AbstractString,
        points::WriteVTK.UnstructuredCoords,
        cells::Vararg{AbstractArray{<:WriteVTK.PolyCell}}; kwargs...)
        Npts = WriteVTK.num_points(dtype, points)
        Ncls = sum(length, cells)

        xvtk = WriteVTK.XMLDocument()
        vtk = WriteVTK.DatasetFile(dtype, xvtk, filename, Npts, Ncls; kwargs...)

        xroot = WriteVTK.vtk_xml_write_header(vtk)
        xGrid = WriteVTK.new_child(xroot, vtk.grid_type)

        xPiece = WriteVTK.new_child(xGrid, "Piece")
        WriteVTK.set_attribute(xPiece, "NumberOfPoints", Npts)

        xPoints = WriteVTK.new_child(xPiece, "Points")
        WriteVTK.data_to_xml(vtk, xPoints, points, "Points", 3)

        vtk
end


function WriteVTK.data_to_xml_appended(vtk::WriteVTK.DatasetFile, xDA::WriteVTK.XMLElement, data)
        @assert vtk.appended
    
        buf = vtk.buf    # append buffer
        buf_check = IOBuffer()    # append buffer
        #compress = vtk.compression_level > 0
        compress = false
    
        # DataArray node
        WriteVTK.set_attribute(xDA, "format", "appended")
        WriteVTK.set_attribute(xDA, "offset", position(buf))
    
        # Size of data array (in bytes).
        nb = WriteVTK.sizeof_data(data)
    
        # if compress
        #     initpos = position(buf)
    
        #     # Write temporary data that will be replaced later with the real header.
        #     let header = ntuple(d -> zero(WriteVTK.HeaderType), Val(4))
        #         write(buf, header...)
        #     end
    
        #     # Write compressed data.
        #     zWriter = WriteVTK.ZlibCompressorStream(buf, level=vtk.compression_level)
        #     WriteVTK.write_array(zWriter, data)
        #     write(zWriter, WriteVTK.TranscodingStreams.TOKEN_END)
        #     flush(zWriter)
        #     WriteVTK.TranscodingStreams.finalize(zWriter.codec) # Release allocated resources (issue #43)
    
        #     # Go back to `initpos` and write real header.
        #     endpos = position(buf)
        #     compbytes = endpos - initpos - 4 * sizeof(WriteVTK.HeaderType)
        #     let header = WriteVTK.HeaderType.((1, nb, nb, compbytes))
        #         seek(buf, initpos)
        #         write(buf, header...)
        #         seek(buf, endpos)
        #     end
        # else
            write(buf, WriteVTK.HeaderType(nb))  # header (uncompressed version)
            nb_write = WriteVTK.write_array(buf, data)
            @assert nb_write == nb

            # buf_check to output
            write(buf_check, WriteVTK.HeaderType(nb))  # header (uncompressed version)
            nb_write = WriteVTK.write_array(buf_check, data)
            @assert nb_write == nb
            println("This is the result from WriteVTK:         _", join(Char.(take!(buf_check))))
        # end
    
        xDA
    end
    

# Initialize containers for VTK data structure
# Create a .vtp file with the specified positions
vtk_grid(raw"E:\SPH\TestOfFileWriteVTK.vtp", Points, all_cells...) do vtk
end;
