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
    dataarray.attributes["type"]               = string(eltype(first(data)))
    dataarray.attributes["Name"]               = name  
    dataarray.attributes["NumberOfComponents"] = string(Int(sizeof(first(data))/sizeof(eltype(first(data)))))
    dataarray.attributes["format"]             = "appended"
    dataarray.attributes["offset"]             = "nan"  # Placeholder, to be replaced later
    
    return dataarray
end

# Function to write a single SVector to a buffer in binary format
function custom_write(io, vec)
   nb = 0
   for element in vec
        nb += write(io, element)
   end

   return nb
end
    

###===========================================================
N              = 1
Points         = [SVector{3,Float64}(1,2,3)]
Kernel         = [Float64.(100)] #rand(Float64,N)
KernelGradient = [SVector{3,Float64}(-1,1,0)]

function PolyDataTemplate(filename::String, points::Vector ; kwargs...)
        xml_doc = Document(Declaration(version=1.0,encoding="utf-8"))
        vtk_file = Element("VTKFile")
        vtk_file.attributes["type"]        = "PolyData"
        vtk_file.attributes["version"]     = "1.0"
        vtk_file.attributes["byte_order"]  = "LittleEndian"
        vtk_file.attributes["header_type"] = "UInt64"

        polydata  = Element("PolyData")
        piece     = Element("Piece")
        piece.attributes["NumberOfPoints"] = string(N)

        points_element    = Element("Points")

        dataarray = create_data_array_element("Points",points)
        dataarray["offset"] = 0

        pointdata  = Element("PointData")

        dataarrays = Vector{XML.Node}()
        for (name, data) in kwargs
            push!(dataarrays, create_data_array_element(string(name),data))
        end

        appendeddata = Element("AppendedData")
        appendeddata.attributes["encoding"] = "raw"

        # Open a file in binary write mode. Change 'yourfile.bin' to your desired file name.
        
        NB = 0
        io = IOBuffer()
        write(io,"\n_")
        NB += write(io, Char(24)*Char(0)^7)
        NB += custom_write(io, points)
        write(io,8)
        write(io,Kernel)
        write(io,24)
        write(io,KernelGradient)

        println(NB)

        offset = 0
        for arr in dataarrays
            arr.attributes["offset"]  = string(NB)
            NB                       += 8
        end

        # dataarray1.attributes["offset"]              = string(8 + sizeof(Float64)*N*3)
        # dataarray2.attributes["offset"]              = string(3 * 8 + sizeof(Float64)*N*3)

        v = take!(io)
        t = Text(String(v))
        write(io,"\n")
        push!(appendeddata,t)
        close(io)

        # Glue all xml pieces together
        push!(xml_doc,vtk_file)
        push!(points_element,dataarray)
        push!(piece,points_element)
        push!(polydata,piece)
        push!(vtk_file,polydata)
        map(x -> push!(pointdata, x), dataarrays)
        push!(piece,pointdata)
        push!(vtk_file,appendeddata)

        XML.write(filename,xml_doc)
end

PolyDataTemplate(raw"E:\SPH\TestOfFile.vtp", Points ; Kernel, KernelGradient)