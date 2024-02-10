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
Points         = [SVector{3,Float64}(1,2,3), SVector{3,Float64}(4,5,6)]
Kernel         = Float64.([100, 200]) #rand(Float64,N)
KernelGradient = [SVector{3,Float64}(-1,1,0), SVector{3,Float64}(1,-1,0)]

function PolyDataTemplate(filename::String, points::AbstractVector ; kwargs...)
        # Generate the XML document and then put in some fixed values
        xml_doc = Document(Declaration(version=1.0,encoding="utf-8"))
        vtk_file = Element("VTKFile")
        vtk_file.attributes["type"]        = "PolyData"
        vtk_file.attributes["version"]     = "1.0"
        vtk_file.attributes["byte_order"]  = "LittleEndian"
        vtk_file.attributes["header_type"] = "UInt64"

        # PolyData is the main section, filling it out
        polydata  = Element("PolyData")
        piece     = Element("Piece")
        N = length(points)
        piece.attributes["NumberOfPoints"] = string(N)

        # This Points element and its associated DataArray has to be constructed individually
        points_element    = Element("Points")
        point_dataarray = create_data_array_element("Points",points)
        point_dataarray["offset"] = 0

        # Generate XML tags for kwargs data
        pointdata  = Element("PointData")
        dataarrays = Vector{XML.Node}()
        for (name, data) in kwargs
            push!(dataarrays, create_data_array_element(string(name),data))
        end

        # Generate appended data element
        appendeddata = Element("AppendedData")
        appendeddata.attributes["encoding"] = "raw"

        # Process of writing file has begun
        
        NB = 0
        io = IOBuffer()
        write(io,"\n_")
        TP = getproperty(Base, Symbol(point_dataarray.attributes["type"]))
        UncompressedHeaderN = N * parse(Int,point_dataarray.attributes["NumberOfComponents"]) *  sizeof(TP)
        NB += write(io, UncompressedHeaderN)
        NB += custom_write(io, points)

        # This loop here calculates the correct offsets and puts the specified data in
        for (arr,keyval) in zip(dataarrays,kwargs)
            T   = getproperty(Base, Symbol(point_dataarray.attributes["type"]))
            Nc  = parse(Int,arr.attributes["NumberOfComponents"])
            N   = length(keyval.second) #data = keyval.second, since it is Pair
            Tsz = sizeof(T)

            arr.attributes["offset"] = string(NB)

            HowManyBytes  = Tsz*Nc*N
            NB += write(io, HowManyBytes)
            NB += custom_write(io,keyval.second) 
        end

        # Take the result from the buffer, turn to string and write it
        v = take!(io)
        t = Text(String(v))
        write(io,"\n")
        push!(appendeddata,t)
        close(io)

        # Glue all xml pieces together
        push!(xml_doc,vtk_file)
        push!(points_element,point_dataarray)
        push!(piece,points_element)
        push!(polydata,piece)
        push!(vtk_file,polydata)
        map(x -> push!(pointdata, x), dataarrays)
        push!(piece,pointdata)
        push!(vtk_file,appendeddata)

        XML.write(filename,xml_doc)
end

PolyDataTemplate(raw"E:\SPH\TestOfFile.vtp", Points; Kernel, KernelGradient)