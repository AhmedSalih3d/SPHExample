using XML
using XML: Document, Declaration, Element, Text

  ### Functions=================================================
    # Function to create a DataArray element for VTK files
function create_data_array_element(name::String, data::AbstractVector{T}, offset::Int) where T
    # Create the DataArray elements
    dataarray = Element("DataArray")
    
    # Set attributes based on the input vector's type
    dataarray.attributes["type"]               = string(eltype(first(data)))
    dataarray.attributes["Name"]               = name  
    dataarray.attributes["NumberOfComponents"] = string(Int(sizeof(first(data))/sizeof(eltype(first(data)))))
    dataarray.attributes["format"]             = "appended"
    dataarray.attributes["offset"]             = string(offset)
    return dataarray
end

function ConvertHDFtoVTP(filename::String, dict)
    points = reinterpret(reshape, SVector{3,Float64}, read(fid["0"]["Position"]))
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
    point_dataarray = create_data_array_element("Points",points,0)
    point_dataarray["offset"] = 0

    
    # Generate appended data element
    appendeddata = Element("AppendedData")
    appendeddata.attributes["encoding"] = "raw"

    # Start writing the file and generating the correct dataarrays with the right offsets in the loop
    NB = 0
    io = IOBuffer()
    write(io,"\n_")
    UncompressedHeaderN  = N * length(first(points)) *  sizeof(typeof(first(points)))
    NB += write(io, UncompressedHeaderN)
    NB += write(io, points)

    # Generate XML tags for kwargs data
    pointdata  = Element("PointData")
    pop!(dict,"Position")
    dataarrays = Vector{XML.Node}(undef,length(dict))

    i = 1
    for (key,value) in dict
        
        T = eltype(value)
        NumberOfFields = fieldcount(eltype(T))

        if NumberOfFields > 0
            T = unique(fieldtypes(eltype(T)))[1]
            data_type = SVector{NumberOfFields, T}
        else
            data_type = T
        end

        val = reinterpret(reshape, data_type, value)
        arg           = val

        dataarrays[i] = create_data_array_element(key,arg,NB)
        A             = typeof(first(arg))
        T             = eltype(A)
        Ni            = length(arg)
        Tsz           = sizeof(T)
        Nc            = Int( sizeof(A) / Tsz )
        HowManyBytes  = Tsz*Nc*Ni + Tsz
        NB           += HowManyBytes
        write(io, NB)
        write(io, arg)

        i += 1
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
    map(x -> push!(pointdata,x), dataarrays)
    push!(piece,pointdata)
    push!(vtk_file,appendeddata)

    XML.write(filename,xml_doc)
end

using HDF5
using StaticArrays
using ChunkSplitters

fid = h5open(raw"E:\SecondApproach\TESTING_CPU\Test.h5", "r")

data = read(fid["150"])

dict_keys = keys(fid)


# Preparing for parallezation but not enabled
@time for (_, inds) in enumerate(chunks(dict_keys; n=Base.Threads.nthreads()))
    for iter_ in inds
        iter = dict_keys[iter_]
        dict = read(fid[iter])
        ConvertHDFtoVTP("E:/SecondApproach/TESTING_CPU/" * iter * ".vtp", dict)
    end
end
