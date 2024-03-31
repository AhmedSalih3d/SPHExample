#https://www.analytech-solutions.com/analytech-solutions/blog/binary-io.html
using EzXML
using StaticArrays

### Functions=================================================
# Function to create a DataArray element for VTK files
function create_data_array_element(parent, name::String, data::AbstractVector{T}, offset::Int) where T
    # Create the DataArray elements
    dataarray = addelement!(parent, "DataArray")
    
    # Set attributes based on the input vector's type
    dataarray["type"              ] =  string(eltype(first(data)))
    dataarray["Name"              ] =  name
    dataarray["NumberOfComponents"] =  string(Int(sizeof(first(data))/sizeof(eltype(first(data)))))
    dataarray["format"            ] =  "appended"
    dataarray["offset"            ] =  string(offset)
    return dataarray
end

function PolyDataTemplate(filename::String, points, variable_names, args...)
        # Generate the XML document and then put in some fixed values
        # xml_doc = XMLDocument() #Declaration(version=1.0,encoding="utf-8")
        xml_doc = parsexml("""
        <?xml version="1.0" encoding="UTF-8" ?>
        <VTKFile>
        </VTKFile>
        """)
        vtk_file      = root(xml_doc)
        setnodecontent!(vtk_file,""" type="PolyData" version="1.0" byte_order="LittleEndian" header_type="UInt64" """)

        polydata = addelement!(vtk_file,"PolyData")

        piece    = addelement!(polydata, "Piece")
        N = length(points)
        piece["NumberOfPoints"]  = string(N)

        # This Points element and its associated DataArray has to be constructed individually
        points_element    = addelement!(piece, "Points")
        point_dataarray   = create_data_array_element(points_element, "Points",points,0)

        
        # Generate appended data element
        appendeddata = addelement!(vtk_file, "AppendedData")
        appendeddata["encoding"] = "raw"

        # Start writing the file and generating the correct dataarrays with the right offsets in the loop
        NB = 0
        io = IOBuffer()
        write(io,"\n_")
        UncompressedHeaderN  = N * length(first(points)) *  sizeof(typeof(first(points)))
        NB += write(io, UncompressedHeaderN)
        NB += write(io, points)

        # Generate XML tags for kwargs data
        pointdata  = addelement!(piece, "PointData")
        dataarrays = Vector{EzXML.Node}(undef,length(args))

        for i in eachindex(args)
            arg           = args[i]
            dataarrays[i] = create_data_array_element(pointdata, variable_names[i],arg,NB)

            A             = typeof(first(arg))
            T             = eltype(A)
            Ni            = length(arg)
            Tsz           = sizeof(T)
            Nc            = Int( sizeof(A) / Tsz )
            HowManyBytes  = Tsz*Nc*Ni + Tsz

            NB           += HowManyBytes

            write(io, NB)
            write(io, arg)
        end

        # Take the result from the buffer, turn to string and write it
        v = take!(io)
        setnodecontent!(appendeddata, Base.unsafe_convert(Cstring, v))
        close(io)

        write(filename,xml_doc)
end

### === 
save_location = raw"E:\SPH\TestOfFile.vtp"

# Points         = [SVector{3,Float64}(1,2,3), SVector{3,Float64}(4,5,6)]
# Kernel         = Float64.([100, 200]) 
# KernelGradient = [SVector{3,Float64}(-1,1,0), SVector{3,Float64}(1,-1,0)]
N              = 6195
Points         = rand(SVector{3,Float64},N) * 10
Kernel         = rand(Float64,N) * 1000
KernelGradient = rand(SVector{3,Float64},N) * 100

d = @report_opt target_modules=(@__MODULE__,) PolyDataTemplate(save_location, Points, ["Kernel", "KernelGradient"], Kernel, KernelGradient)
println(d)

@profview PolyDataTemplate(save_location, Points, ["Kernel", "KernelGradient"], Kernel, KernelGradient)

b = @b PolyDataTemplate($save_location, $Points, $(["Kernel", "KernelGradient"]), $Kernel, $KernelGradient)
display(b)

# @code_warnt ype PolyDataTemplate(save_location, Points, ["Kernel", "KernelGradient"], Kernel, KernelGradient)

PolyDataTemplate(save_location, Points, ["Kernel", "KernelGradient"], Kernel, KernelGradient)