fid = h5open(raw"E:\SecondApproach\TESTING_CPU\Test.h5", "r")

data = read(fid["150"])

# Function to convert a type to its corresponding SVector type or return the original type.
function type_to_svector_or_same(T::Type)
    # Check if T is a named tuple type with the specific structure we're interested in.
    if T.name === NamedTuple.name && haskey(T.parameters[1], :data)
        # Extract the :data field type, which should be another NamedTuple.
        data_field_type = T.parameters[1][:data]
        
        # Count the number of elements and their types in the :data field.
        element_types = Tuple([v for v in values(data_field_type.parameters[1])])
        
        # Return the corresponding SVector type.
        return SVector{length(element_types), eltype(element_types)}
    else
        # If T is not a matching named tuple type, return it unchanged.
        return T
    end
end

for (key, value) in data
    T = eltype(value)
    NumberOfFields = fieldcount(eltype(T))

    if NumberOfFields > 0
        T = unique(fieldtypes(eltype(T)))[1]
        data_type = SVector{NumberOfFields, T}
    else
        data_type = T
    end

    val = reinterpret(reshape, data_type, value)
end

keys(fid)