fid = h5open(raw"E:\SecondApproach\TESTING_CPU\Test.h5", "r")

data = read(fid["150"])

i = 0
while i < length(keys(fid))
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

        if key == "Position"
            ExportVTP("E:/SecondApproach/TESTING_CPU/" * string(i) * ".vtp", val)
        end
    end

    i += 1
end

keys(fid)