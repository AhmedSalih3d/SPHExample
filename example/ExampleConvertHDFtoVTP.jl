using Distributed
@time let
    FileToConvert = "E:/SecondApproach/TESTING_CPU/Test.h5"
    SavePath      = dirname(FileToConvert)
    SaveName      = splitext(basename(FileToConvert))[1]
    println(SaveName)

    nWorkers      = Base.Threads.nthreads()

    if nworkers() < nWorkers  # Check if the desired number of workers is already available
        addprocs(nWorkers - nworkers())  # Add more workers if needed
    end

    # This block ensures that each worker is using the same environment as the master process
    @everywhere begin
        # using Pkg
        # Pkg.activate(".") # Activate the current project environment. Adjust the path as necessary.
        # Pkg.instantiate() # Ensure all dependencies are installed
        using SPHExample
    end

    @everywhere using HDF5

    fid = h5open(FileToConvert,"r")
            
    all_keys = keys(fid)

    # Distribute processing of keys across workers
    results_future = @distributed for key in all_keys
        # Each worker reads and processes a subset of keys
        local DictVariable = nothing
        # It's generally safer to open the file separately in each worker to avoid concurrency issues
        h5open(FileToConvert, "r") do file
            DictVariable = read(file[key])
        end
        filepath = SavePath * "/" * SaveName * "_" * key * ".vtp"  # Adjust output path as needed
        ConvertHDFtoVTP(filepath, DictVariable)
        # Return value from the loop if needed, e.g., a status message or result
        "Processed $(key)"
    end

    close(fid)
end