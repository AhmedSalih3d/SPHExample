"""
Utility wrappers for launching external programs such as a text editor or
ParaView. These helpers are optional conveniences used at the end of a
simulation run to quickly inspect the produced output.
"""
module OpenExternalPrograms

export AutoOpenLogFile, AutoOpenParaview, OpenParaviewFile

using ..SimulationLoggerConfiguration
using ..SimulationMetaDataConfiguration
using ..SimulationConstantsConfiguration
using ..SimulationGeometry: Fluid

# Include one particle spacing because positions represent particle centres.
function hydrostatic_pressure_range(particles, constants)
    heights = (Float64(last(particles.Position[i]))
        for i in eachindex(particles.Type) if particles.Type[i] == Fluid)
    isempty(heights) && return nothing
    low, high = extrema(heights)
    maximum_pressure = Float64(constants.ρ₀) * abs(Float64(constants.g)) *
        (high - low + Float64(constants.dx))
    return maximum_pressure > 0 ? (0.0, maximum_pressure) : nothing
end

"""
    _default_open_command(path)

Return a platform specific [`Cmd`] used to open `path` with the default
application.
"""
function _default_open_command(path::AbstractString)
    if Sys.iswindows()
        return `notepad $(path)`
    elseif Sys.isapple()
        return `open $(path)`
    else
        return `xdg-open $(path)`
    end
end

"""
    OpenParaviewFile(filepath; paraview_cmd = "paraview")

Open an existing VTK or VTKHDF file in ParaView. Pass `paraview_cmd = nothing`
to skip launching ParaView.
"""
function OpenParaviewFile(filepath::AbstractString;
                          paraview_cmd::Union{String,Nothing}="paraview")
    isfile(filepath) ||
        throw(ArgumentError("ParaView input file does not exist: $filepath"))
    paraview_cmd === nothing && return nothing
    if isnothing(Sys.which(paraview_cmd))
        @warn("ParaView command $(paraview_cmd) not found; skipping visualisation")
        return nothing
    end

    try
        run(`$(paraview_cmd) $(abspath(filepath))`; wait=false)
    catch e
        @error("Unable to open $(filepath) in ParaView", e)
    end
    return nothing
end


"""
    AutoOpenLogFile(logger, metadata; editor_cmd=nothing)

Open the simulation log file using an external editor. If `editor_cmd` is not
provided, a platform specific default is used. Setting `editor_cmd = nothing`
disables the automatic opening entirely.
"""
function AutoOpenLogFile(SimLogger::SimulationLogger,
                         SimMetaData::SimulationMetaData;
                         editor_cmd::Union{String,Nothing}=nothing)
    log_file = replace(strip(SimLogger.LoggerIo.name, ['<', '>']), "file " => "")
    if SimMetaData.OpenLogFile && !isempty(log_file)
        cmd = editor_cmd === nothing ? _default_open_command(log_file) :
              `$(editor_cmd) $(log_file)`
        try
            run(cmd; wait=false)
        catch e
            @warn("Unable to open log file automatically", e)
        end
    end

    return nothing
end

"""
    AutoOpenParaview(metadata, constants, variable_names;
                     paraview_cmd="paraview",
                     representation="Point Gaussian",
                     color_variable=nothing, pressure_range=nothing)

Write a ParaView state file for the given simulation and optionally
launch ParaView to visualise the results. `variable_names` should contain the
point arrays stored in the output files. Pass `paraview_cmd = nothing` to skip
launching ParaView automatically. With `ExportGridCells = true`, load the cell
grid alongside the particles in the same view, displayed as a wireframe.
Color by pressure when exported, otherwise density. `pressure_range = (min, max)`
sets a fixed pressure color scale in Pa, also used when switching to pressure
coloring later. Without a range, use the initial output's pressure data range.
"""
function AutoOpenParaview(SimMetaData::SimulationMetaData, 
                          SimConstants::SimulationConstants,
                          OutputVariableNames;
                          paraview_cmd::Union{String,Nothing}="paraview",
                          representation::String="Point Gaussian",
                          color_variable::Union{String,Nothing}="Density",
                          pressure_range=nothing)
    ## Generate auto paraview py
    if pressure_range !== nothing
        length(pressure_range) == 2 ||
            throw(ArgumentError("pressure_range must contain two values in Pa"))
        pmin, pmax = Float64.(pressure_range)
        all(isfinite, (pmin, pmax)) && pmin < pmax ||
            throw(ArgumentError("pressure_range must be finite and increasing"))
        pressure_range = (pmin, pmax)
    end

    if SimMetaData.ExportSingleVTKHDF
        ParaViewStateFileName = joinpath(SimMetaData.SaveLocation, SimMetaData.SimulationName) * "_SingleVTKHDFStateFile.py"
    else
        ParaViewStateFileName = joinpath(SimMetaData.SaveLocation, SimMetaData.SimulationName) * "_StateFile.py"
    end

    ExtractDimensionalityMetaData(::SimulationMetaData{N, FloatType}) where {N, FloatType} = N
    ViewDimension = ExtractDimensionalityMetaData(SimMetaData) == 2 ? "2D" : "3D"

    template_path = joinpath(@__DIR__, "AutoParaviewTemplate.py")
    template = read(template_path, String)
    script = replace(template,
                     "__SAVE_LOCATION__" => replace(abspath(SimMetaData.SaveLocation),
                                                   '\\' => '/'),
                     "__SINGLE_FILE__" => (SimMetaData.ExportSingleVTKHDF ? "True" : "False"),
                     "__EXPORT_GRID__" => (SimMetaData.ExportGridCells ? "True" : "False"),
                     "__SIM_NAME__" => SimMetaData.SimulationName,
                     "__OUTPUT_VARIABLES__" => "['" * join(OutputVariableNames, "', '") * "']",
                     "__REPRESENTATION__" => representation,
                     "__COLOR_VAR__" => color_variable,
                     "__PRESSURE_RANGE__" => (pressure_range === nothing ? "None" :
                         "[$(pressure_range[1]), $(pressure_range[2])]"),
                     "__VIEW_DIMENSION__" => ViewDimension,
                     "__GAUSSIAN_RADIUS__" => SimConstants.dx / 2,
                     )
    open(ParaViewStateFileName, "w") do io
        write(io, script)
    end

    if SimMetaData.VisualizeInParaview && paraview_cmd !== nothing
        if isnothing(Sys.which(paraview_cmd))
            @warn("ParaView command $(paraview_cmd) not found; skipping visualisation")
        else
            try
                OpenInParaview = `$(paraview_cmd) --state="$(ParaViewStateFileName)"`
                run(OpenInParaview; wait=false)
            catch e
                @error("You must add Paraview to path as $(paraview_cmd) and use at minimum version 5.12", e)
            end
        end
    end

    return nothing
end

end
