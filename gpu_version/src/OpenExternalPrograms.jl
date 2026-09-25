"""
Utility wrappers for launching external programs such as a text editor or
ParaView. These helpers are optional conveniences used at the end of a
simulation run to quickly inspect the produced output.
"""
module OpenExternalPrograms

export AutoOpenLogFile, AutoOpenParaview

using ..SimulationLoggerConfiguration
using ..SimulationMetaDataConfiguration

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
    AutoOpenParaview(metadata, variable_names;
                     paraview_cmd="paraview",
                     representation="Point Gaussian",
                     color_variable="Density")

Write a ParaView state file for the given simulation and optionally
launch ParaView to visualise the results. `variable_names` should contain the
point arrays stored in the output files. Pass `paraview_cmd = nothing` to skip
launching ParaView automatically.
"""
function AutoOpenParaview(SimMetaData::SimulationMetaData, OutputVariableNames;
                          paraview_cmd::Union{String,Nothing}="paraview",
                          representation::String="Point Gaussian",
                          color_variable::String="Density")
    ## Generate auto paraview py

    if SimMetaData.ExportSingleVTKHDF
        ParaViewStateFileName = joinpath(SimMetaData.SaveLocation, SimMetaData.SimulationName) * "_SingleVTKHDFStateFile.py"
        py_regex = "$(SimMetaData.SimulationName).vtkhdf"
    else
        ParaViewStateFileName = joinpath(SimMetaData.SaveLocation, SimMetaData.SimulationName) * "_StateFile.py"
        py_regex = "^$(SimMetaData.SimulationName)_(\\d+).vtk" #^ means to anchor the regex to the start of the string
    end

    ExtractDimensionalityMetaData(::SimulationMetaData{N, FloatType}) where {N, FloatType} = N
    ViewDimension = ExtractDimensionalityMetaData(SimMetaData) == 2 ? "2D" : "3D"

    template_path = joinpath(@__DIR__, "AutoParaviewTemplate.py")
    template = read(template_path, String)
    script = replace(template,
                     "__SAVE_LOCATION__" => SimMetaData.SaveLocation,
                     "__PY_REGEX__" => py_regex,
                     "__SIM_NAME__" => SimMetaData.SimulationName,
                     "__OUTPUT_VARIABLES__" => "['" * join(OutputVariableNames, "', '") * "']",
                     "__REPRESENTATION__" => representation,
                     "__COLOR_VAR__" => color_variable,
                     "__VIEW_DIMENSION__" => ViewDimension,
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
