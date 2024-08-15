module OpenExternalPrograms

export AutoOpenLogFile

using ..SimulationLoggerConfiguration
using ..SimulationMetaDataConfiguration


function AutoOpenLogFile(SimLogger::SimulationLogger, SimMetaData::SimulationMetaData)
    LogFileName = replace(strip(SimLogger.LoggerIo.name,['<', '>']), "file " => "")
    if SimMetaData.OpenLogFile
        try
            OpenLogFileCommand = `notepad "$(LogFileName)"`
            run(OpenLogFileCommand; wait = false)
        catch e
            @warn("Unable to open log file automatically. It uses notepad for Windows by default.", e)
        end
    end
end

end