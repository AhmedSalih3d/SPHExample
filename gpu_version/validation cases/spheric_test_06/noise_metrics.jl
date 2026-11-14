# Shared deterministic analysis for native/DualSPHysics drag; no simulation.
using CSV, Statistics, LinearAlgebra

function drag_metrics(times, pressure, viscous, reference)
    result = Dict{String,Float64}()
    spectra = NamedTuple[]
    for (name, values, column) in (("pressure",pressure,2), ("viscous",viscous,3))
        sample(t,v,x) = SphericTest06.interpolate(t,v,x)
        edges = collect(0.1:0.1:8.)
        estimate = [bin_average(times,values,a,b) for (a,b) in zip(edges[1:end-1],edges[2:end])]
        target = [bin_average(reference[:,1],reference[:,column],a,b) for (a,b) in zip(edges[1:end-1],edges[2:end])]
        result[name*"_nrmse"] = norm(estimate-target)/norm(target)
        # Exact time integral of the squared piecewise-linear error on the
        # union of both time grids. This exposes noise hidden by 0.1 s bins.
        knots = sort!(unique!([0.1; times[(times .> 0.1) .& (times .< 8)];
            reference[(reference[:,1] .> 0.1) .& (reference[:,1] .< 8),1]; 8.0]))
        predicted_raw = [sample(times,values,t) for t in knots]
        target_raw = [sample(reference[:,1],reference[:,column],t) for t in knots]
        residual = predicted_raw-target_raw
        square_integral(v) = sum(diff(knots).*(v[1:end-1].^2+v[1:end-1].*v[2:end]+v[2:end].^2))/3
        result[name*"_raw_nrmse"] = sqrt(square_integral(residual)/square_integral(target_raw))
        result[name*"_late_bias"] = mean(estimate[40:end])/mean(target[40:end])-1
        nodes = collect(0.3:0.02:7.7)
        difference = [sample(times,values,t)-sample(reference[:,1],reference[:,column],t) for t in nodes]
        selected = collect(1.5:0.02:6.5)
        error = [sample(nodes,difference,t) for t in selected]
        noise = [sample(nodes,difference,t)-bin_average(nodes,difference,t-0.25,t+0.25) for t in selected]
        result[name*"_highpass_rms"] = sqrt(mean(abs2,noise))
        result[name*"_raw_error_rms"] = sqrt(mean(abs2,error))
        # Remove mean/linear drift before applying a Hann window.
        design = hcat(ones(length(selected)), selected .- mean(selected))
        detrended = error-design*(design\error)
        window = 0.5 .- 0.5cos.(2π .* (0:length(error)-1)/(length(error)-1))
        frequencies = collect(1:floor(Int,20length(error)*0.02)) ./ (length(error)*0.02)
        amplitudes = [2abs(sum(detrended .* window .* cis.(-2π*f.*selected)))/sum(window) for f in frequencies]
        band = findall(>=(1),frequencies)
        result[name*"_dominant_hz"] = frequencies[band[argmax(amplitudes[band])]]
        for (frequency,amplitude) in zip(frequencies,amplitudes)
            push!(spectra,(; component = name, frequency, amplitude))
        end
    end
    return result, spectra
end
