# Load the editable case's definitions without its final interactive run call.
Base.include(expr -> expr isa Expr && expr.head === :call && first(expr.args) === :run_wavemaker ? nothing : expr,
    @__MODULE__, joinpath(@__DIR__, "WavemakerCase.jl"))
