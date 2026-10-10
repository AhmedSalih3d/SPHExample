# Experimental source transformations; these never change the installed solver.
const HAND_SOLVE_SOURCE = raw"""
@inline function hand_solve4(A::SMatrix{4,4,T}, b::SVector{4,T}) where {T}
    r1 = SVector{5,T}(A[1,1], A[1,2], A[1,3], A[1,4], b[1])
    r2 = SVector{5,T}(A[2,1], A[2,2], A[2,3], A[2,4], b[2])
    r3 = SVector{5,T}(A[3,1], A[3,2], A[3,3], A[3,4], b[3])
    r4 = SVector{5,T}(A[4,1], A[4,2], A[4,3], A[4,4], b[4])
    sign = one(T)
    if abs(r2[1]) > abs(r1[1]); r1, r2 = r2, r1; sign = -sign; end
    if abs(r3[1]) > abs(r1[1]); r1, r3 = r3, r1; sign = -sign; end
    if abs(r4[1]) > abs(r1[1]); r1, r4 = r4, r1; sign = -sign; end
    r2 -= (r2[1] / r1[1]) * r1
    r3 -= (r3[1] / r1[1]) * r1
    r4 -= (r4[1] / r1[1]) * r1
    if abs(r3[2]) > abs(r2[2]); r2, r3 = r3, r2; sign = -sign; end
    if abs(r4[2]) > abs(r2[2]); r2, r4 = r4, r2; sign = -sign; end
    r3 -= (r3[2] / r2[2]) * r2
    r4 -= (r4[2] / r2[2]) * r2
    if abs(r4[3]) > abs(r3[3]); r3, r4 = r4, r3; sign = -sign; end
    r4 -= (r4[3] / r3[3]) * r3
    determinant = sign * r1[1] * r2[2] * r3[3] * r4[4]
    result = zero(b)
    if abs(determinant) >= 1e-3
        x4 = r4[5] / r4[4]
        x3 = (r3[5] - r3[4] * x4) / r3[3]
        x2 = ((r2[5] - r2[4] * x4) - r2[3] * x3) / r2[2]
        x1 = (((r1[5] - r1[4] * x4) - r1[3] * x3) - r1[2] * x2) / r1[1]
        result = SVector{4,T}(x1, x2, x3, x4)
    end
    return determinant, result
end

@inline function hand_solve4(A::SMatrix{3,3,T}, b::SVector{3,T}) where {T}
    determinant = det(A)
    return determinant, abs(determinant) >= 1e-3 ? A \ b : zero(b)
end
"""

function hand_solve_source(source)
    source = replace(source, "# mDBC: ghost node interpolation and density correction" =>
        "# mDBC: ghost node interpolation and density correction\n" * HAND_SOLVE_SOURCE)
    return replace(source, "if abs(det(A)) >= 1e-3\n            sol  = A \\ b" =>
        "determinant, sol = hand_solve4(A, b)\n        if abs(determinant) >= 1e-3")
end

function const_load_source(source)
    marker = "(; m₀, dx)  = SimConstants"
    loads = join(("    $field = CUDA.Const($field)" for field in
        ("Pairs", "Density", "InvDensity", "Pressure", "Velocity", "ParticleType", "CellStart", "CellID")), '\n')
    return replace(source, marker => loads * "\n    " * marker)
end

function advanced_screen(cases)
    baseline = kernel_variant("PerfAdvancedBaseline")
    handwritten = kernel_variant("PerfHandSolve", hand_solve_source)
    readonly = kernel_variant("PerfConstLoads", const_load_source)
    combined = kernel_variant("PerfCombinedRegisters", source -> replace(source,
        "for row in cell_rows(grid)" =>
            "for row in GPUCellGrid.CellRows{D,typeof(grid).parameters[2]}(grid.dims[1], grid.dims[1] * grid.dims[2])",
        "@cuda threads=threads" => "@cuda maxregs=64 threads=threads"))
    variants = [("baseline", baseline, 128), ("hand_solve", handwritten, 128),
        ("const_loads", readonly, 128), ("lazy_regs64", combined, 128)]
    Base.invokelatest(measure_kernel_screen, cases, variants)
end
