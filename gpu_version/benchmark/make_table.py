"""Combine the CPU / GPU benchmark logs into a markdown table with speedups.

    python gpu_version/benchmark/make_table.py final_cpu.log final_gpu_f32.log final_gpu_f64.log
"""
import re
import sys

ROW = re.compile(r"((?:StillWedge|DamBreak|MovingSquare|Duckling)[A-Za-z0-9_.]+)\s+(\d+)\s+(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)")


def parse(path):
    rows = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = ROW.search(line)
            if m:
                name, n, steps, loop, ms, total = m.groups()
                rows[name] = dict(n=int(n), steps=int(steps), ms=float(ms))
    return rows


def main(cpu_path, f32_path, f64_path):
    cpu, f32, f64 = parse(cpu_path), parse(f32_path), parse(f64_path)
    order = [k for k in cpu if k in f32]
    print("| Case | Dim | Particles | CPU 24 threads, Float64 [ms/step] | GPU Float32 [ms/step] | Speedup Float32 | GPU Float64 [ms/step] | Speedup Float64 |")
    print("|------|-----|----------:|------------:|------------:|--------:|------------:|--------:|")
    for k in order:
        dim = "3D" if "3D" in k else "2D"
        c = cpu[k]["ms"]
        g32 = f32[k]["ms"]
        g64 = f64.get(k, {}).get("ms")
        g64s = f"{g64:.2f}" if g64 else "-"
        s64 = f"{c / g64:.1f}x" if g64 else "-"
        print(f"| {k} | {dim} | {cpu[k]['n']:,} | {c:.2f} | {g32:.3f} | **{c / g32:.1f}x** | {g64s} | {s64} |")


if __name__ == "__main__":
    main(*sys.argv[1:4])
