"""Create a ParaView state and screenshot comparing raw SPH drag with SPHERIC FD data.

The state contains side-by-side pressure and viscous drag charts. Each available
SPH resolution in the selected results directory is plotted without filtering.
An optional generated simulation state places the particles above the charts.
"""
import argparse
import csv
import math
import re
import runpy
from textwrap import dedent
from pathlib import Path

from paraview.simple import (
    CSVReader,
    CreateLayout,
    CreateView,
    GetLayout,
    GetViews,
    GetAnimationScene,
    ProgrammableSource,
    RenderAllViews,
    ResetSession,
    SaveScreenshot,
    SaveState,
    Show,
)


CASE = Path(__file__).resolve().parent
DEFAULT_RESULTS = CASE / "results" / "validation"
DEFAULT_OUTPUT = CASE / "runs" / "paraview_force_plot"
RESOLUTION_COLORS = {
    20: (0.83, 0.36, 0.16),
    40: (0.14, 0.56, 0.54),
    60: (0.36, 0.32, 0.68),
}
REFERENCE_COLOR = (0.12, 0.12, 0.12)


def read_drag_history(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"time", "cd_pressure", "cd_viscous"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"No drag samples in {path}")

    last_time = -math.inf
    samples = []
    for row_number, row in enumerate(rows, start=2):
        try:
            time, pressure, viscous = (
                float(row[name]) for name in ("time", "cd_pressure", "cd_viscous")
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid drag sample at {path}:{row_number}") from error
        if not all(math.isfinite(value) for value in (time, pressure, viscous)):
            raise ValueError(f"Nonfinite drag sample at {path}:{row_number}")
        if time <= last_time:
            raise ValueError(f"Drag times must increase strictly in {path}")
        last_time = time
        samples.append((time, pressure, viscous))
    return samples


def read_reference(path: Path):
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            values = [float(value) for value in stripped.split()]
        except ValueError:
            if rows:
                raise ValueError(f"Unexpected reference text at {path}:{line_number}")
            continue
        if len(values) != 3 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"Invalid reference sample at {path}:{line_number}")
        if rows and values[0] <= rows[-1][0]:
            raise ValueError(f"Reference times must increase strictly in {path}")
        rows.append(values)
    if not rows:
        raise ValueError(f"No reference samples in {path}")
    return rows


def write_reference_csv(rows, path: Path):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time", "cd_pressure", "cd_viscous"))
        writer.writerows(rows)


def reference_time_offset(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if "motion_time" not in reader.fieldnames:
            return 0.0
        offsets = [float(row["time"]) - float(row["motion_time"]) for row in reader]
    delay = offsets[0]
    if not math.isfinite(delay) or delay < 0 or any(
        not math.isfinite(value) or not math.isclose(value, delay, abs_tol=1e-9)
        for value in offsets
    ):
        raise ValueError(f"Inconsistent motion-time offset in {path}")
    return delay


def find_histories(results: Path, reynolds: int, resolution):
    if not results.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results}")
    pattern = re.compile(rf"Re{reynolds:03d}_n(\d+)\.csv")
    matches = []
    for path in results.iterdir():
        match = pattern.fullmatch(path.name)
        if match and (resolution is None or int(match.group(1)) == resolution):
            matches.append((int(match.group(1)), path))
    matches.sort()
    if not matches:
        suffix = "" if resolution is None else f" at L/dp={resolution}"
        raise FileNotFoundError(f"No Re={reynolds} drag histories{suffix} in {results}")
    histories = []
    for n, path in matches:
        end_time = read_drag_history(path)[-1][0]
        histories.append((n, path, end_time))
    return histories


def make_chart_view(layout, anchor, reynolds, component, histories, reference_reader, end_time,
                    location=0):
    view = CreateView("XYChartView")
    if anchor is None:
        if layout.SMProxy.AssignViewToAnyCell(view.SMProxy, location) < 0:
            raise RuntimeError("Could not place the first force chart in the ParaView layout")
    else:
        split_location = layout.SplitViewHorizontal(anchor, 0.5)
        if split_location < 0:
            raise RuntimeError("Could not split the ParaView layout for the second force chart")
        assigned_location = layout.SMProxy.AssignViewToAnyCell(view.SMProxy, split_location)
        if assigned_location < 0:
            raise RuntimeError("Could not place the second force chart in the ParaView layout")

    component_column = f"cd_{component}"
    component_label = "pressure" if component == "pressure" else "viscous"
    component_symbol = "p" if component == "pressure" else "v"
    view.ChartTitle = f"Re = {reynolds}: {component_label} drag"
    view.BottomAxisTitle = "Time [s]"
    view.LeftAxisTitle = f"Drag coefficient Cd,{component_symbol} [-]"
    view.BottomAxisUseCustomRange = 1
    view.BottomAxisRangeMinimum = 0.0
    view.BottomAxisRangeMaximum = end_time
    view.ShowBottomAxisGrid = 1
    view.ShowLeftAxisGrid = 1
    view.ShowLegend = 1
    view.HideTimeMarker = 1
    view.ChartTitleFontSize = 14
    view.BottomAxisTitleFontSize = 12
    view.LeftAxisTitleFontSize = 12
    view.LegendFontSize = 9

    for n, reader, _ in histories:
        display = Show(reader, view, "XYChartRepresentation")
        display.UseIndexForXAxis = 0
        display.XArrayName = "time"
        display.SeriesVisibility = [component_column]
        display.SeriesLabel = [component_column, f"Julia SPH, L/dp={n}"]
        color = RESOLUTION_COLORS.get(n, (0.28, 0.48, 0.72))
        display.SeriesColor = [component_column, *(str(value) for value in color)]
        display.SeriesLineStyle = [component_column, "1"]

    reference_display = Show(reference_reader, view, "XYChartRepresentation")
    reference_display.UseIndexForXAxis = 0
    reference_display.XArrayName = "time"
    reference_display.SeriesVisibility = [component_column]
    reference_display.SeriesLabel = [component_column, "Official finite-difference reference"]
    reference_display.SeriesColor = [component_column, *(str(value) for value in REFERENCE_COLOR)]
    reference_display.SeriesLineStyle = [component_column, "2"]
    return view


def add_time_cursor(view, component, histories, reference_rows):
    n, path, _ = histories[-1]
    samples = read_drag_history(path)
    column = 1 if component == "pressure" else 2
    times = [sample[0] for sample in samples]
    values = [sample[column] for sample in samples]
    all_values = [row[column] for row in reference_rows]
    for _, history_path, _ in histories:
        all_values.extend(row[column] for row in read_drag_history(history_path))
    low, high = min(all_values), max(all_values)
    margin = 0.05 * (high - low) if high > low else 0.05 * max(1.0, abs(low))
    marker = f"Current SPH drag, L/dp={n}"
    cursor = ProgrammableSource(registrationName=f"{component.capitalize()} time cursor")
    cursor.OutputDataSetType = "vtkTable"
    cursor.ScriptRequestInformation = dedent(f"""\
        from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline as pipeline
        times = {times!r}
        info = self.GetExecutive().GetOutputInformation(0)
        info.Set(pipeline.TIME_STEPS(), times, len(times))
        info.Set(pipeline.TIME_RANGE(), [times[0], times[-1]], 2)
        """)
    cursor.Script = dedent(f"""\
        import builtins
        from bisect import bisect_right
        from vtkmodules.vtkCommonCore import vtkDoubleArray
        from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline as pipeline
        times, values = {times!r}, {values!r}
        info = self.GetExecutive().GetOutputInformation(0)
        time = info.Get(pipeline.UPDATE_TIME_STEP()) if info.Has(pipeline.UPDATE_TIME_STEP()) else times[0]
        if times[0] <= time <= times[-1]:
            if len(times) == 1:
                drag = values[0]
            else:
                index = builtins.min(bisect_right(times, time) - 1, len(times) - 2)
                fraction = (time - times[index]) / (times[index + 1] - times[index])
                drag = values[index] + fraction * (values[index + 1] - values[index])
        else:
            drag = float("nan")
        table = self.GetOutputDataObject(0)
        table.Initialize()
        for name, data in (("time", (time, time)),
                           ("Time cursor", ({low - margin!r}, {high + margin!r})),
                           ({marker!r}, (drag, drag))):
            array = vtkDoubleArray()
            array.SetName(name)
            for value in data:
                array.InsertNextValue(value)
            table.AddColumn(array)
        """)
    cursor.UpdatePipeline()
    display = Show(cursor, view, "XYChartRepresentation")
    display.UseIndexForXAxis = 0
    display.XArrayName = "time"
    display.SeriesVisibility = ["Time cursor", marker]
    display.SeriesColor = ["Time cursor", "0.45", "0.45", "0.45",
                           marker, "1", "0.5", "0"]
    display.SeriesLineStyle = ["Time cursor", "2", marker, "0"]
    display.SeriesMarkerStyle = ["Time cursor", "0", marker, "4"]
    display.SeriesMarkerSize = ["Time cursor", "0", marker, "9"]


def make_force_plot(reynolds: int, args):
    histories = find_histories(args.results, reynolds, args.n)
    reference_path = CASE / "reference" / f"Force_Re{reynolds:03d}.dat"
    reference_rows = read_reference(reference_path)
    delays = [reference_time_offset(path) for _, path, _ in histories]
    if any(not math.isclose(delay, delays[0], abs_tol=1e-9) for delay in delays):
        raise ValueError("All force histories in one comparison must have the same initial hold")
    reference_rows = [(time + delays[0], pressure, viscous) for time, pressure, viscous in reference_rows]
    if args.simulation_state is not None:
        if not args.simulation_state.is_file():
            raise FileNotFoundError(f"Simulation state does not exist: {args.simulation_state}")
        runpy.run_path(str(args.simulation_state.resolve()))
        render_views = [view for view in GetViews() if view.GetXMLName() == "RenderView"]
        if len(render_views) != 1:
            raise ValueError("The simulation state must contain exactly one particle render view")
        layout = GetLayout(render_views[0])
        chart_location = layout.SplitViewVertical(render_views[0], 0.55)
        if chart_location < 0:
            raise RuntimeError("Could not split the particle view to add the force charts")
    else:
        layout = CreateLayout(name=f"SPHERIC Test 06 Re{reynolds:03d} force comparison")
        chart_location = 0
    args.output.mkdir(parents=True, exist_ok=True)
    reference_csv = args.output / f"Force_Re{reynolds:03d}_reference.csv"
    write_reference_csv(reference_rows, reference_csv)

    reference_reader = CSVReader(
        registrationName=f"Official_FD_reference_Re{reynolds:03d}",
        FileName=[str(reference_csv.resolve())],
    )
    reference_reader.UpdatePipeline()
    readers = []
    for n, path, end_time in histories:
        reader = CSVReader(
            registrationName=f"Julia_Re{reynolds:03d}_Ldp{n:03d}",
            FileName=[str(path.resolve())],
        )
        reader.UpdatePipeline()
        readers.append((n, reader, end_time))

    components = ("pressure", "viscous") if args.component == "both" else (args.component,)
    first_view = None
    end_time = max(sample_end for _, _, sample_end in readers)
    for component in components:
        view = make_chart_view(
            layout, first_view, reynolds, component, readers, reference_reader, end_time,
            location=chart_location,
        )
        add_time_cursor(view, component, histories, reference_rows)
        if first_view is None:
            first_view = view

    image_resolution = [1800 if len(components) == 2 else 1200,
                        1200 if args.simulation_state is not None else 850]
    layout.SetSize(*image_resolution)
    GetAnimationScene().UpdateAnimationUsingDataTimeSteps()
    RenderAllViews()
    state_path = args.output / f"ForceComparison_Re{reynolds:03d}.pvsm"
    image_path = args.output / f"ForceComparison_Re{reynolds:03d}.png"
    SaveState(str(state_path.resolve()))
    SaveScreenshot(
        str(image_path.resolve()),
        layout,
        ImageResolution=image_resolution,
    )
    print(f"Saved ParaView state: {state_path}")
    print(f"Saved ParaView screenshot: {image_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--re", choices=("all", "50", "100", "150"), default="all")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--component", choices=("both", "pressure", "viscous"), default="both")
    parser.add_argument("--simulation-state", type=Path, default=None,
                        help="Generated particle-state Python script to show above the force charts")
    args = parser.parse_args()
    if args.n is not None and args.n <= 0:
        parser.error("--n must be a positive L/dp resolution")
    if args.simulation_state is not None and args.re == "all":
        parser.error("--simulation-state requires a single --re matching the simulation")
    return args


def main():
    args = parse_args()
    reynolds_numbers = (50, 100, 150) if args.re == "all" else (int(args.re),)
    for index, reynolds in enumerate(reynolds_numbers):
        if index:
            ResetSession()
        make_force_plot(reynolds, args)


if __name__ == "__main__":
    main()
