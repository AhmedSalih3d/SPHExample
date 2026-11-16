"""Create a ParaView state and screenshot comparing raw SPH drag with SPHERIC FD data.

The state contains side-by-side pressure and viscous drag charts. Each available
SPH resolution in the selected results directory is plotted without filtering.
"""
import argparse
import csv
import math
import re
from pathlib import Path

from paraview.simple import (
    CSVReader,
    CreateLayout,
    CreateView,
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
    return last_time


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
        end_time = read_drag_history(path)
        histories.append((n, path, end_time))
    return histories


def make_chart_view(layout, anchor, reynolds, component, histories, reference_reader, end_time):
    view = CreateView("XYChartView")
    if anchor is None:
        layout.AssignView(0, view)
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


def make_force_plot(reynolds: int, args):
    histories = find_histories(args.results, reynolds, args.n)
    reference_path = CASE / "reference" / f"Force_Re{reynolds:03d}.dat"
    reference_rows = read_reference(reference_path)
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
    layout = CreateLayout(name=f"SPHERIC Test 06 Re{reynolds:03d} force comparison")
    first_view = None
    end_time = max(sample_end for _, _, sample_end in readers)
    for component in components:
        view = make_chart_view(
            layout, first_view, reynolds, component, readers, reference_reader, end_time
        )
        if first_view is None:
            first_view = view

    layout.SetSize(1800 if len(components) == 2 else 1200, 850)
    RenderAllViews()
    state_path = args.output / f"ForceComparison_Re{reynolds:03d}.pvsm"
    image_path = args.output / f"ForceComparison_Re{reynolds:03d}.png"
    SaveState(str(state_path.resolve()))
    SaveScreenshot(
        str(image_path.resolve()),
        layout,
        ImageResolution=[1800, 850] if len(components) == 2 else [1200, 850],
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
    args = parser.parse_args()
    if args.n is not None and args.n <= 0:
        parser.error("--n must be a positive L/dp resolution")
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
