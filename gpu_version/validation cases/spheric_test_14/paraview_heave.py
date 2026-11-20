"""Build a synchronized ParaView particle view and experimental heave chart."""
import argparse
import csv
import math
import re
from pathlib import Path
from textwrap import dedent

from paraview.simple import (
    AnnotateTimeFilter,
    Clip,
    ConvertToPointCloud,
    CreateLayout,
    CreateView,
    CSVReader,
    ExtractBlock,
    GetAnimationScene,
    ProgrammableSource,
    RenderAllViews,
    SaveScreenshot,
    SaveState,
    SetActiveSource,
    SetActiveView,
    Show,
    Text,
    Threshold,
    VTKHDFReader,
    _DisableFirstRenderCameraReset,
)

FLUID = (0.16, 0.54, 0.81)
FLOAT = (0.88, 0.35, 0.12)
INK = (0.12, 0.17, 0.23)


def read_history(path, experimental=False):
    with path.open(encoding="utf-8", newline="") as stream:
        if experimental:
            rows = [tuple(map(float, line.split())) for line in stream if line.strip()]
        else:
            rows = [(float(row["time"]), float(row["heave_mm"])) for row in csv.DictReader(stream)]
    if (len(rows) < 2 or any(len(row) != 2 or not all(map(math.isfinite, row)) for row in rows)
            or any(b[0] <= a[0] for a, b in zip(rows, rows[1:]))):
        raise ValueError(f"Expected finite heave samples with strictly increasing times in {path}")
    return rows


def chart_series(rows, name, state, chart, color, style):
    path = state.with_name(f"{state.stem}_{name}.csv")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time", "heave_mm"))
        writer.writerows(rows)
    source = CSVReader(registrationName=name, FileName=[path.as_posix()])
    display = Show(source, chart, "XYChartRepresentation")
    display.UseIndexForXAxis = 0
    display.XArrayName = "time"
    display.SeriesVisibility = ["heave_mm"]
    display.SeriesLabel = ["heave_mm", name]
    display.SeriesColor = ["heave_mm", *map(str, color)]
    display.SeriesLineStyle = ["heave_mm", str(style)]
    display.SeriesLineThickness = ["heave_mm", "2"]


def chart_playhead(chart, times, lower, upper):
    source = ProgrammableSource(registrationName="Animation playhead")
    source.OutputDataSetType = "vtkTable"
    source.ScriptRequestInformation = dedent(f"""\
        from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline as pipeline
        times = {times!r}
        info = self.GetExecutive().GetOutputInformation(0)
        info.Set(pipeline.TIME_STEPS(), times, len(times))
        info.Set(pipeline.TIME_RANGE(), (times[0], times[-1]), 2)
        """)
    source.Script = dedent(f"""\
        from vtkmodules.vtkCommonCore import vtkDoubleArray
        from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline as pipeline
        info = self.GetExecutive().GetOutputInformation(0)
        time = info.Get(pipeline.UPDATE_TIME_STEP()) if info.Has(pipeline.UPDATE_TIME_STEP()) else {times[0]!r}
        table = self.GetOutputDataObject(0)
        table.Initialize()
        for name, values in (("time", (time, time)), ("Playhead", ({lower!r}, {upper!r}))):
            column = vtkDoubleArray()
            column.SetName(name)
            for value in values:
                column.InsertNextValue(value)
            table.AddColumn(column)
        """)
    display = Show(source, chart, "XYChartRepresentation")
    display.UseIndexForXAxis = 0
    display.XArrayName = "time"
    display.SeriesVisibility = ["Playhead"]
    display.SeriesLabel = ["Playhead", "Current time"]
    display.SeriesColor = ["Playhead", "0.30", "0.44", "0.55"]
    display.SeriesLineStyle = ["Playhead", "3"]
    display.SeriesLineThickness = ["Playhead", "2"]


def particle_type(particles, kind, name):
    selected = Threshold(registrationName=name, Input=particles)
    selected.Scalars = ["POINTS", "Type"]
    selected.LowerThreshold = kind
    selected.UpperThreshold = kind
    selected.ThresholdMethod = "Between"
    return selected


def show_particles(source, view, color, radius):
    display = Show(source, view, "GeometryRepresentation")
    display.Representation = "Point Gaussian"
    display.ColorArrayName = ["POINTS", ""]
    display.DiffuseColor = color
    display.AmbientColor = color
    display.GaussianRadius = radius
    display.ShaderPreset = "Sphere"


def overlay(source, view, position, size):
    display = Show(source, view, "TextSourceRepresentation")
    display.WindowLocation = position
    display.Color = INK
    display.FontSize = size


def build_view(args):
    simulation = read_history(args.history)
    reference = read_history(args.reference, experimental=True)
    metadata = args.history.with_suffix(".toml")
    spacing = 0.025
    if metadata.is_file():
        saved_spacing = re.search(r"^dx\s*=\s*(\S+)", metadata.read_text(encoding="utf-8"), re.MULTILINE)
        if saved_spacing:
            spacing = float(saved_spacing[1])

    _DisableFirstRenderCameraReset()
    layout = CreateLayout(name="Free heave comparison")
    view = CreateView("RenderView")
    layout.AssignView(0, view)
    if args.layout == "stacked":
        layout.SplitVertical(0, 0.65)
    else:
        layout.SplitHorizontal(0, 0.53)
    chart = CreateView("XYChartView")
    layout.AssignView(2, chart)

    reader = VTKHDFReader(registrationName="Test 14 particle history", FileName=[args.particles.as_posix()])
    reader.PointArrayStatus = ["Type"]
    particles = ExtractBlock(registrationName="Particles", Input=reader)
    particles.Selectors = ["/Root/Particles"]
    # Point exports have no cells. Give each point a vertex for Threshold.
    vertices = ConvertToPointCloud(registrationName="Particle vertices", Input=particles,
                                   CellGenerationMode="Vertex cells")
    water = particle_type(vertices, 1, "Fluid particles")
    rear_water = Clip(registrationName="Rear half of fluid", Input=water)
    rear_water.ClipType = "Plane"
    rear_water.ClipType.Origin = [0.0, 0.04, 0.0]
    rear_water.ClipType.Normal = [0.0, 1.0, 0.0]
    rear_water.Invert = 0
    float_particles = particle_type(vertices, 4, "Floating body")
    show_particles(rear_water, view, FLUID, 0.43 * spacing)
    show_particles(float_particles, view, FLOAT, 0.46 * spacing)

    title = Text(registrationName="Case title")
    title.Text = args.title or "SPHERIC Test 14"
    overlay(title, view, "Upper Left Corner", 17)
    caption = Text(registrationName="Particle colors")
    caption.Text = "Blue: fluid section   Orange: float"
    overlay(caption, view, "Lower Left Corner", 12)
    clock = AnnotateTimeFilter(registrationName="Simulation time", Input=reader)
    clock.Format = "t = {time:.2f} s"
    overlay(clock, view, "Upper Right Corner", 14)

    view.UseColorPaletteForBackground = 0
    view.Background = [0.96, 0.97, 0.98]
    view.OrientationAxesVisibility = 0
    view.CameraParallelProjection = 1
    view.CameraPosition = [0.7, -4.0, 0.68]
    view.CameraFocalPoint = [0.0, 0.14, -0.29]
    view.CameraViewUp = [0.0, 0.0, 1.0]
    view.CameraParallelScale = 0.82
    view.CenterOfRotation = [0.0, 0.0, -0.13]

    chart.ChartTitle = "Free heave"
    chart.BottomAxisTitle = "Time since release (s)"
    chart.LeftAxisTitle = "Heave (mm)"
    chart.BottomAxisUseCustomRange = 1
    chart.BottomAxisRangeMinimum = simulation[0][0]
    chart.BottomAxisRangeMaximum = simulation[-1][0]
    values = [heave for time, heave in simulation + reference if simulation[0][0] <= time <= simulation[-1][0]]
    margin = max(1, 0.06 * (max(values) - min(values)))
    chart.LeftAxisUseCustomRange = 1
    chart.LeftAxisRangeMinimum = min(values) - margin
    chart.LeftAxisRangeMaximum = max(values) + margin
    chart.ShowBottomAxisGrid = 1
    chart.ShowLeftAxisGrid = 1
    chart.ShowLegend = 1
    chart.HideTimeMarker = 1
    chart.ChartTitleFontSize = 17
    chart.BottomAxisTitleFontSize = 13
    chart.LeftAxisTitleFontSize = 13
    chart.LegendFontSize = 12
    chart_series(simulation, "SPHExampleGPU", args.state, chart, FLOAT, 1)
    chart_series(reference, "SPHERIC experiment", args.state, chart, INK, 2)
    chart_playhead(chart, [row[0] for row in simulation], chart.LeftAxisRangeMinimum, chart.LeftAxisRangeMaximum)

    scene = GetAnimationScene()
    scene.UpdateAnimationUsingDataTimeSteps()
    scene.AnimationTime = simulation[0][0]
    for source in (rear_water, float_particles):
        source.UpdatePipeline(scene.AnimationTime)
        if source.GetDataInformation().GetNumberOfPoints() == 0:
            raise ValueError(f"No {source.GetXMLLabel()} points are visible in {args.particles}")
    SetActiveView(view)
    SetActiveSource(rear_water)
    RenderAllViews()
    SaveState(args.state.as_posix())
    if args.screenshot:
        # ParaView captures the layout at its current on-screen size.
        SaveScreenshot(args.screenshot.as_posix(), layout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("particles", type=Path)
    parser.add_argument("history", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--title")
    parser.add_argument("--layout", choices=("side-by-side", "stacked"), default="side-by-side")
    args = parser.parse_args()
    for name in ("particles", "history", "reference", "state", "screenshot"):
        path = getattr(args, name)
        if path is not None:
            setattr(args, name, path.resolve())
    args.state.parent.mkdir(parents=True, exist_ok=True)
    build_view(args)
