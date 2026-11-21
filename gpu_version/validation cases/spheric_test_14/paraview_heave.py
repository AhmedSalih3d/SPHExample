"""Build a synchronized ParaView particle view and experimental heave chart."""
import argparse
import csv
import math
import re
from pathlib import Path
from textwrap import dedent

from paraview.simple import (
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
CURSOR = (1.0, 0.5, 0.0)


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


def temporal_source(name, times, output_type):
    source = ProgrammableSource(registrationName=name)
    source.OutputDataSetType = output_type
    source.ScriptRequestInformation = dedent(f"""\
        from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline as pipeline
        times = {times!r}
        info = self.GetExecutive().GetOutputInformation(0)
        info.Set(pipeline.TIME_STEPS(), times, len(times))
        info.Set(pipeline.TIME_RANGE(), (times[0], times[-1]), 2)
        """)
    return source


def interpolate_script(times, values):
    return dedent(f"""\
        import builtins
        from bisect import bisect_right
        from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline as pipeline
        times, samples = {times!r}, {values!r}
        info = self.GetExecutive().GetOutputInformation(0)
        time = info.Get(pipeline.UPDATE_TIME_STEP()) if info.Has(pipeline.UPDATE_TIME_STEP()) else times[0]
        index = builtins.max(0, builtins.min(bisect_right(times, time) - 1, len(times) - 2))
        fraction = builtins.max(0., builtins.min(1., (time - times[index]) / (times[index + 1] - times[index])))
        value = [a + fraction * (b - a) for a, b in zip(samples[index], samples[index + 1])]
        """)


def chart_playhead(chart, rows, lower, upper):
    times = [row[0] for row in rows]
    source = temporal_source("Animation playhead", times, "vtkTable")
    source.Script = interpolate_script(times, [[row[1]] for row in rows]) + dedent(f"""\
        from vtkmodules.vtkCommonCore import vtkDoubleArray
        table = self.GetOutputDataObject(0)
        table.Initialize()
        for name, values in (("time", (time, time)), ("Playhead", ({lower!r}, {upper!r})),
                             ("Current heave", (value[0], value[0]))):
            column = vtkDoubleArray()
            column.SetName(name)
            for item in values:
                column.InsertNextValue(item)
            table.AddColumn(column)
        """)
    display = Show(source, chart, "XYChartRepresentation")
    display.UseIndexForXAxis = 0
    display.XArrayName = "time"
    display.SeriesVisibility = ["Playhead", "Current heave"]
    display.SeriesLabel = ["Playhead", "Current time", "Current heave", "Current SPH heave"]
    display.SeriesColor = ["Playhead", *map(str, CURSOR), "Current heave", *map(str, CURSOR)]
    display.SeriesLineStyle = ["Playhead", "2", "Current heave", "0"]
    display.SeriesLineThickness = ["Playhead", "2"]
    display.SeriesMarkerStyle = ["Playhead", "0", "Current heave", "4"]
    display.SeriesMarkerSize = ["Playhead", "0", "Current heave", "11"]
    return source


def read_centers(particles, planar):
    path = particles.with_name(particles.stem + "_Floating.csv")
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len({row["Body"] for row in rows}) != 1:
        raise ValueError("The heave view requires exactly one floating body")
    centers = []
    for row in rows:
        center = [float(row[f"Center:{axis}"]) for axis in range(2 if planar else 3)]
        if planar:
            center.append(0.)
        centers.append((float(row["Time"]), center))
    if (len(centers) < 2 or any(not all(map(math.isfinite, [t, *p])) for t, p in centers)
            or any(b[0] <= a[0] for a, b in zip(centers, centers[1:]))):
        raise ValueError(f"Invalid center-of-mass history in {path}")
    return centers


def motion_point(view, centers, planar):
    source = temporal_source("Moving center of mass", [row[0] for row in centers], "vtkPolyData")
    source.Script = interpolate_script([row[0] for row in centers], [row[1] for row in centers]) + dedent("""\
        from vtkmodules.vtkCommonCore import vtkPoints
        from vtkmodules.vtkCommonDataModel import vtkCellArray
        output = self.GetOutputDataObject(0)
        output.Initialize()
        points = vtkPoints()
        points.SetDataTypeToDouble()
        points.InsertNextPoint(*value)
        vertices = vtkCellArray()
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(0)
        output.SetPoints(points)
        output.SetVerts(vertices)
        """)
    display = show_particles(source, view, CURSOR, 0.016)
    display.ShaderPreset = "Plain circle"
    # Display offsets keep the marker just in front of the planar/cutaway face.
    # The point data themselves retain the physical center-of-mass coordinates.
    display.Translation = [0., 0., 0.025] if planar else [0., -0.015, 0.]
    return source


def scene_references(view, centers, planar):
    def coordinate(x, z):
        return (x, z, 0.012) if planar else (x, -0.008, z)
    grid = ProgrammableSource(registrationName="Motion grid (0.1 m)")
    grid.OutputDataSetType = "vtkPolyData"
    lines = [(coordinate(i / 10, -0.8), coordinate(i / 10, 0.3)) for i in range(-12, 13)]
    lines += [(coordinate(-1.2, i / 10), coordinate(1.2, i / 10)) for i in range(-8, 4)]
    grid.Script = line_script(lines)
    display = Show(grid, view, "GeometryRepresentation")
    display.ColorArrayName = ["POINTS", ""]
    display.DiffuseColor = [0.27, 0.37, 0.43]
    display.Opacity = 0.20
    display.LineWidth = 1.
    vertical = 1 if planar else 2
    guide = ProgrammableSource(registrationName="Center-of-mass motion guide")
    guide.OutputDataSetType = "vtkPolyData"
    low = min(p[vertical] for _, p in centers) - 0.035
    high = max(p[vertical] for _, p in centers) + 0.035
    guide.Script = line_script([(coordinate(centers[0][1][0], low), coordinate(centers[0][1][0], high))])
    display = Show(guide, view, "GeometryRepresentation")
    display.ColorArrayName = ["POINTS", ""]
    display.DiffuseColor = INK
    display.LineWidth = 2.
    view.AxesGrid.Visibility = 1
    view.AxesGrid.UseCustomBounds = 1
    view.AxesGrid.CustomBounds = [-1.2, 1.2, -0.8, 0.3, 0., 0.] if planar else [-1.2, 1.2, 0., 0.4, -0.8, 0.3]
    view.AxesGrid.XTitle = "x (m)"
    view.AxesGrid.YTitle = "z (m)" if planar else ""
    view.AxesGrid.ZTitle = "" if planar else "z (m)"
    view.AxesGrid.AxesToLabel = 3 if planar else 5


def line_script(lines):
    return dedent(f"""\
        from vtkmodules.vtkCommonCore import vtkPoints
        from vtkmodules.vtkCommonDataModel import vtkCellArray
        output = self.GetOutputDataObject(0)
        output.Initialize()
        points, cells = vtkPoints(), vtkCellArray()
        for a, b in {lines!r}:
            first = points.InsertNextPoint(*a)
            last = points.InsertNextPoint(*b)
            cells.InsertNextCell(2)
            cells.InsertCellPoint(first)
            cells.InsertCellPoint(last)
        output.SetPoints(points)
        output.SetLines(cells)
        """)


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
    display.Ambient = 0.65
    display.Diffuse = 0.35
    return display


def overlay(source, view, position, size):
    display = Show(source, view, "TextSourceRepresentation")
    display.WindowLocation = position
    display.Color = INK
    display.FontSize = size


def build_view(args):
    simulation = read_history(args.history)
    reference = read_history(args.reference, experimental=True)
    planar = getattr(args, "planar", False)
    centers = read_centers(args.particles, planar)
    vertical = 1 if planar else 2
    if len(centers) != len(simulation) or any(abs(a[0] - b[0]) > 1e-8 for a, b in zip(centers, simulation)):
        raise ValueError("Center-of-mass and heave histories must have matching time samples")
    datum = centers[0][1][vertical] - simulation[0][1] / 1000
    if any(abs(p[vertical] - datum - heave / 1000) > 1e-8 for (_, p), (_, heave) in zip(centers, simulation)):
        raise ValueError("Center-of-mass coordinates disagree with the heave history")
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
    visible_water = water
    if not planar:
        visible_water = Clip(registrationName="Rear half of fluid", Input=water)
        visible_water.ClipType = "Plane"
        visible_water.ClipType.Origin = [0.0, 0.04, 0.0]
        visible_water.ClipType.Normal = [0.0, 1.0, 0.0]
        visible_water.Invert = 0
    float_particles = particle_type(vertices, 4, "Floating body")
    visible_float = float_particles
    if not planar:
        visible_float = Clip(registrationName="Rear half of float", Input=float_particles)
        visible_float.ClipType = "Plane"
        visible_float.ClipType.Origin = [0., 0., 0.]
        visible_float.ClipType.Normal = [0., 1., 0.]
        visible_float.Invert = 0
    water_display = show_particles(visible_water, view, FLUID, 0.60 * spacing)
    water_display.ShaderPreset = "Plain circle"
    show_particles(visible_float, view, FLOAT, 0.50 * spacing)
    scene_references(view, centers, planar)
    motion_point(view, centers, planar)

    title = Text(registrationName="Case title")
    title.Text = args.title or "SPHERIC Test 14"
    overlay(title, view, "Upper Left Corner", 17)
    caption = Text(registrationName="Particle colors")
    caption.Text = "Blue: fluid   Rust: float   Orange dot: center of mass   Grid: 0.1 m"
    overlay(caption, view, "Lower Left Corner", 12)
    clock = temporal_source("Simulation time", [row[0] for row in simulation], "vtkTable")
    clock.Script = dedent("""\
        from vtkmodules.vtkCommonCore import vtkStringArray
        from vtkmodules.vtkCommonExecutionModel import vtkStreamingDemandDrivenPipeline as pipeline
        info = self.GetExecutive().GetOutputInformation(0)
        time = info.Get(pipeline.UPDATE_TIME_STEP()) if info.Has(pipeline.UPDATE_TIME_STEP()) else 0.
        table = self.GetOutputDataObject(0)
        table.Initialize()
        text = vtkStringArray()
        text.SetName("Text")
        text.InsertNextValue(f"t = {time:.2f} s")
        table.AddColumn(text)
        """)
    overlay(clock, view, "Upper Right Corner", 14)

    view.UseColorPaletteForBackground = 0
    view.Background = [0.96, 0.97, 0.98]
    view.OrientationAxesVisibility = 0
    view.CameraParallelProjection = 1
    view.CameraPosition = [0.0, -0.29, 4.0] if planar else [0.7, -4.0, 0.68]
    view.CameraFocalPoint = [0.0, -0.29, 0.0] if planar else [0.0, 0.14, -0.29]
    view.CameraViewUp = [0.0, 1.0, 0.0] if planar else [0.0, 0.0, 1.0]
    view.CameraParallelScale = 0.82
    view.CenterOfRotation = [0.0, -0.13, 0.0] if planar else [0.0, 0.0, -0.13]

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
    chart_playhead(chart, simulation, chart.LeftAxisRangeMinimum, chart.LeftAxisRangeMaximum)

    scene = GetAnimationScene()
    scene.UpdateAnimationUsingDataTimeSteps()
    scene.AnimationTime = simulation[0][0]
    for source in (visible_water, visible_float):
        source.UpdatePipeline(scene.AnimationTime)
        if source.GetDataInformation().GetNumberOfPoints() == 0:
            raise ValueError(f"No {source.GetXMLLabel()} points are visible in {args.particles}")
    SetActiveView(view)
    SetActiveSource(visible_water)
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
    parser.add_argument("--planar", action="store_true")
    args = parser.parse_args()
    for name in ("particles", "history", "reference", "state", "screenshot"):
        path = getattr(args, name)
        if path is not None:
            setattr(args, name, path.resolve())
    args.state.parent.mkdir(parents=True, exist_ok=True)
    build_view(args)
