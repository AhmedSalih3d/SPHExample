from pathlib import Path
from tempfile import TemporaryDirectory

from paraview.simple import (
    AssignViewToLayout, CSVReader, CreateLayout, CreateView, RenderAllViews,
    SaveScreenshot, SaveState, SetActiveView, Show,
)

def add_circulation_chart(directory, layout, particle_view, oscillation, boundary):
    run = Path(directory)
    location = layout.SplitViewVertical(particle_view, 0.55)
    view = CreateView("XYChartView")
    AssignViewToLayout(view=view, layout=layout, hint=location)
    view.ChartTitle = f"SPHERIC Test 04 - Oscillation {oscillation} - Central-crest vortex"
    view.BottomAxisTitle = "Vortex life-cycle phase (degrees)"
    view.LeftAxisTitle = "Vortex circulation magnitude (m²/s)"


    def plot_table(filename, columns, chart=view, x_axis="phase_deg"):
        reader = CSVReader(FileName=[str(run / filename)])
        reader.UpdatePipeline()
        display = Show(reader, chart)
        display.UseIndexForXAxis = 0
        display.XArrayName = x_axis
        display.SeriesVisibility = columns
        return display


    fine, coarse = "gamma_grid_2p5mm_m2_s", "gamma_grid_5mm_m2_s"
    simulation = plot_table("circulation.csv", [fine, coarse])
    simulation.SeriesLabel = [fine, f"{boundary}, 2.5 mm analysis grid", coarse, f"{boundary}, 5 mm analysis grid"]
    simulation.SeriesColor = [fine, "0.05", "0.37", "0.65", coarse, "0.8", "0.31", "0.06"]
    simulation.SeriesLineStyle = [fine, "1", coarse, "2"]
    simulation.SeriesLineThickness = [fine, "2", coarse, "2"]

    gamma = "gamma_m2_s"
    reference = plot_table("circulation_reference.csv", [gamma])
    reference.SeriesLabel = [gamma, "SPHERIC Fig. 2, digitized"]
    reference.SeriesColor = [gamma, "0.12", "0.12", "0.12"]
    reference.SeriesLineStyle = [gamma, "0"]
    reference.SeriesMarkerStyle = [gamma, "4"]
    reference.SeriesMarkerSize = [gamma, "9"]

    moments = CreateLayout(name="Vortex centre and radius")
    for i, (quantity, title) in enumerate((("x", "Horizontal centre from crest"),
                                           ("y", "Vertical centre from crest"),
                                           ("radius", "Characteristic radius"))):
        chart = CreateView("XYChartView")
        if i:
            cell = moments.SplitViewVertical(previous, 0.5 if i == 2 else 1/3)
        else:
            cell = 0
        AssignViewToLayout(view=chart, layout=moments, hint=cell)
        previous = chart
        chart.ChartTitle = title
        chart.LeftAxisTitle = title + " (m)"
        chart.BottomAxisTitle = "Vortex life-cycle phase (degrees)"
        columns = [f"{quantity}_{grid}_m" for grid in ("2p5mm", "5mm")]
        curves = plot_table("circulation.csv", columns, chart)
        curves.SeriesLabel = [columns[0], "2.5 mm grid", columns[1], "5 mm grid"]
        curves.SeriesColor = [columns[0], "0.05", "0.37", "0.65", columns[1], "0.8", "0.31", "0.06"]
        column = quantity + "_m"
        points = plot_table("circulation_reference.csv", [column], chart, "landmark_phase_deg")
        points.SeriesLabel = [column, "Earnshaw, approximate landmarks"]
        points.SeriesLineStyle = [column, "0"]
        points.SeriesMarkerStyle = [column, "4"]
        points.SeriesColor = [column, "0.12", "0.12", "0.12"]
        points.SeriesMarkerSize = [column, "9"]
    moments.SetSize(1200, 1000)
    layout.SetSize(1400, 1000)
    RenderAllViews()
    SetActiveView(particle_view)
    SaveState(str(run / "circulation_paraview.pvsm"))
    with TemporaryDirectory(dir=run) as scratch:
        image = Path(scratch) / "circulation_paraview.png"
        SaveScreenshot(str(image), view, ImageResolution=[1280, 720])
        image.replace(run / image.name)
