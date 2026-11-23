from pathlib import Path

from paraview.simple import (
    AssignViewToLayout, CSVReader, CreateView, RenderAllViews,
    SaveScreenshot, SaveState, Show,
)

def add_circulation_chart(directory, layout, particle_view, oscillation, boundary):
    run = Path(directory)
    location = layout.SplitViewVertical(particle_view, 0.55)
    view = CreateView("XYChartView")
    AssignViewToLayout(view=view, layout=layout, hint=location)
    view.ChartTitle = f"SPHERIC Test 04 - Oscillation {oscillation} - Pilot comparison"
    view.BottomAxisTitle = "Oscillation phase (degrees)"
    view.LeftAxisTitle = "Vortex circulation magnitude (m²/s)"


    def plot_table(filename, columns):
        reader = CSVReader(FileName=[str(run / filename)])
        reader.UpdatePipeline()
        display = Show(reader, view)
        display.UseIndexForXAxis = 0
        display.XArrayName = "phase_deg"
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

    layout.SetSize(1400, 1000)
    RenderAllViews()
    SaveState(str(run / "circulation_paraview.pvsm"))
    SaveScreenshot(str(run / "circulation_paraview.png"), view, ImageResolution=[1280, 720])
