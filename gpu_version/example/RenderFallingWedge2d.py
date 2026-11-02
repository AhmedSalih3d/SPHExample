"""Render the falling wedge and both nested refinement zones from one VTKHDF.

pvpython --dr RenderFallingWedge2d.py FallingWedge2D.vtkhdf impact.png
Add --color resolution, --overview, --time 0.01 or --all-frames as needed.
"""
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import vtk
from RenderRefinement2d import blocks, particle_actor, geometry_actor


def render(reader, time, destination, args):
    reader.UpdateTimeStep(time)
    datasets = blocks(reader.GetOutputDataObject(0))
    particles, geometry = datasets["Particles"], datasets["Geometry"]
    fluid = vtk.vtkThresholdPoints()
    fluid.SetInputData(particles)
    fluid.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "Type")
    fluid.SetLowerThreshold(1)
    fluid.SetUpperThreshold(1)
    fluid.SetThresholdFunction(vtk.vtkThresholdPoints.THRESHOLD_BETWEEN)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(fluid.GetOutputPort())
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray("Resolution" if args.color == "resolution" else "Pressure")
    lookup = vtk.vtkLookupTable()
    lookup.SetNumberOfTableValues(3 if args.color == "resolution" else 256)
    lookup.SetTableRange(0, 2 if args.color == "resolution" else args.pressure_max)
    lookup.SetHueRange(0.66, 0.0)
    lookup.Build()
    mapper.SetLookupTable(lookup)
    mapper.UseLookupTableScalarRangeOn()
    mapper.SetScalarRange(0, 2 if args.color == "resolution" else args.pressure_max)
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(3)
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.98, 0.985, 1)
    renderer.SetViewport(0, 0, 0.86, 1)
    renderer.AddActor(actor)
    renderer.AddActor(particle_actor(particles, 2, (0.3, 0.35, 0.4)))
    renderer.AddActor(particle_actor(particles, 4, (0.95, 0.6, 0.12)))
    for region, color, wire in [(1, (0.3, 0.35, 0.4), False), (2, (0.95, 0.6, 0.12), True),
                                (3, (0.65, 0.2, 0.8), True), (4, (0.1, 0.65, 0.4), True)]:
        renderer.AddActor(geometry_actor(geometry, region, color, wire))
    title = vtk.vtkTextActor()
    title.SetInput(f"Falling wedge | three resolution levels | t = {time:.5f} s")
    title.SetDisplayPosition(22, 22)
    title.GetTextProperty().SetFontSize(22)
    title.GetTextProperty().SetColor(0.1, 0.15, 0.2)
    renderer.AddViewProp(title)
    bar = vtk.vtkScalarBarActor()
    bar.SetLookupTable(lookup)
    bar.SetTitle("Resolution level" if args.color == "resolution" else "Pressure [Pa]")
    bar.SetNumberOfLabels(3 if args.color == "resolution" else 5)
    bar.SetPosition(0.08, 0.22)
    bar.SetWidth(0.8)
    bar.SetHeight(0.6)
    bar.SetLabelFormat("%.0f")
    bar.UnconstrainedFontSizeOn()
    for text in (bar.GetTitleTextProperty(), bar.GetLabelTextProperty()):
        text.SetColor(0.1, 0.15, 0.2)
        text.SetFontSize(20)
        text.BoldOff()
        text.ItalicOff()
        text.ShadowOff()
    legend = vtk.vtkRenderer()
    legend.SetBackground(0.98, 0.985, 1)
    legend.SetViewport(0.86, 0, 1, 1)
    legend.AddViewProp(bar)
    camera = renderer.GetActiveCamera()
    camera.SetPosition(0, 1.4, 10)
    camera.SetFocalPoint(0, 1.4, 0)
    camera.SetViewUp(0, 1, 0)
    camera.ParallelProjectionOn()
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(1500, 1000)
    window.AddRenderer(renderer)
    window.AddRenderer(legend)
    renderer.ResetCamera()
    if args.overview:
        camera.SetParallelScale(1.2)
    else:
        camera.SetFocalPoint(0, 1.45, 0)
        camera.SetPosition(0, 1.45, 10)
        camera.SetParallelScale(0.57)
    renderer.ResetCameraClippingRange()
    window.Render()
    capture = vtk.vtkWindowToImageFilter()
    capture.SetInput(window)
    capture.Update()
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkPNGWriter()
    writer.SetFileName(str(destination))
    writer.SetInputConnection(capture.GetOutputPort())
    writer.Write()
    window.Finalize()
    print(f"Rendered {particles.GetNumberOfPoints()} particles at t={time}: {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--time", type=float)
    parser.add_argument("--all-frames", action="store_true")
    parser.add_argument("--overview", action="store_true")
    parser.add_argument("--color", choices=("pressure", "resolution"), default="pressure")
    parser.add_argument("--pressure-max", type=float, default=100000)
    args = parser.parse_args()
    reader = vtk.vtkHDFReader()
    reader.SetFileName(str(args.input.resolve()))
    reader.UpdateInformation()
    times = list(reader.GetOutputInformation(0).Get(vtk.vtkStreamingDemandDrivenPipeline.TIME_STEPS()) or [0])
    if args.all_frames:
        for index, time in enumerate(times):
            render(reader, time, args.output / f"frame_{index:06d}.png", args)
    else:
        time = times[-1] if args.time is None else min(times, key=lambda t: abs(t-args.time))
        render(reader, time, args.output, args)


if __name__ == "__main__":
    main()
