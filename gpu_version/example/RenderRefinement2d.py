"""Render a frame from either body-following example's combined VTKHDF.

pvpython --dr RenderRefinement2d.py case.vtkhdf preview.png [--time 0.1]
Use --all-frames with an output directory for a numbered PNG animation.
"""

import argparse
from pathlib import Path

import vtk


def blocks(data):
    iterator = data.NewIterator()
    iterator.InitTraversal()
    result = {}
    while not iterator.IsDoneWithTraversal():
        metadata = iterator.GetCurrentMetaData()
        name = metadata.Get(vtk.vtkCompositeDataSet.NAME())
        result[name] = iterator.GetCurrentDataObject()
        iterator.GoToNextItem()
    return result


def particle_actor(data, particle_type, color):
    selected = vtk.vtkThresholdPoints()
    selected.SetInputData(data)
    selected.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "Type")
    selected.SetLowerThreshold(particle_type)
    selected.SetUpperThreshold(particle_type)
    selected.SetThresholdFunction(vtk.vtkThresholdPoints.THRESHOLD_BETWEEN)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(selected.GetOutputPort())
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetPointSize(4)
    return actor


def geometry_actor(data, region, color, wireframe=False):
    selected = vtk.vtkThreshold()
    selected.SetInputData(data)
    selected.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "Region")
    selected.SetLowerThreshold(region)
    selected.SetUpperThreshold(region)
    selected.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
    mapper = vtk.vtkDataSetMapper()
    mapper.SetInputConnection(selected.GetOutputPort())
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    if wireframe:
        actor.GetProperty().SetRepresentationToWireframe()
        actor.GetProperty().SetLineWidth(2)
    # Slightly toward the camera so the link/outline stays visible over points.
    actor.SetPosition(0, 0, 0.001)
    return actor


def render(reader, time, destination):
    reader.UpdateTimeStep(time)
    data = blocks(reader.GetOutputDataObject(0))
    particles, geometry = data["Particles"], data["Geometry"]
    if particles.GetNumberOfPoints() == 0 or geometry.GetNumberOfCells() == 0:
        raise RuntimeError(f"Empty particle or geometry block at t={time}")
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.97, 0.98, 1.0)
    for kind, color in ((1, (0.1, 0.45, 0.8)), (2, (0.35, 0.39, 0.43)),
                        (3, (0.1, 0.65, 0.4)), (4, (0.95, 0.6, 0.1))):
        renderer.AddActor(particle_actor(particles, kind, color))
    for region, color, wire in ((1, (0.35, 0.39, 0.43), False),
                                (2, (0.95, 0.6, 0.1), True),
                                (3, (0.6, 0.2, 0.7), True),
                                (4, (0.85, 0.15, 0.15), False)):
        renderer.AddActor(geometry_actor(geometry, region, color, wire))
    title = vtk.vtkTextActor()
    title.SetInput(f"Body-following refinement   t = {time:.4f} s")
    title.SetDisplayPosition(24, 24)
    title.GetTextProperty().SetColor(0.1, 0.15, 0.2)
    title.GetTextProperty().SetFontSize(20)
    renderer.AddViewProp(title)
    camera = renderer.GetActiveCamera()
    camera.SetPosition(0, 0, 10)
    camera.SetFocalPoint(0, 0, 0)
    camera.SetViewUp(0, 1, 0)
    camera.ParallelProjectionOn()
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetSize(1600, 650)
    window.AddRenderer(renderer)
    renderer.ResetCamera()
    bounds = renderer.ComputeVisiblePropBounds()
    aspect = window.GetSize()[0] / window.GetSize()[1]
    camera.SetParallelScale(0.55 * max(bounds[3] - bounds[2], (bounds[1] - bounds[0]) / aspect))
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
    args = parser.parse_args()
    reader = vtk.vtkHDFReader()
    reader.SetFileName(str(args.input.resolve()))
    reader.UpdateInformation()
    info = reader.GetOutputInformation(0)
    times = list(info.Get(vtk.vtkStreamingDemandDrivenPipeline.TIME_STEPS()) or [0.0])
    if args.all_frames:
        for index, time in enumerate(times):
            render(reader, time, args.output / f"frame_{index:06d}.png")
    else:
        time = times[-1] if args.time is None else min(times, key=lambda value: abs(value - args.time))
        render(reader, time, args.output)


if __name__ == "__main__":
    main()
