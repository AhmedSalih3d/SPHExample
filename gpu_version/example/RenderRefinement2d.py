"""Render a frame from either body-following example's combined VTKHDF.

pvpython --dr RenderRefinement2d.py case.vtkhdf preview.png [--time 0.1]
Use --all-frames with an output directory for a numbered PNG animation.
Add --spring-detail to inspect the force-coloured coil and its attachment.
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


def spring_lookup(maximum_force):
    lookup = vtk.vtkColorTransferFunction()
    for fraction, color in ((0, (0.267, 0.005, 0.329)), (0.25, (0.230, 0.322, 0.546)),
                            (0.5, (0.128, 0.567, 0.551)), (0.75, (0.369, 0.789, 0.383)),
                            (1, (0.993, 0.906, 0.144))):
        lookup.AddRGBPoint(fraction * maximum_force, *color)
    return lookup


def render(reader, time, destination, *, spring_detail=False, force_max=1.0):
    reader.UpdateTimeStep(time)
    data = blocks(reader.GetOutputDataObject(0))
    particles, geometry = data["Particles"], data["Geometry"]
    if particles.GetNumberOfPoints() == 0 or geometry.GetNumberOfCells() == 0:
        raise RuntimeError(f"Empty particle or geometry block at t={time}")
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.97, 0.98, 1.0)
    spring = data.get("Springs")
    if spring is not None:
        renderer.SetViewport(0, 0, 0.82, 1)
    for kind, color in ((1, (0.1, 0.45, 0.8)), (2, (0.35, 0.39, 0.43)),
                        (3, (0.1, 0.65, 0.4)), (4, (0.95, 0.6, 0.1))):
        renderer.AddActor(particle_actor(particles, kind, color))
    for region, color, wire in ((1, (0.35, 0.39, 0.43), False),
                                (2, (0.95, 0.6, 0.1), True),
                                (3, (0.6, 0.2, 0.7), True)):
        renderer.AddActor(geometry_actor(geometry, region, color, wire))
    legend = None
    if spring is not None:
        lookup = spring_lookup(force_max)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(spring)
        mapper.SetScalarModeToUseCellFieldData()
        mapper.SelectColorArray("ForceMagnitude")
        mapper.SetLookupTable(lookup)
        mapper.SetScalarRange(0, force_max)
        mapper.UseLookupTableScalarRangeOn()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        # The coil is display geometry for a 2D case; keep it visible over points.
        actor.SetPosition(0, 0, 0.04)
        renderer.AddActor(actor)
        bar = vtk.vtkScalarBarActor()
        bar.SetLookupTable(lookup)
        bar.SetTitle("Spring + damper\nforce [N]")
        bar.SetPosition(0.12, 0.23)
        bar.SetWidth(0.75)
        bar.SetHeight(0.55)
        bar.SetNumberOfLabels(5)
        bar.SetLabelFormat("%.3g")
        bar.UnconstrainedFontSizeOn()
        for prop in (bar.GetTitleTextProperty(), bar.GetLabelTextProperty()):
            prop.SetFontSize(19)
            prop.SetColor(0.1, 0.15, 0.2)
            prop.BoldOff()
            prop.ItalicOff()
            prop.ShadowOff()
        legend = vtk.vtkRenderer()
        legend.SetViewport(0.82, 0, 1, 1)
        legend.SetBackground(0.97, 0.98, 1.0)
        legend.AddViewProp(bar)
    else:
        renderer.AddActor(geometry_actor(geometry, 4, (0.85, 0.15, 0.15), False))
    title = vtk.vtkTextActor()
    label = f"Body-following refinement   t = {time:.4f} s"
    if spring is not None:
        force = spring.GetCellData().GetArray("ForceMagnitude").GetValue(0)
        axial = spring.GetCellData().GetArray("AxialForce").GetValue(0)
        state = "tension" if axial > 0 else "compression" if axial < 0 else "unloaded"
        label += f"\nSpring + damper: {force:.3f} N | {state}"
    title.SetInput(label)
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
    window.SetSize(1200, 1000) if spring_detail else window.SetSize(1600, 650)
    window.AddRenderer(renderer)
    if legend is not None:
        window.AddRenderer(legend)
    renderer.ResetCamera()
    bounds = renderer.ComputeVisiblePropBounds()
    aspect = window.GetSize()[0] * (0.82 if spring is not None else 1) / window.GetSize()[1]
    camera.SetParallelScale(0.55 * max(bounds[3] - bounds[2], (bounds[1] - bounds[0]) / aspect))
    if spring_detail:
        if spring is None:
            raise RuntimeError("This file has no Springs block; regenerate its spring visualization first")
        anchor = spring.GetCellData().GetArray("Anchor").GetTuple3(0)
        attachment = spring.GetCellData().GetArray("Attachment").GetTuple3(0)
        center = ((anchor[0]+attachment[0])/2, (anchor[1]+attachment[1])/2+0.12)
        camera.SetFocalPoint(*center, 0)
        camera.SetPosition(*center, 10)
        camera.SetParallelScale(max(0.36, 0.8*abs(attachment[1]-anchor[1])))
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
    parser.add_argument("--spring-detail", action="store_true")
    parser.add_argument("--force-max", type=float, help="Fixed spring force scale maximum in N")
    args = parser.parse_args()
    reader = vtk.vtkHDFReader()
    reader.SetFileName(str(args.input.resolve()))
    reader.UpdateInformation()
    info = reader.GetOutputInformation(0)
    times = list(info.Get(vtk.vtkStreamingDemandDrivenPipeline.TIME_STEPS()) or [0.0])
    force_max = args.force_max
    if force_max is not None and (not 0 < force_max < float("inf")):
        parser.error("--force-max must be positive and finite")
    if force_max is None:
        force_max = 0.0
        # One scale across the complete animation makes force colours comparable.
        for time in times:
            reader.UpdateTimeStep(time)
            spring = blocks(reader.GetOutputDataObject(0)).get("Springs")
            if spring is None:
                break
            force_max = max(force_max, spring.GetCellData().GetArray("ForceMagnitude").GetRange()[1])
        force_max = max(force_max, 1.0)
    if args.all_frames:
        for index, time in enumerate(times):
            render(reader, time, args.output / f"frame_{index:06d}.png", spring_detail=args.spring_detail, force_max=force_max)
    else:
        time = times[-1] if args.time is None else min(times, key=lambda value: abs(value - args.time))
        render(reader, time, args.output, spring_detail=args.spring_detail, force_max=force_max)


if __name__ == "__main__":
    main()
