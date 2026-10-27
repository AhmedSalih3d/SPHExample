"""Render actual wheel VTKHDF particles using ParaView's bundled VTK Python.

pvpython --dr RenderWheelPeriodic.py WheelPeriodic3D.vtkhdf preview.png --spacing 0.025
Use --all-frames with an empty/new output directory to render an animation.
Surface smoothing is cosmetic and does not change simulation data.
"""

import argparse
import math
from pathlib import Path

import vtk


def particle_group(data, particle_type):
    # Solver output is a point cloud without vertex cells. A cell threshold
    # would silently discard every point, so select the point data directly.
    select = vtk.vtkThresholdPoints()
    select.SetInputData(data)
    select.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "Type")
    if hasattr(select, "SetThresholdFunction"):
        select.SetLowerThreshold(particle_type)
        select.SetUpperThreshold(particle_type)
        select.SetThresholdFunction(vtk.vtkThresholdPoints.THRESHOLD_BETWEEN)
    else:
        select.ThresholdBetween(particle_type, particle_type)
    select.Update()
    return select.GetOutput()


def actor_for(data, spacing, color, smooth=False):
    if smooth:
        bounds = list(data.GetBounds())
        for axis in range(3):
            bounds[2 * axis] -= 2 * spacing
            bounds[2 * axis + 1] += 2 * spacing
        extent = [bounds[2 * i + 1] - bounds[2 * i] for i in range(3)]
        splat = vtk.vtkGaussianSplatter()
        splat.SetInputData(data)
        splat.SetModelBounds(bounds)
        splat.SetSampleDimensions(*[max(12, int(length / (spacing / 2))) for length in extent])
        splat.SetRadius(1.6 * spacing / max(extent))
        splat.SetExponentFactor(-4)
        splat.ScalarWarpingOff()
        splat.NormalWarpingOff()
        splat.SetAccumulationModeToSum()
        contour = vtk.vtkContourFilter()
        contour.SetInputConnection(splat.GetOutputPort())
        contour.SetValue(0, 0.22)
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(contour.GetOutputPort())
        normals.SetFeatureAngle(60)
        source = normals
    else:
        cube = vtk.vtkCubeSource()
        cube.SetXLength(spacing)
        cube.SetYLength(spacing)
        cube.SetZLength(spacing)
        source = vtk.vtkGlyph3D()
        source.SetInputData(data)
        source.SetSourceConnection(cube.GetOutputPort())
        source.ScalingOff()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(source.GetOutputPort())
    mapper.ScalarVisibilityOff()
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    material = actor.GetProperty()
    material.SetColor(*color)
    material.SetInterpolationToPhong()
    material.SetAmbient(0.2)
    material.SetDiffuse(0.75)
    material.SetSpecular(0.55 if smooth else 0.1)
    material.SetSpecularPower(65)
    return actor


def label(renderer, text, x, y, size, color=(0.12, 0.18, 0.23)):
    actor = vtk.vtkTextActor()
    actor.SetInput(text)
    actor.SetPosition(x, y)
    actor.GetTextProperty().SetFontFamilyToArial()
    actor.GetTextProperty().SetFontSize(size)
    actor.GetTextProperty().SetColor(*color)
    renderer.AddViewProp(actor)
    return actor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--spacing", type=float, required=True, help="simulation particle spacing in metres")
    parser.add_argument("--step", type=int, default=-1, help="saved frame index; default: last")
    parser.add_argument("--all-frames", action="store_true", help="output is a new/empty directory of PNG frames")
    args = parser.parse_args()
    if not args.input.is_file() or not math.isfinite(args.spacing) or not args.spacing > 0:
        parser.error("input must exist and spacing must be positive")
    if args.output.exists() and (not args.all_frames or not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("output must be new (or an empty directory for --all-frames)")
    reader = vtk.vtkHDFReader()
    reader.SetFileName(str(args.input.resolve()))
    reader.UpdateInformation()
    count = reader.GetNumberOfSteps()
    step = args.step if args.step >= 0 else count - 1
    if count < 1 or not 0 <= step < count:
        parser.error(f"step must be in [0, {count - 1}]")
    if args.all_frames:
        args.output.mkdir(parents=True, exist_ok=True)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.94, 0.95, 0.96)
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.AddRenderer(renderer)
    window.SetSize(1440, 1000)
    window.SetMultiSamples(4)
    for index in range(count) if args.all_frames else [step]:
        reader.SetStep(index)
        reader.Update()
        data = reader.GetOutput()
        if data.GetNumberOfPoints() == 0 or data.GetPointData().GetArray("Type") is None:
            parser.error("expected a nonempty SPH particle frame with the Type field")
        renderer.RemoveAllViewProps()
        renderer.AddActor(actor_for(particle_group(data, 2), args.spacing, (0.34, 0.24, 0.16)))
        renderer.AddActor(actor_for(particle_group(data, 4), args.spacing, (0.61, 0.27, 0.10), smooth=True))
        renderer.AddActor(actor_for(particle_group(data, 1), args.spacing, (0.055, 0.38, 0.57), smooth=True))
        camera = renderer.GetActiveCamera()
        # Keep the camera fixed across frames, including splashes and droplets.
        bounds = data.GetBounds()
        center = [(bounds[2 * d] + bounds[2 * d + 1]) / 2 for d in range(3)]
        if index == 0 or not args.all_frames:
            length = bounds[1] - bounds[0]
            camera.SetFocalPoint(center[0], center[1], center[2] * 0.8)
            camera.SetPosition(center[0] + 1.35 * length, center[1] - 1.7 * length, center[2] + 1.4 * length)
            camera.SetViewUp(0, 0, 1)
            camera.ParallelProjectionOn()
            camera.SetParallelScale(0.7 * length)
        renderer.ResetCameraClippingRange()
        label(renderer, "MOTOR-DRIVEN PADDLE WHEEL", 55, 925, 31)
        label(renderer, "3D SPH  /  floating body with a prescribed hinge  /  periodic channel", 55, 887, 20)
        label(renderer, f"t = {reader.GetTimeValue():.2f} s     |     particle spacing = {args.spacing:g} m", 55, 64, 22)
        label(renderer, "Illustrative inputs. Surfaces reconstructed from simulated particles.", 55, 30, 17, (0.35, 0.39, 0.43))
        window.Render()
        capture = vtk.vtkWindowToImageFilter()
        capture.SetInput(window)
        capture.ReadFrontBufferOff()
        capture.Update()
        output = args.output / f"wheel-{index:04d}.png" if args.all_frames else args.output
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(str(output.resolve()))
        writer.SetInputConnection(capture.GetOutputPort())
        writer.Write()
        print(f"Rendered frame {index}/{count - 1}: {output}", flush=True)
    window.Finalize()


if __name__ == "__main__":
    main()
