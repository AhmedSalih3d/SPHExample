"""Integration check with existing output: pvpython --force-offscreen-rendering this.py launcher.py."""
import math
import builtins
import runpy
import sys
from bisect import bisect_right
from pathlib import Path

from paraview import servermanager
from paraview.simple import (
    FindSource, GetAnimationScene, GetLayout, GetRepresentation, GetViews,
    LoadState, RenderAllViews, ResetSession, SaveScreenshot, SaveState,
)


def interpolate(rows, time):
    i = builtins.max(0, builtins.min(bisect_right([row[0] for row in rows], time) - 1, len(rows) - 2))
    fraction = (time - rows[i][0]) / (rows[i + 1][0] - rows[i][0])
    return rows[i][1] + fraction * (rows[i + 1][1] - rows[i][1])


def check(launcher):
    namespace = runpy.run_path(str(launcher))
    args = namespace["args"]
    helpers = runpy.run_path(str(Path(__file__).with_name("paraview_heave.py")))
    history = helpers["read_history"](args.history)
    centers = helpers["read_centers"](args.particles, getattr(args, "planar", False))
    planar = getattr(args, "planar", False)
    SaveState(str(args.state))
    ResetSession()
    LoadState(str(args.state))
    view = next(v for v in GetViews() if v.GetXMLName() == "RenderView")
    chart = next(v for v in GetViews() if v.GetXMLName() == "XYChartView")
    layout = GetLayout(view)
    layout.SetSize(1400, 1000)
    assert view.AxesGrid.Visibility == 1
    assert GetRepresentation(FindSource("Grid distance labels"), view).Visibility == 1
    assert GetRepresentation(FindSource("Heave ruler (mm, equilibrium zero)"), view).Visibility == 1
    assert GetRepresentation(FindSource("Motion grid (0.1 m)"), view).Visibility == 1
    assert GetRepresentation(FindSource("Fluid particles" if planar else "Rear half of fluid"), view).Visibility == 1
    cursor = FindSource("Animation playhead")
    marker = FindSource("Moving center of mass")
    display = GetRepresentation(cursor, chart)
    assert display.Visibility == 1
    assert chart.BottomAxisRangeMinimum < history[0][0]
    assert chart.BottomAxisRangeMaximum > history[-1][0]
    assert "Current heave" in display.SeriesVisibility
    assert list(display.SeriesMarkerStyle)[-1] == "4"
    assert list(display.SeriesLineStyle)[1] == "2"
    scene = GetAnimationScene()
    for time in (history[0][0], 0.35, 1.101, 0.7, history[-1][0]):
        if time > history[-1][0]:
            continue
        scene.AnimationTime = time
        cursor.UpdatePipeline(time)
        marker.UpdatePipeline(time)
        table = servermanager.Fetch(cursor)
        point = servermanager.Fetch(marker).GetPoint(0)
        expected = interpolate(history, time)
        pointer = FindSource("Current height on heave ruler")
        pointer.UpdatePipeline(time)
        tip = servermanager.Fetch(pointer).GetPoint(1)
        vertical = 1 if planar else 2
        assert math.isclose(tip[vertical], point[vertical], abs_tol=1e-10)
        readout = FindSource("Live heave measurement")
        readout.UpdatePipeline(time)
        measurement = servermanager.Fetch(readout).GetColumnByName("Text").GetValue(0)
        assert f"Heave: {expected:+.1f} mm" in measurement
        assert math.isclose(table.GetColumnByName("time").GetValue(0), time, abs_tol=1e-10)
        assert math.isclose(table.GetColumnByName("Current heave").GetValue(0), expected, abs_tol=1e-8)
        for axis in range(3):
            values = [(t, p[axis]) for t, p in centers]
            assert math.isclose(point[axis], interpolate(values, time), abs_tol=1e-10)
        water = FindSource("Fluid particles" if planar else "Rear half of fluid")
        water.UpdatePipeline(time)
        assert water.GetDataInformation().GetNumberOfPoints() > 0
        clock = FindSource("Simulation time")
        clock.UpdatePipeline(time)
        assert servermanager.Fetch(clock).GetColumnByName("Text").GetValue(0) == f"t = {time:.2f} s"
    scene.AnimationTime = builtins.min(0.35, history[-1][0])
    RenderAllViews()
    SaveScreenshot(str(args.state.with_name(args.state.stem + "_Motion.png")), layout)
    SaveState(str(args.state))
    print("PASS: fluid and labelled grid visible; time cursor, physical COM, ruler pointer and numeric heave synchronize after state reload, including fractional time and reverse seeking.")


if __name__ == "__main__":
    check(Path(sys.argv[1]).resolve())
