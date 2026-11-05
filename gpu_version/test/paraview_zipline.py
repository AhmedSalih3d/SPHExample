"""Optional ParaView integration check for a generated ZipLine default state.

pvpython --no-mpi --disable-registry --force-offscreen-rendering \
    test/paraview_zipline.py <ZipLine3D_SingleVTKHDFStateFile.py>
"""
import runpy
import sys
from paraview.simple import GetAnimationScene, GetDisplayProperties, RenderAllViews

state = runpy.run_path(sys.argv[1])
overview, detail = state["renderView1"], state["detailView"]
scene = GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()
times = list(state["Simulation_vtkhdf"].TimestepValues)
assert len(times) >= 2
assert "Blue: gravity" in state["ForceKey"].Text
assert state["detailKey"].Visibility == 1
assert state["CableAxis"].GetDataInformation().GetNumberOfPoints() == 2
assert state["FixedGeometry"].GetDataInformation().GetNumberOfCells() == 5
for vector, arrows, _ in state["native_force_arrows"]:
    if vector == "TotalForce":
        assert GetDisplayProperties(arrows, overview).Visibility == 0
    else:
        assert GetDisplayProperties(arrows, detail).Visibility == 1
for time in (times[0], times[len(times) // 2], times[-1]):
    scene.AnimationTime = time
    state["Ragdoll"].UpdatePipeline(time)
    RenderAllViews()
    bounds = state["Ragdoll"].GetDataInformation().GetBounds()
    center = [(bounds[2*d] + bounds[2*d+1]) / 2 for d in range(3)]
    if time == times[0]:
        # Initial framing includes the gravity-arrow tips as well as the body.
        assert all(bounds[2*d] <= detail.CameraFocalPoint[d] <= bounds[2*d+1] for d in range(3))
    else:
        assert max(abs(a-b) for a,b in zip(center, detail.CameraFocalPoint)) < 1e-6, (time, center, detail.CameraFocalPoint)
    assert state["Ragdoll"].GetDataInformation().GetNumberOfPoints() == 127
    assert overview.CameraViewUp[2] > 0.99 and detail.CameraViewUp[2] > 0.99
print("ZipLine default ParaView state passed: geometry, force visibility, Z-up views and camera tracking.")
