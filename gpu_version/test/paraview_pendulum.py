"""Run with pvpython against a generated Pendulum3D default state file."""
import runpy
import sys
from pathlib import Path
from paraview.simple import GetAnimationScene, GetDisplayProperties, RenderAllViews, SaveScreenshot, GetLayout

state = runpy.run_path(sys.argv[1])
overview, detail = state['renderView1'], state['detailView']
scene = GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()
times = list(state['Simulation_vtkhdf'].TimestepValues)
assert len(times) >= 2
assert state['FixedGeometry'].GetDataInformation().GetNumberOfCells() == 5
assert state['Pivot'].Center[:] == [5,5,5]
snapshots = Path(sys.argv[2]) if len(sys.argv) > 2 else None
if snapshots:
    snapshots.mkdir(parents=True, exist_ok=True)
for vector, arrows, _ in state['native_force_arrows']:
    view = overview if vector == 'TotalForce' else detail
    assert GetDisplayProperties(arrows,view).Visibility == (0 if vector == 'TotalForce' else 1)
for time in (times[0],times[len(times)//2],times[-1]):
    scene.AnimationTime = time
    state['Pendulum'].UpdatePipeline(time)
    RenderAllViews()
    bounds = state['Pendulum'].GetDataInformation().GetBounds()
    center = [(bounds[2*d]+bounds[2*d+1])/2 for d in range(3)]
    assert state['Pendulum'].GetDataInformation().GetNumberOfPoints() == 196
    if time != times[0]:
        assert max(abs(a-b) for a,b in zip(center,detail.CameraFocalPoint)) < 1e-6
    assert overview.CameraViewUp[2] > .99 and detail.CameraViewUp[2] > .99
    if snapshots:
        SaveScreenshot(str(snapshots / f'pendulum_{time:.2f}.png'), GetLayout(overview), ImageResolution=[1600,900])
print('Pendulum default state passed: reference meshes, force visibility, Z-up camera and tracking.')
