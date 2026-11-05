"""Run with pvpython against a generated Demolisher3D default state file."""
import runpy
import sys
from pathlib import Path
from paraview import servermanager
from paraview.simple import GetAnimationScene, GetDisplayProperties, RenderAllViews, SaveScreenshot, GetLayout

state = runpy.run_path(sys.argv[1])
overview, detail = state['renderView1'], state['detailView']
scene = GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()
times = list(state['Simulation_vtkhdf'].TimestepValues)
assert len(times) >= 2
assert state['FixedGeometry'].GetDataInformation().GetNumberOfCells() == 5
assert state['Pivot'].Center[:] == [5,5,5.2]
assert state['Simulation_vtkhdfDisplay'].Visibility == 0
assert len(state['BodyColors'].IndexedColors) == 63
snapshots = Path(sys.argv[2]) if len(sys.argv) > 2 else None
if snapshots:
    snapshots.mkdir(parents=True, exist_ok=True)
for vector, arrows, _ in state['native_force_arrows']:
    view = overview if vector == 'TotalForce' else detail
    assert GetDisplayProperties(arrows,view).Visibility == (0 if vector == 'TotalForce' else 1)
for time in (times[0],times[len(times)//2],times[-1]):
    scene.AnimationTime = time
    state['Demolisher'].UpdatePipeline(time)
    RenderAllViews()
    state['TimeLabel'].UpdatePipeline(time)
    annotation = servermanager.Fetch(state['TimeLabel']).GetRowData().GetAbstractArray(0).GetValue(0)
    assert f'{time:.2f}' in annotation
    bounds = state['Demolisher'].GetDataInformation().GetBounds()
    center = [(bounds[2*d]+bounds[2*d+1])/2 for d in range(3)]
    assert state['Demolisher'].GetDataInformation().GetNumberOfPoints() == 878
    if time != times[0]:
        assert max(abs(a-b) for a,b in zip(center,detail.CameraFocalPoint)) < 1e-6
    assert overview.CameraViewUp[2] > .99 and detail.CameraViewUp[2] > .99
    if snapshots:
        SaveScreenshot(str(snapshots / f'demolisher_{time:.2f}.png'), GetLayout(overview), ImageResolution=[1600,900])
print('Demolisher default state passed: reference meshes, force visibility, Z-up camera and tracking.')
