"""Check the generated coupled Pelamis state with pvpython."""
import runpy
import sys
from pathlib import Path
from paraview import servermanager
from paraview.simple import GetAnimationScene, GetDisplayProperties, RenderAllViews, SaveScreenshot, GetLayout

state = runpy.run_path(sys.argv[1])
overview,detail = state['renderView1'],state['detailView']
scene = GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()
times = list(state['Simulation_vtkhdf'].TimestepValues)
snapshots = Path(sys.argv[2]) if len(sys.argv) > 2 else None
if snapshots:
    snapshots.mkdir(parents=True, exist_ok=True)
assert state['Simulation_vtkhdfDisplay'].Visibility == 0
assert state['Beach'].GetDataInformation().GetNumberOfCells() == 2
assert state['SlidingAxis'].Point1[0] == 5.8
for vector,arrows,_ in state['native_force_arrows']:
    assert GetDisplayProperties(arrows,overview).Visibility == 0
    if vector != 'TotalForce':
        assert GetDisplayProperties(arrows,detail).Visibility == 1
for time in (times[0],times[len(times)//2],times[-1]):
    scene.AnimationTime = time
    RenderAllViews()
    state['Pelamis'].UpdatePipeline(time)
    assert state['Pelamis'].GetDataInformation().GetNumberOfPoints() == 43644
    assert state['Pelamis'].GetDataInformation().GetNumberOfCells() == 70168
    assert state['WaterSurface'].GetDataInformation().GetNumberOfCells() > 26000
    assert state['SpeedColors'].RGBPoints[0] == 0 and state['SpeedColors'].RGBPoints[-4] == 1
    assert detail.CameraViewUp[1] > .99
    if time != times[0]:
        bounds = state['Pelamis'].GetDataInformation().GetBounds()
        center = [(bounds[2*d]+bounds[2*d+1])/2 for d in range(3)]
        assert max(abs(a-b) for a,b in zip(center,detail.CameraFocalPoint)) < 1e-6
    state['TimeLabel'].UpdatePipeline(time)
    text = servermanager.Fetch(state['TimeLabel']).GetRowData().GetAbstractArray(0).GetValue(0)
    assert f'{time:.2f}' in text
    if snapshots:
        SaveScreenshot(str(snapshots / f'pelamis_{time:.2f}.png'),GetLayout(overview),ImageResolution=[1600,900])
print('Pelamis default state passed: fluid surface, copied CAD, force arrays, time and planar tracking.')
