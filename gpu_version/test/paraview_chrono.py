"""Verify a generated reference-case state and save first/middle/last frames."""
import runpy
import sys
from pathlib import Path
from paraview import servermanager
from paraview.simple import GetAnimationScene, GetDisplayProperties, GetLayout, RenderAllViews, SaveScreenshot

state = runpy.run_path(sys.argv[1])
directory = Path(sys.argv[2])
directory.mkdir(parents=True, exist_ok=True)
scene = GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()
times = list(state['Simulation_vtkhdf'].TimestepValues)
assert len(times) >= 2
assert state['Simulation_vtkhdfDisplay'].Visibility == 0
for vector,arrows,_ in state.get('native_force_arrows',[]):
    if vector != 'TotalForce':
        assert GetDisplayProperties(arrows,state['renderView1']).Visibility == 1
for time in (times[0],times[len(times)//2],times[-1]):
    scene.AnimationTime = time
    RenderAllViews()
    state['Bodies'].UpdatePipeline(time)
    assert state['Bodies'].GetDataInformation().GetNumberOfPoints() > 0
    assert state['Bodies'].GetDataInformation().GetNumberOfCells() > 0
    for extra in ('Coil','Pointer'):
        if extra in state:
            state[extra].UpdatePipeline(time)
            assert state[extra].GetDataInformation().GetNumberOfPoints() > 0
            assert state[extra].GetDataInformation().GetNumberOfCells() > 0
    if 'BodySpringLUT' in state:
        assert 'ForceMagnitude' in state['Coil'].CellData.keys()
        assert state['BodySpringLUT'].RGBPoints[0] == 0
        assert abs(state['BodySpringLUT'].RGBPoints[-4]-state['BodySpringForceLimit']) < 1e-6
    expected_speed = float(sys.argv[3]) if len(sys.argv) > 3 else 2.
    assert state['Colors'].RGBPoints[0] == 0 and state['Colors'].RGBPoints[-4] == expected_speed
    state['TimeLabel'].UpdatePipeline(time)
    text = servermanager.Fetch(state['TimeLabel']).GetRowData().GetAbstractArray(0).GetValue(0)
    assert f'{time:.2f}' in text
    SaveScreenshot(str(directory / f'frame_{time:.2f}.png'),GetLayout(state['renderView1']),ImageResolution=[1600,900])
print('Reference-case state passed: hidden raw particles, CAD geometry, forces, speed range and time.')
