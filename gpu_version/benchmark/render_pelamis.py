"""Render the generated state: pvpython render_pelamis.py state.py frames/ [stride]."""
import runpy
import sys
from pathlib import Path
from paraview.simple import GetAnimationScene, GetLayout, RenderAllViews, SaveScreenshot

state = runpy.run_path(sys.argv[1])
directory = Path(sys.argv[2])
directory.mkdir(parents=True, exist_ok=True)
scene = GetAnimationScene()
scene.UpdateAnimationUsingDataTimeSteps()
times = list(state['Simulation_vtkhdf'].TimestepValues)
stride = int(sys.argv[3]) if len(sys.argv) > 3 else 1
assert stride > 0
selected = times[::stride]
if selected[-1] != times[-1]:
    selected.append(times[-1])
for index,time in enumerate(selected):
    scene.AnimationTime = time
    RenderAllViews()
    SaveScreenshot(str(directory / f'frame_{index:04d}.png'), GetLayout(state['renderView1']), ImageResolution=[1600,900])
print(f'Rendered {len(selected)} frames from {times[0]:.2f} to {times[-1]:.2f} s.')
