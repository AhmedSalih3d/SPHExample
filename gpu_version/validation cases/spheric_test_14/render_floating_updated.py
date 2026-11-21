"""Render the completed floating UpdatedMDBC comparison with pvpython."""
from pathlib import Path
import runpy
from paraview.simple import GetLayout, GetAnimationScene, RenderAllViews, SaveScreenshot, SaveState

root = Path(__file__).resolve().parent
launcher = root / "runs/floating_updated_study/UpdatedMDBC/float1_down/FreeHeave_Comparison.py"
runpy.run_path(str(launcher))
layout = GetLayout()
layout.SetSize(1200,1400)
GetAnimationScene().AnimationTime = 1.2
RenderAllViews()
SaveScreenshot(str(root / "results/floating_updated_study/paraview_updated.png"),layout)
SaveState(str(launcher.with_suffix(".pvsm")))
