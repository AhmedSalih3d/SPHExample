"""Run with pvpython --force-offscreen-rendering --disable-registry."""
from pathlib import Path
import runpy
import sys
from paraview.simple import GetLayout, GetAnimationScene, RenderAllViews, SaveScreenshot, SaveState

root = Path(__file__).resolve().parent
name = sys.argv[1] if len(sys.argv) > 1 else "float1_down"
if name not in ("float1_down", "float1_up", "float2_down"):
    raise ValueError("Choose a Test 14 release name")
launcher = root / "runs/full_mdbc_study/SimpleMDBC" / name / "FreeHeave_Comparison.py"
runpy.run_path(str(launcher))
layout = GetLayout()
layout.SetSize(1200, 1400)
GetAnimationScene().AnimationTime = 1.2
RenderAllViews()
SaveScreenshot(str(root / "results/full_mdbc_study" / f"{name}_paraview.png"), layout)
SaveState(str(launcher.with_suffix(".pvsm")))
