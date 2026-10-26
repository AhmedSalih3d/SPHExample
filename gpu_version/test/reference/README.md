# Independent DualSPHysics motion reference

[dual_motion_reference.cpp](dual_motion_reference.cpp) is an independently
written fixture generator that calls the **unmodified** upstream `JMotion`
engine. The Julia implementation uses its own trajectory and transform
representation; upstream C++ is not vendored into this package.

Upstream source: [DualSPHysics commit ef3721a861fda961f0e2f9ec4cd317b19de99086](https://github.com/DualSPHysics/DualSPHysics/tree/ef3721a861fda961f0e2f9ec4cd317b19de99086/src/source).
The fixture was generated with MSVC 19.44, C++14, on Windows. The selected
inputs cover all nine motion families and the rotational/circular and
degree/radian variants (13 inputs, 33 positions each). They are not copies of
the distributed Motion01–09 XML files or geometry.

To regenerate, check out that upstream commit separately. From a compiler
shell in its `src/source` directory, compile the generator with these upstream
translation units (adjust the generator/output paths):

```text
cl /nologo /EHsc /O2 /std:c++14 /DWIN32 /D_CRT_SECURE_NO_WARNINGS /I. <path>/dual_motion_reference.cpp JMotion.cpp JMotionObj.cpp JMotionPos.cpp JMotionData.cpp JMotionList.cpp JMotionMov.cpp JObject.cpp JException.cpp Functions.cpp JReadDatafile.cpp JXml.cpp tinyxml.cpp tinyxmlerror.cpp tinyxmlparser.cpp tinystr.cpp /Fe:<output>/dual_motion_reference.exe
```

Run the executable in a scratch directory; it writes three small input data
files there. Redirect its standard output to `dual_motion_reference.csv`.
On Windows use `cmd /c` redirection or UTF-8 output, avoiding PowerShell's
legacy UTF-16 default. The CSV includes positions and the velocity averaged
over the **preceding 0.125-second interval**. This averaged quantity is not
the upstream SPH solver's complete boundary-velocity evaluation.

Run the comparison from the repository root:

```powershell
julia --project=gpu_version -t 2,0 gpu_version/test/reference/check_dual_motion.jl
julia --project=gpu_version -t 2,0 gpu_version/test/prescribed_motion.jl
```

For the supplied inputs, the largest absolute position difference on the
validation machine was `8.7e-15` m. Tests allow `2e-14` m for position and
`2e-13` m/s for finite displacement rates. The upstream axis endpoint convention
maps to our right-hand axis as `axisp1 - axisp2`; the Julia fixtures make that
mapping explicit. File inputs subtract their first value and rebase time.

This comparison verifies the selected kinematics. It does not establish
bitwise equality, wall-velocity equality or fluid-solver equivalence.
Analytical derivatives intentionally differ from finite-increment wall
velocities, and arbitrary combinations of simultaneous operations and parent
axis updates need separate semantic validation. The tests explicitly
demonstrate a case with matching positions and different wall velocity.
