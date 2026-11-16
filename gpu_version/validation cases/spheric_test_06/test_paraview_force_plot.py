"""Run with pvpython --no-mpi --force-offscreen-rendering."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from paraview import servermanager
from paraview.simple import (
    GetAnimationScene, GetLayout, GetSources, GetViews, LoadState, RenderAllViews, ResetSession,
)

from paraview_force_plot import CASE, make_force_plot, read_reference


class ForcePlotTests(unittest.TestCase):
    def tearDown(self):
        ResetSession()

    def test_combined_layout_survives_state_reload_with_raw_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = root / "Re100_n040.csv"
            history.write_text(
                "time,cd_pressure,cd_viscous\n0,0,0\n0.02,-3,0.8\n0.04,7,0.2\n",
                encoding="utf-8",
            )
            simulation_state = root / "particles.py"
            simulation_state.write_text(
                "from paraview.simple import CreateLayout, CreateView\n"
                "layout = CreateLayout()\n"
                "layout.AssignView(0, CreateView('RenderView'))\n",
                encoding="utf-8",
            )
            args = SimpleNamespace(results=root, output=root / "plots", n=40,
                                   component="both", simulation_state=simulation_state)
            make_force_plot(100, args)
            ResetSession()
            LoadState(str(args.output / "ForceComparison_Re100.pvsm"))

            views = GetViews()
            self.assertEqual(len(views), 3)
            render_view = next(view for view in views if view.GetXMLName() == "RenderView")
            layout = GetLayout(render_view)
            self.assertEqual(layout.GetViewLocation(render_view), 1)
            charts = {view.ChartTitle: view for view in views
                      if view.GetXMLName() == "XYChartView"}
            for title, location, column in (
                ("Re = 100: pressure drag", 5, "cd_pressure"),
                ("Re = 100: viscous drag", 6, "cd_viscous"),
            ):
                chart = charts[title]
                self.assertEqual(GetLayout(chart), layout)
                self.assertEqual(layout.GetViewLocation(chart), location)
                self.assertEqual(len(chart.Representations), 3)
                for display in chart.Representations:
                    self.assertEqual(display.XArrayName, "time")
                    if display.Input.GetXMLName() == "CSVReader":
                        self.assertEqual(list(display.SeriesVisibility), [column])

            reader = next(source for source in GetSources().values()
                          if source.GetXMLName() == "CSVReader"
                          and Path(source.FileName[0]) == history)
            table = servermanager.Fetch(reader)
            self.assertEqual(table.GetNumberOfRows(), 3)
            for name, expected in (("time", (0, 0.02, 0.04)),
                                   ("cd_pressure", (0, -3, 7)),
                                   ("cd_viscous", (0, 0.8, 0.2))):
                column = table.GetColumnByName(name)
                self.assertEqual(tuple(column.GetValue(i) for i in range(3)), expected)

            scene = GetAnimationScene()
            cursors = [source for source in GetSources().values()
                       if source.GetXMLName() == "ProgrammableSource"]
            self.assertEqual(len(cursors), 2)
            for time, pressure, viscous in ((0, 0, 0), (0.01, -1.5, 0.4),
                                            (0.02, -3, 0.8), (0.04, 7, 0.2)):
                scene.AnimationTime = time
                RenderAllViews()
                for cursor in cursors:
                    table = servermanager.Fetch(cursor)
                    self.assertEqual(table.GetColumnByName("time").GetValue(0), time)
                    expected = pressure if "Pressure" in cursor.GetLogName() else viscous
                    self.assertAlmostEqual(
                        table.GetColumnByName("Current SPH drag, L/dp=40").GetValue(0), expected
                    )

    def test_chart_only_single_component_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Re100_n040.csv").write_text(
                "time,cd_pressure,cd_viscous\n0,0,0\n0.02,1,0.2\n", encoding="utf-8"
            )
            args = SimpleNamespace(results=root, output=root / "plots", n=40,
                                   component="viscous", simulation_state=None)
            make_force_plot(100, args)
            views = GetViews()
            self.assertEqual(len(views), 1)
            self.assertEqual(views[0].ChartTitle, "Re = 100: viscous drag")
            self.assertEqual(GetLayout(views[0]).GetViewLocation(views[0]), 0)

    def test_initial_hold_shifts_reference_but_not_simulation_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Re100_n040.csv").write_text(
                "time,motion_time,cd_pressure,cd_viscous\n"
                "0,-4,0,0\n4,0,0,0\n4.02,0.02,1,0.2\n", encoding="utf-8"
            )
            args = SimpleNamespace(results=root, output=root / "plots", n=40,
                                   component="pressure", simulation_state=None)
            make_force_plot(100, args)
            reference = next(source for source in GetSources().values()
                             if source.GetXMLName() == "CSVReader"
                             and "reference" in source.FileName[0])
            table = servermanager.Fetch(reference)
            reference_start = read_reference(CASE / "reference" / "Force_Re100.dat")[0][0]
            self.assertEqual(table.GetColumnByName("time").GetValue(0), reference_start + 4.0)
            self.assertEqual(GetViews()[0].BottomAxisRangeMaximum, 4.02)


if __name__ == "__main__":
    unittest.main()
