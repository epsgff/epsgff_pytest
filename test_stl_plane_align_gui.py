"""窗口程序的端到端测试（模拟鼠标框选）。缺少界面库或显示器时自动跳过。"""

import os
import sys

import numpy as np
import pytest

pytest.importorskip("pyvistaqt")
pytest.importorskip("PySide6")
if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
    pytest.skip("no display available", allow_module_level=True)

import pyvista as pv  # noqa: E402
from qtpy import QtWidgets  # noqa: E402

import plane_align_core as core  # noqa: E402
import stl_plane_align  # noqa: E402


@pytest.fixture
def window(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    # 粗糙顶面 + 底座，倾斜约 5.8°
    rng = np.random.default_rng(1)
    top = pv.Plane(center=(1.5, 1, 0.4), i_size=3, j_size=2, i_resolution=120, j_resolution=80)
    top.points[:, 2] += 0.01 * rng.normal(size=top.n_points)
    solid = top.triangulate().extrude((0, 0, -0.6), capping=True).triangulate()
    solid.clear_data()
    solid = solid.clean()
    solid.rotate_x(3.0, inplace=True)
    solid.rotate_y(-5.0, inplace=True)
    solid.save(tmp_path / "scan.stl")

    w = stl_plane_align.MainWindow()
    w.show()
    w.load(str(tmp_path / "scan.stl"))
    w.plotter.view_xy()
    app.processEvents()
    yield w
    w.close()


def drag(w, p0, p1, shift=0, ctrl=0):
    iren = w.selector.iren
    for (x, y), ev in ((p0, "LeftButtonPressEvent"), (p1, "MouseMoveEvent"), (p1, "LeftButtonReleaseEvent")):
        iren.SetEventInformation(int(x), int(y), ctrl, shift)
        iren.InvokeEvent(ev)


def test_box_select_align_undo_save(window, tmp_path, monkeypatch):
    w = window
    W, H = w.plotter.renderer.GetSize()
    w.act_select.setChecked(True)
    drag(w, (W * 0.4, H * 0.4), (W * 0.6, H * 0.6))
    assert w.selected.any()
    assert w.centers[w.selected][:, 2].min() > w.centers[:, 2].min() + 0.3  # 没选到底面
    assert core.tilt_degrees(w._fit()[1]) == pytest.approx(np.hypot(3, 5), abs=0.2)

    pv_before = w.heat.mapper.scalar_range
    w.align()
    assert core.tilt_degrees(w._fit()[1]) < 1e-6
    # 热图跟着更新：摆正后顶面高度范围（色标）明显变窄
    pv_after = w.heat.mapper.scalar_range
    assert np.allclose(w.mesh.point_data["Z"], w.mesh.points[:, 2])
    assert (pv_after[1] - pv_after[0]) < 0.5 * (pv_before[1] - pv_before[0])

    out = tmp_path / "aligned.stl"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **k: (str(out), ""))
    w.save_dialog()
    assert pv.read(out).n_cells == w.mesh.n_cells

    w.undo()
    assert core.tilt_degrees(w._fit()[1]) > 5


def test_shift_adds_ctrl_subtracts_and_drag_rotates(window):
    w = window
    W, H = w.plotter.renderer.GetSize()
    drag(w, (W * 0.3, H * 0.3), (W * 0.5, H * 0.5), shift=1)
    a = w.selected.sum()
    drag(w, (W * 0.5, H * 0.5), (W * 0.7, H * 0.7), shift=1)
    assert w.selected.sum() > a
    drag(w, (W * 0.3, H * 0.3), (W * 0.7, H * 0.7), ctrl=1)
    assert not w.selected.any()

    cam = np.array(w.plotter.camera.position)
    drag(w, (W * 0.4, H * 0.5), (W * 0.6, H * 0.5))  # 普通拖动 = 旋转视图
    assert not np.allclose(cam, w.plotter.camera.position)
