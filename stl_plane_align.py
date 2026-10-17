"""STL 平面摆正工具。

打开 STL → 框选一块区域 → 用选中的面拟合平面 → 按该平面的倾斜把整个物体旋转摆正 → 保存。

运行:  python stl_plane_align.py [file.stl]
"""

import os
import sys

import numpy as np

os.environ.setdefault("QT_API", "pyside6")

import pyvista as pv  # noqa: E402
from pyvistaqt import QtInteractor  # noqa: E402
from qtpy import QtCore, QtGui, QtWidgets  # noqa: E402
from vtkmodules.util.numpy_support import vtk_to_numpy  # noqa: E402
from vtkmodules.vtkCommonCore import vtkFloatArray, vtkPoints  # noqa: E402
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData  # noqa: E402
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera  # noqa: E402
from vtkmodules.vtkRenderingCore import vtkActor2D, vtkPolyDataMapper2D  # noqa: E402

import plane_align_core as core  # noqa: E402

APP_NAME = "STL 平面摆正"
DEPTH_TOLERANCE = 0.03  # 比可见表面深多少（模型尺寸的比例）以内仍算选中


class BoxSelector:
    """左键旋转视图；框选模式下（或按住 Shift/Ctrl）左键拖动画框选择。"""

    def __init__(self, plotter, on_box):
        self.on_box = on_box
        self.select_mode = False
        self._drag = None
        self.iren = plotter.iren.interactor
        self.style = vtkInteractorStyleTrackballCamera()
        self.iren.SetInteractorStyle(self.style)
        self.style.AddObserver("LeftButtonPressEvent", self._press)
        self.style.AddObserver("MouseMoveEvent", self._move)
        self.style.AddObserver("LeftButtonReleaseEvent", self._release)
        self.style.AddObserver("CharEvent", lambda *_: None)  # 屏蔽 VTK 自带按键（q 退出等）

        self._pts = vtkPoints()
        self._lines = vtkCellArray()
        poly = vtkPolyData()
        poly.SetPoints(self._pts)
        poly.SetLines(self._lines)
        self._poly = poly
        mapper = vtkPolyDataMapper2D()
        mapper.SetInputData(poly)
        actor = vtkActor2D()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1.0, 0.85, 0.1)
        actor.GetProperty().SetLineWidth(2)
        plotter.add_actor(actor, name="_box", reset_camera=False, pickable=False)

    def _draw_box(self, p0=None, p1=None):
        self._pts.Reset()
        self._lines.Reset()
        if p0 is not None:
            (x0, y0), (x1, y1) = p0, p1
            for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
                self._pts.InsertNextPoint(x, y, 0)
            self._lines.InsertNextCell(5, [0, 1, 2, 3, 0])
        self._pts.Modified()
        self._poly.Modified()
        self.iren.GetRenderWindow().Render()

    def _press(self, *_):
        shift, ctrl = self.iren.GetShiftKey(), self.iren.GetControlKey()
        if not (self.select_mode or shift or ctrl):
            self.style.OnLeftButtonDown()
            return
        mode = "subtract" if ctrl else "add" if shift else "replace"
        pos = self.iren.GetEventPosition()
        self._drag = [mode, pos, pos]

    def _move(self, *_):
        if self._drag is None:
            self.style.OnMouseMove()
            return
        self._drag[2] = self.iren.GetEventPosition()
        self._draw_box(self._drag[1], self._drag[2])

    def _release(self, *_):
        if self._drag is None:
            self.style.OnLeftButtonUp()
            return
        mode, p0, p1 = self._drag
        self._drag = None
        self._draw_box()
        self.on_box((*p0, *p1), mode)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1500, 850)
        self.setAcceptDrops(True)
        self.path = None
        self.mesh = None
        self.selected = None
        self.view_ref = None  # 选择时朝向镜头的方向，用来确定法向正负
        self.history = []  # 撤销用

        self.plotter = QtInteractor(self)
        self.plotter.set_background("#3a3a3a")
        self.plotter.add_axes()
        self.selector = BoxSelector(self.plotter, self.select_box)

        # 右侧：顶视图高度热图（左键平移、滚轮缩放）
        self.heat = QtInteractor(self)
        self.heat.set_background("white")
        self.heat.enable_parallel_projection()
        self.heat.enable_2d_style()
        heat_box = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(heat_box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        title = QtWidgets.QLabel("顶视图高度热图")
        title.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(title)
        lay.addWidget(self.heat.interactor, 1)
        self.heat_info = QtWidgets.QLabel()
        self.heat_info.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(self.heat_info)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self.plotter.interactor)
        splitter.addWidget(heat_box)
        splitter.setSizes([700, 500])
        self.setCentralWidget(splitter)

        tb = self.addToolBar("工具")
        tb.setMovable(False)

        def add(text, slot, key, checkable=False):
            act = QtGui.QAction(text, self)
            act.setShortcut(QtGui.QKeySequence(key))
            act.setToolTip(f"{text} ({key})")
            act.setCheckable(checkable)
            (act.toggled if checkable else act.triggered).connect(slot)
            tb.addAction(act)
            return act

        add("打开", self.open_dialog, "Ctrl+O")
        self.act_select = add("框选模式", self.set_select_mode, "S", checkable=True)
        add("清除选择", self.clear_selection, "Esc")
        self.act_align = add("摆正", self.align, "Return")
        self.act_undo = add("撤销", self.undo, "Ctrl+Z")
        self.act_save = add("保存", self.save_dialog, "Ctrl+S")
        tb.addSeparator()
        add("顶视图", self.plotter.view_xy, "T")
        add("等轴视图", self.plotter.view_isometric, "I")

        self.info = QtWidgets.QLabel()
        self.statusBar().addPermanentWidget(self.info)
        self.statusBar().showMessage(
            "框选模式(S)下左键拖框选择；Shift+拖=添加，Ctrl+拖=减去。左键旋转·中键平移·滚轮缩放")
        self._refresh()

    # ---- 文件 ----------------------------------------------------------------
    def open_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开 STL", "", "STL (*.stl);;所有文件 (*)")
        if path:
            self.load(path)

    def load(self, path):
        try:
            mesh = pv.read(path)
            if not isinstance(mesh, pv.PolyData):
                mesh = mesh.extract_surface()
            mesh = mesh.triangulate()
            mesh.clear_data()
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, APP_NAME, f"无法读取文件：\n{exc}")
            return
        mesh.points = mesh.points.astype(np.float64)
        self.mesh = mesh
        self.tris = mesh.faces.reshape(-1, 4)[:, 1:]
        self.path = path
        self.selected = np.zeros(mesh.n_cells, dtype=bool)
        self.view_ref = None
        self.history.clear()
        self._update_geometry()
        actor = self.plotter.add_mesh(mesh, name="mesh", color="#c8c8c8", specular=0.3)
        actor.mapper.SetScalarVisibility(False)  # 3D 视图保持灰色，高度颜色只用在热图
        self.heat.add_mesh(mesh, name="heat", scalars="Z", cmap="turbo", lighting=False,
                           scalar_bar_args={"title": "Z", "vertical": True, "fmt": "%.4g",
                                            "color": "black",  # 白底上用黑字
                                            "position_x": 0.82, "position_y": 0.1,
                                            "height": 0.8, "width": 0.1})
        self._draw_selection()
        self._update_heatmap()
        self.plotter.view_isometric()
        self.setWindowTitle(f"{APP_NAME} — {os.path.basename(path)}")
        self._refresh()

    def save_dialog(self):
        if self.mesh is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "保存", os.path.splitext(self.path)[0] + "_aligned.stl", "STL (*.stl)")
        if path:
            pv.PolyData(self.mesh.points, self.mesh.faces).save(path, binary=True)
            self.statusBar().showMessage(f"已保存 {path}")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.load(event.mimeData().urls()[0].toLocalFile())

    # ---- 选择 ----------------------------------------------------------------
    def set_select_mode(self, on):
        self.selector.select_mode = on

    def select_box(self, rect, mode):
        if self.mesh is None:
            return
        ren = self.plotter.renderer
        cam = ren.GetActiveCamera()
        m = cam.GetCompositeProjectionTransformMatrix(ren.GetTiledAspectRatio(), -1, 1)
        world_to_ndc = np.array([[m.GetElement(i, j) for j in range(4)] for i in range(4)])
        origin = np.array(ren.GetOrigin())
        xy, in_front = core.project_to_display(self.centers, world_to_ndc, origin, ren.GetSize())
        hit = core.in_rect(xy, rect) & in_front
        if hit.any():  # 只要看得见的表面，排除被挡住的底面/背面
            idx = np.flatnonzero(hit)
            view_dir = np.array(cam.GetDirectionOfProjection())
            depth = (self.centers[idx] - np.array(cam.GetPosition())) @ view_dir
            hit[idx] = core.near_visible_surface(
                xy[idx] - origin, depth, self._surface_depth(), DEPTH_TOLERANCE * self.mesh.length)
            self.view_ref = -view_dir
        if mode == "replace":
            self.selected = hit
        elif mode == "add":
            self.selected |= hit
        else:
            self.selected &= ~hit
        self._draw_selection()
        self._refresh()

    def _surface_depth(self):
        """每个像素上可见表面到相机的距离。"""
        sel = self.plotter.actors.get("selection")
        if sel is not None:
            sel.SetVisibility(False)
        rw = self.plotter.render_window
        rw.Render()
        ren = self.plotter.renderer
        (x0, y0), (w, h) = ren.GetOrigin(), ren.GetSize()
        z = vtkFloatArray()
        rw.GetZbufferData(x0, y0, x0 + w - 1, y0 + h - 1, z)
        if sel is not None:
            sel.SetVisibility(True)
        cam = ren.GetActiveCamera()
        near, far = cam.GetClippingRange()
        return core.zbuffer_to_view_depth(vtk_to_numpy(z).reshape(h, w), near, far,
                                          cam.GetParallelProjection())

    def clear_selection(self):
        if self.mesh is not None:
            self.selected[:] = False
            self._draw_selection()
            self._refresh()

    # ---- 拟合与摆正 ----------------------------------------------------------
    def _fit(self):
        """用选中三角面的中心（按面积加权）拟合平面。返回 (中心, 法向) 或 None。"""
        if self.mesh is None or self.selected.sum() < 3:
            return None
        try:
            c, n = core.fit_plane(self.centers[self.selected], self.areas[self.selected])
        except ValueError:
            return None
        # 法向取“选择时朝向镜头”的一侧，这样看着顶面选，摆正后顶面朝上
        return c, core.orient_normal(n, self.view_ref if self.view_ref is not None else core.UP)

    def align(self):
        fit = self._fit()
        if fit is None:
            return
        centroid, normal = fit
        # 摆正，并上下平移使选区平均高度为 0
        M = core.level_and_zero_matrix(centroid, normal)
        self.history.append((self.mesh.points.copy(), self.view_ref))
        self.mesh.points = core.apply_transform(self.mesh.points, M)
        if self.view_ref is not None:
            self.view_ref = M[:3, :3] @ self.view_ref
        self._update_geometry()
        self._draw_selection()
        self._update_heatmap()
        self._refresh(f"已旋转 {core.tilt_degrees(normal):.4f}°，Z 平移 {-centroid[2]:+.4g}"
                      "（选区平均高度 = 0）")

    def undo(self):
        if self.history:
            self.mesh.points, self.view_ref = self.history.pop()
            self._update_geometry()
            self._draw_selection()
            self._update_heatmap()
            self._refresh("已撤销")

    # ---- 显示 ----------------------------------------------------------------
    def _update_geometry(self):
        self.centers, self.normals, self.areas = core.triangle_geometry(self.mesh.points, self.tris)
        self.mesh.point_data["Z"] = self.mesh.points[:, 2]

    def _update_heatmap(self):
        """按当前（摆正后的）高度刷新热图色标并重新取顶视图。"""
        if self.mesh is None:
            return
        zmin, zmax = core.top_view_height_range(self.mesh.points)
        self.heat.update_scalar_bar_range([zmin, zmax], name="Z")
        self.heat.view_xy()
        self.heat.reset_camera()
        cam = self.heat.camera  # 缩小一点并左移，给右边的色标留位置
        cam.zoom(0.85)
        w, h = self.heat.renderer.GetSize()
        shift = 0.09 * 2 * cam.parallel_scale * w / max(h, 1)
        cam.focal_point = np.add(cam.focal_point, (shift, 0, 0))
        cam.position = np.add(cam.position, (shift, 0, 0))
        self.heat.render()
        self.heat_info.setText(f"顶面高度 {zmin:.4g} ~ {zmax:.4g}（PV {zmax - zmin:.4g}）")

    def _draw_selection(self):
        if self.mesh is None or not self.selected.any():
            self.plotter.remove_actor("selection")
            return
        faces = np.c_[np.full(self.selected.sum(), 3), self.tris[self.selected]].ravel()
        actor = self.plotter.add_mesh(pv.PolyData(self.mesh.points, faces), name="selection",
                                      color="#ff3b30", pickable=False, reset_camera=False)
        actor.mapper.SetResolveCoincidentTopologyToPolygonOffset()
        actor.mapper.SetRelativeCoincidentTopologyPolygonOffsetParameters(-2, -2)

    def _refresh(self, message=None):
        fit = self._fit()
        self.act_align.setEnabled(fit is not None)
        self.act_undo.setEnabled(bool(self.history))
        self.act_save.setEnabled(self.mesh is not None)
        if self.mesh is None:
            self.info.setText("打开或拖入一个 STL 文件")
        elif fit is None:
            self.info.setText(f"{self.mesh.n_cells:,} 个面 | 请框选参考区域")
        else:
            self.info.setText(f"已选 {self.selected.sum():,} 个面 | 选区平面倾斜 "
                              f"{core.tilt_degrees(fit[1]):.4f}°")
        if message:
            self.statusBar().showMessage(message)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    if len(sys.argv) > 1:
        win.load(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
