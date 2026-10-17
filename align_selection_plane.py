"""Blender add-on: fit a plane to the selected part of a mesh and level the object.

用途 / Usage
------------
1. Blender → Edit → Preferences → Add-ons → Install... 选择本文件并启用。
2. 选中扫描得到的 STL 物体，Tab 进入编辑模式。
3. 选中一块你认为“应该是水平”的区域（面或顶点都可以，C 刷选 / 框选均可）。
4. 3D 视图按 N → “Align” 标签页 → 点“对齐到选区平面”，
   或者 Mesh 菜单 / F3 搜索 “Align Selection Plane”。
5. 按 F9（或左下角的操作面板）可以改拟合方法、目标轴、是否旋转网格数据等。

拟合方法
--------
- 最小二乘 (LSQ)：对选中顶点做 PCA/SVD，取方差最小的方向作为法向。
  对噪声大的扫描数据最稳定，推荐。
- 面法线平均 (NORMALS)：选中面法线按面积加权求平均。需要选中面。

The fitting math is plain numpy (no bpy) so it can be unit-tested outside Blender.
"""

import numpy as np

bl_info = {
    "name": "Align Selection Plane",
    "author": "epsgff",
    "version": (1, 0, 0),
    "blender": (2, 93, 0),
    "location": "View3D > Sidebar > Align  (Edit Mode)",
    "description": "Fit a plane to the selected vertices/faces and rotate the object so that plane is level",
    "category": "Mesh",
}

AXES = {
    "POS_Z": (0.0, 0.0, 1.0),
    "NEG_Z": (0.0, 0.0, -1.0),
    "POS_X": (1.0, 0.0, 0.0),
    "NEG_X": (-1.0, 0.0, 0.0),
    "POS_Y": (0.0, 1.0, 0.0),
    "NEG_Y": (0.0, -1.0, 0.0),
}


# ---------------------------------------------------------------------------
# Pure numpy math
# ---------------------------------------------------------------------------

def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0.0:
        raise ValueError("zero-length vector")
    return v / n


def fit_plane_lstsq(points, weights=None):
    """Least-squares plane through points.

    Returns (centroid, unit_normal). The normal is the direction of smallest
    variance of the (optionally weighted) point cloud.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) < 3:
        raise ValueError("need at least 3 points to fit a plane")
    if weights is None:
        w = np.full(len(pts), 1.0 / len(pts))
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape[0] != len(pts) or np.any(w < 0) or w.sum() <= 0:
            raise ValueError("weights must be non-negative, one per point, with positive sum")
        w = w / w.sum()

    centroid = w @ pts
    d = pts - centroid
    cov = (d * w[:, None]).T @ d
    evals, evecs = np.linalg.eigh(cov)  # ascending eigenvalues
    if evals[2] <= 0 or evals[1] <= 1e-12 * evals[2]:
        raise ValueError("points are coincident or collinear; cannot define a plane")
    return centroid, _unit(evecs[:, 0])


def fit_plane_normals(normals, areas, centers):
    """Plane from area-weighted average of face normals.

    Returns (centroid, unit_normal); centroid is the area-weighted face centre.
    """
    normals = np.asarray(normals, dtype=float).reshape(-1, 3)
    areas = np.asarray(areas, dtype=float).reshape(-1)
    centers = np.asarray(centers, dtype=float).reshape(-1, 3)
    if len(normals) == 0:
        raise ValueError("no faces given")
    total = areas.sum()
    if total <= 0:
        raise ValueError("selected faces have zero area")
    n = (normals * areas[:, None]).sum(axis=0)
    if np.linalg.norm(n) <= 1e-12 * total:
        raise ValueError("face normals cancel out; cannot define a plane")
    centroid = (centers * areas[:, None]).sum(axis=0) / total
    return centroid, _unit(n)


def orient_normal(normal, reference):
    """Flip normal so it points to the same side as reference."""
    normal = np.asarray(normal, dtype=float)
    return -normal if np.dot(normal, reference) < 0 else normal


def rotation_between(a, b):
    """3x3 rotation matrix taking direction a onto direction b (shortest arc)."""
    a = _unit(a)
    b = _unit(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        # 180 degrees: rotate around any axis perpendicular to a.
        helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = _unit(np.cross(a, helper))
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def alignment_matrix(centroid, normal, target, move_to_zero=True):
    """4x4 world-space transform that rotates normal onto target about centroid.

    If move_to_zero is True the plane is also shifted along target so it passes
    through the origin's level (e.g. z = 0 for target +Z).
    """
    centroid = np.asarray(centroid, dtype=float)
    target = _unit(target)
    R = rotation_between(normal, target)
    new_centroid = centroid - np.dot(centroid, target) * target if move_to_zero else centroid
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = new_centroid - R @ centroid
    return M


def plane_rms(points, centroid, normal):
    """RMS distance of points from the plane (a flatness measure)."""
    d = (np.asarray(points, dtype=float).reshape(-1, 3) - centroid) @ _unit(normal)
    return float(np.sqrt(np.mean(d * d)))


def tilt_degrees(normal, target):
    c = np.clip(np.dot(_unit(normal), _unit(target)), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


# ---------------------------------------------------------------------------
# Blender integration (only when running inside Blender)
# ---------------------------------------------------------------------------

try:
    import bpy
    import bmesh
    from mathutils import Matrix
except ImportError:  # running under plain Python / pytest
    bpy = None


def _read_selection(obj):
    """Return world-space selected vertices and selected face data as numpy arrays."""
    obj.update_from_editmode()
    me = obj.data
    mw = np.array(obj.matrix_world, dtype=float)
    m3 = mw[:3, :3]

    nv = len(me.vertices)
    co = np.empty(nv * 3, dtype=np.float32)
    me.vertices.foreach_get("co", co)
    vsel = np.empty(nv, dtype=bool)
    me.vertices.foreach_get("select", vsel)
    verts = co.reshape(-1, 3).astype(float)[vsel] @ m3.T + mw[:3, 3]

    npoly = len(me.polygons)
    psel = np.empty(npoly, dtype=bool)
    me.polygons.foreach_get("select", psel)
    if not psel.any():
        empty = np.zeros((0, 3))
        return verts, empty, np.zeros(0), empty

    nrm = np.empty(npoly * 3, dtype=np.float32)
    me.polygons.foreach_get("normal", nrm)
    area = np.empty(npoly, dtype=np.float32)
    me.polygons.foreach_get("area", area)
    ctr = np.empty(npoly * 3, dtype=np.float32)
    me.polygons.foreach_get("center", ctr)

    nrm = nrm.reshape(-1, 3).astype(float)[psel]
    area = area.astype(float)[psel]
    ctr = ctr.reshape(-1, 3).astype(float)[psel] @ m3.T + mw[:3, 3]

    # Normals transform with the inverse-transpose; area scales accordingly.
    nmat = np.linalg.inv(m3).T
    nw = nrm @ nmat.T
    scale = np.linalg.norm(nw, axis=1)
    area = area * scale * abs(np.linalg.det(m3))
    nw = nw / np.where(scale > 0, scale, 1.0)[:, None]
    return verts, nw, area, ctr


if bpy is not None:

    class MESH_OT_align_selection_plane(bpy.types.Operator):
        """Fit a plane to the selection and rotate the object so that plane is level"""

        bl_idname = "mesh.align_selection_plane"
        bl_label = "Align Selection Plane"
        bl_options = {"REGISTER", "UNDO"}

        method: bpy.props.EnumProperty(
            name="Method",
            items=[
                ("LSQ", "Least squares (vertices)", "Best-fit plane through selected vertices (robust to noise)"),
                ("NORMALS", "Average face normals", "Area-weighted average of selected face normals"),
            ],
            default="LSQ",
        )
        target_axis: bpy.props.EnumProperty(
            name="Target axis",
            items=[
                ("POS_Z", "+Z", ""), ("NEG_Z", "-Z", ""),
                ("POS_X", "+X", ""), ("NEG_X", "-X", ""),
                ("POS_Y", "+Y", ""), ("NEG_Y", "-Y", ""),
            ],
            default="POS_Z",
        )
        apply_to: bpy.props.EnumProperty(
            name="Apply to",
            items=[
                ("OBJECT", "Object transform", "Change the object's rotation/location; mesh data untouched"),
                ("MESH", "Mesh data", "Rotate the vertices themselves; object transform untouched"),
            ],
            default="OBJECT",
        )
        move_to_zero: bpy.props.BoolProperty(
            name="Move plane to 0",
            description="Also shift the object so the fitted plane sits at 0 along the target axis",
            default=True,
        )

        @classmethod
        def poll(cls, context):
            return context.mode == "EDIT_MESH" and context.edit_object is not None

        def execute(self, context):
            obj = context.edit_object
            verts, fnormals, fareas, fcenters = _read_selection(obj)
            target = np.array(AXES[self.target_axis])

            try:
                if self.method == "NORMALS":
                    if len(fnormals) == 0:
                        self.report({"ERROR"}, "Average face normals needs selected faces")
                        return {"CANCELLED"}
                    centroid, normal = fit_plane_normals(fnormals, fareas, fcenters)
                else:
                    centroid, normal = fit_plane_lstsq(verts)
                    # Keep the side the faces point to; otherwise take the shortest rotation.
                    if len(fnormals):
                        normal = orient_normal(normal, (fnormals * fareas[:, None]).sum(axis=0))
                    else:
                        normal = orient_normal(normal, target)
            except ValueError as exc:
                self.report({"ERROR"}, str(exc))
                return {"CANCELLED"}

            delta = alignment_matrix(centroid, normal, target, self.move_to_zero)
            mw = np.array(obj.matrix_world, dtype=float)

            if self.apply_to == "OBJECT":
                obj.matrix_world = Matrix((delta @ mw).tolist())
            else:
                local = np.linalg.inv(mw) @ delta @ mw
                bm = bmesh.from_edit_mesh(obj.data)
                bmesh.ops.transform(bm, matrix=Matrix(local.tolist()), verts=bm.verts)
                bmesh.update_edit_mesh(obj.data)

            rms_pts = verts if len(verts) else fcenters
            self.report(
                {"INFO"},
                "Rotated %.4f°, %d verts, plane RMS %.4g"
                % (tilt_degrees(normal, target), len(verts), plane_rms(rms_pts, centroid, normal)),
            )
            return {"FINISHED"}

    class VIEW3D_PT_align_selection_plane(bpy.types.Panel):
        bl_label = "Align Selection Plane"
        bl_space_type = "VIEW_3D"
        bl_region_type = "UI"
        bl_category = "Align"
        bl_context = "mesh_edit"

        def draw(self, context):
            col = self.layout.column(align=True)
            col.label(text="Select the reference area, then:")
            op = col.operator(MESH_OT_align_selection_plane.bl_idname, text="对齐 (最小二乘)")
            op.method = "LSQ"
            op = col.operator(MESH_OT_align_selection_plane.bl_idname, text="对齐 (面法线平均)")
            op.method = "NORMALS"
            col.label(text="F9: change axis / apply-to options")

    def _menu_func(self, context):
        self.layout.operator(MESH_OT_align_selection_plane.bl_idname)

    _classes = (MESH_OT_align_selection_plane, VIEW3D_PT_align_selection_plane)

    def register():
        for cls in _classes:
            bpy.utils.register_class(cls)
        bpy.types.VIEW3D_MT_edit_mesh.append(_menu_func)

    def unregister():
        bpy.types.VIEW3D_MT_edit_mesh.remove(_menu_func)
        for cls in reversed(_classes):
            bpy.utils.unregister_class(cls)

    if __name__ == "__main__":  # run from Blender's Text Editor
        register()
