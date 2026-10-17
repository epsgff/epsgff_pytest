"""STL 平面摆正工具的计算部分（纯 numpy，不依赖界面）。"""

import numpy as np

UP = np.array([0.0, 0.0, 1.0])


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0.0:
        raise ValueError("zero-length vector")
    return v / n


def fit_plane(points, weights=None):
    """最小二乘平面：返回 (中心点, 单位法向)。weights 可选（如三角面面积）。"""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(pts) < 3:
        raise ValueError("至少需要 3 个点")
    w = np.ones(len(pts)) if weights is None else np.asarray(weights, dtype=float).reshape(-1)
    if w.sum() <= 0:
        raise ValueError("权重之和必须为正")
    w = w / w.sum()
    centroid = w @ pts
    d = pts - centroid
    evals, evecs = np.linalg.eigh((d * w[:, None]).T @ d)
    if evals[1] <= 1e-12 * evals[2]:
        raise ValueError("选中的点共线，无法确定平面")
    return centroid, _unit(evecs[:, 0])


def orient_normal(normal, reference):
    """让法向与 reference 同侧。"""
    return -normal if np.dot(normal, reference) < 0 else normal


def rotation_between(a, b):
    """把方向 a 转到方向 b 的最小旋转（3x3）。"""
    a, b = _unit(a), _unit(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = _unit(np.cross(a, helper))
        return 2.0 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def level_matrix(centroid, normal, target=UP):
    """4x4 变换：绕平面中心旋转，使平面法向对准 target（默认 +Z）。"""
    R = rotation_between(normal, target)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = centroid - R @ centroid
    return M


def level_and_zero_matrix(centroid, normal, target=UP):
    """摆正后再沿 Z 平移，让选区的平均高度（即拟合中心的 Z）变为 0。

    旋转绕拟合中心进行，中心本身不动；而拟合中心就是选区（按面积加权）的平均位置，
    所以摆正后只需把整体下移 centroid[2]。
    """
    M = level_matrix(centroid, normal, target)
    M[2, 3] -= float(np.asarray(centroid, dtype=float)[2])
    return M


def apply_transform(points, M):
    return np.asarray(points, dtype=float) @ M[:3, :3].T + M[:3, 3]


def tilt_degrees(normal, target=UP):
    return float(np.degrees(np.arccos(np.clip(np.dot(_unit(normal), _unit(target)), -1.0, 1.0))))


def triangle_geometry(points, triangles):
    """每个三角面的中心、单位法向、面积。"""
    p = np.asarray(points, dtype=float)
    v0, v1, v2 = p[triangles[:, 0]], p[triangles[:, 1]], p[triangles[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    length = np.linalg.norm(cross, axis=1)
    normals = cross / np.where(length > 0, length, 1.0)[:, None]
    return (v0 + v1 + v2) / 3.0, normals, 0.5 * length


# --- 屏幕框选 ---------------------------------------------------------------

def project_to_display(points, world_to_ndc, origin, size):
    """世界坐标 -> 屏幕像素。返回 (xy, 是否在相机前方)。"""
    pts = np.asarray(points, dtype=float)
    M = np.asarray(world_to_ndc, dtype=float)
    h = pts @ M[:3, :3].T + M[:3, 3]
    w = pts @ M[3, :3] + M[3, 3]
    in_front = w > 1e-12
    ndc = h[:, :2] / np.where(in_front, w, 1.0)[:, None]
    return np.asarray(origin, float) + (ndc + 1.0) * 0.5 * np.asarray(size, float), in_front


def in_rect(xy, rect):
    """rect = (x0, y0, x1, y1)，角点顺序任意。"""
    x0, y0, x1, y1 = rect
    xa, xb = sorted((x0, x1))
    ya, yb = sorted((y0, y1))
    return (xy[:, 0] >= xa) & (xy[:, 0] <= xb) & (xy[:, 1] >= ya) & (xy[:, 1] <= yb)


def zbuffer_to_view_depth(zbuf, near, far, parallel):
    """深度缓冲值 [0,1] -> 沿视线方向到相机的距离。"""
    z = np.asarray(zbuf, dtype=float)
    if parallel:
        return near + z * (far - near)
    ndc = 2.0 * z - 1.0
    return 2.0 * far * near / ((far + near) - ndc * (far - near))


def near_visible_surface(xy, depth, surface_depth, tolerance):
    """只保留不比可见表面深 tolerance 以上的点（排除被挡住的底面/背面）。"""
    h, w = surface_depth.shape
    px = np.clip(np.floor(xy[:, 0]).astype(int), 0, w - 1)
    py = np.clip(np.floor(xy[:, 1]).astype(int), 0, h - 1)
    return depth <= surface_depth[py, px] + tolerance


# --- 顶视图高度热图 -----------------------------------------------------------

def top_view_height_range(points, bins=None, percentiles=(0.5, 99.5)):
    """顶视图里能看到的表面的高度范围，用作热图色标。

    把点按 (x, y) 分格，每格取最高点（俯视时看到的就是它），
    再取这些高度的分位数，避免被底座底面和个别尖刺拉宽色标。
    """
    p = np.asarray(points, dtype=float)
    if bins is None:  # 格子要比网格点间距粗，每格才一定落有顶面点
        bins = int(np.clip(np.sqrt(len(p) / 16), 8, 256))
    lo, hi = p[:, :2].min(axis=0), p[:, :2].max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    ij = np.minimum(((p[:, :2] - lo) / span * bins).astype(int), bins - 1)
    top = np.full(bins * bins, -np.inf)
    np.maximum.at(top, ij[:, 1] * bins + ij[:, 0], p[:, 2])
    top = top[np.isfinite(top)]
    zmin, zmax = np.percentile(top, percentiles)
    if zmax <= zmin:
        zmax = zmin + 1e-9
    return float(zmin), float(zmax)
