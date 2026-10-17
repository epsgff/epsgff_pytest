import numpy as np
import pytest

from plane_align_core import (
    apply_transform,
    fit_plane,
    in_rect,
    level_matrix,
    near_visible_surface,
    orient_normal,
    project_to_display,
    rotation_between,
    tilt_degrees,
    triangle_geometry,
    zbuffer_to_view_depth,
)


def rough_tilted_points(normal, n=3000, noise=0.03, seed=0):
    rng = np.random.default_rng(seed)
    normal = np.asarray(normal, float) / np.linalg.norm(normal)
    u = np.cross(normal, [1.0, 0, 0]) if abs(normal[0]) < 0.9 else np.cross(normal, [0, 1.0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    a, b = rng.uniform(-5, 5, (2, n))
    return np.array([1.0, 2, 3]) + np.outer(a, u) + np.outer(b, v) + np.outer(rng.normal(0, noise, n), normal)


@pytest.mark.parametrize("normal", [(0, 0, 1), (0.1, -0.05, 1), (1, 1, 1)])
def test_fit_plane_recovers_normal(normal):
    c, n = fit_plane(rough_tilted_points(normal))
    assert tilt_degrees(orient_normal(n, normal), normal) < 0.5


def test_fit_plane_rejects_collinear():
    with pytest.raises(ValueError):
        fit_plane([[0, 0, 0], [1, 1, 1], [2, 2, 2]])


def test_fit_plane_weights():
    pts = np.vstack([rough_tilted_points((0, 0, 1), noise=0), [[0, 0, 100.0]]])
    w = np.ones(len(pts))
    w[-1] = 0
    _, n = fit_plane(pts, w)
    assert tilt_degrees(orient_normal(n, (0, 0, 1))) < 1e-6


@pytest.mark.parametrize("a", [(0, 0, 1), (1, 0, 0), (0.3, -0.2, 0.9), (0, 0, -1)])
def test_rotation_between(a):
    R = rotation_between(a, (0, 0, 1))
    assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1)
    assert np.allclose(R @ (np.array(a) / np.linalg.norm(a)), [0, 0, 1])


def test_level_matrix_levels_whole_object_and_keeps_centroid():
    pts = rough_tilted_points((0.08, -0.05, 1), n=5000)
    patch = pts[:500]  # 只用一部分拟合
    c, n = fit_plane(patch)
    n = orient_normal(n, (0, 0, 1))
    M = level_matrix(c, n)
    out = apply_transform(pts, M)
    _, n2 = fit_plane(out[:500])
    assert tilt_degrees(orient_normal(n2, (0, 0, 1))) < 1e-6
    assert np.allclose(apply_transform(c[None], M)[0], c)  # 绕中心旋转，中心不动
    assert np.allclose(np.linalg.norm(out - out[0], axis=1), np.linalg.norm(pts - pts[0], axis=1))


def test_triangle_geometry():
    pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    c, n, a = triangle_geometry(pts, np.array([[0, 1, 2], [0, 2, 3]]))
    assert np.allclose(a, 0.5) and np.allclose(n, [0, 0, 1])
    assert np.allclose(c[0], [2 / 3, 1 / 3, 0])


def test_project_to_display():
    M = np.diag([0.1, 0.1, 0.1, 1.0])
    xy, front = project_to_display(np.array([[0, 0, 0], [10, 10, 0.0]]), M, (0, 0), (200, 100))
    assert np.allclose(xy, [[100, 50], [200, 100]]) and front.all()


def test_in_rect_any_corner_order():
    xy = np.array([[1, 1], [5, 5], [11, 3]], float)
    assert in_rect(xy, (10, 0, 0, 10)).tolist() == [True, True, False]


@pytest.mark.parametrize("parallel", [False, True])
def test_zbuffer_depth_endpoints(parallel):
    assert np.allclose(zbuffer_to_view_depth(np.array([0.0, 1.0]), 2.0, 50.0, parallel), [2, 50])


def test_zbuffer_perspective_matches_projection():
    n, f, zv = 2.0, 50.0, 10.0
    ndc = (f + n) / (f - n) - 2 * f * n / ((f - n) * zv)
    assert np.isclose(zbuffer_to_view_depth((ndc + 1) / 2, n, f, False), zv)


def test_near_visible_surface_drops_hidden_bottom():
    surface = np.full((4, 4), 10.0)
    xy = np.array([[1.5, 1.5], [1.5, 1.5], [2.2, 3.9]])
    depth = np.array([10.05, 13.0, 9.0])  # 顶面、被挡住的底面、凸起
    assert near_visible_surface(xy, depth, surface, 0.1).tolist() == [True, False, True]
