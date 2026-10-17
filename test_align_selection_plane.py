import numpy as np
import pytest

from align_selection_plane import (
    alignment_matrix,
    fit_plane_lstsq,
    fit_plane_normals,
    orient_normal,
    plane_rms,
    rotation_between,
    tilt_degrees,
)


def tilted_surface(normal, n=2000, noise=0.0, seed=0):
    """Random rough points around a plane with the given normal through (1, 2, 3)."""
    rng = np.random.default_rng(seed)
    normal = np.asarray(normal, float) / np.linalg.norm(normal)
    u = np.cross(normal, [1.0, 0.0, 0.0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(normal, [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    a, b = rng.uniform(-5, 5, (2, n))
    h = rng.normal(0, noise, n)
    return np.array([1.0, 2.0, 3.0]) + np.outer(a, u) + np.outer(b, v) + np.outer(h, normal)


def angle_between(a, b):
    return tilt_degrees(a, b)


@pytest.mark.parametrize("normal", [(0, 0, 1), (0.1, -0.05, 1), (1, 1, 1), (0.3, 0, -1)])
def test_lstsq_recovers_normal(normal):
    pts = tilted_surface(normal, noise=0.05)
    c, n = fit_plane_lstsq(pts)
    n = orient_normal(n, normal)
    assert angle_between(n, normal) < 0.5
    assert np.allclose(c, pts.mean(axis=0))


def test_lstsq_exact_plane_has_zero_rms():
    pts = tilted_surface((0.2, 0.1, 1))
    c, n = fit_plane_lstsq(pts)
    assert plane_rms(pts, c, n) < 1e-9


def test_lstsq_rejects_degenerate():
    with pytest.raises(ValueError):
        fit_plane_lstsq([[0, 0, 0], [1, 1, 1]])
    with pytest.raises(ValueError):
        fit_plane_lstsq([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])


def test_weighted_lstsq_ignores_zero_weight_outlier():
    pts = tilted_surface((0, 0, 1))
    pts = np.vstack([pts, [[0, 0, 100.0]]])
    w = np.ones(len(pts))
    w[-1] = 0
    c, n = fit_plane_lstsq(pts, w)
    assert angle_between(orient_normal(n, (0, 0, 1)), (0, 0, 1)) < 1e-6


def test_normals_average_area_weighted():
    normals = [[0, 0, 1], [1, 0, 0]]
    c, n = fit_plane_normals(normals, [3.0, 1.0], [[0, 0, 0], [4, 0, 0]])
    assert np.allclose(n, np.array([1, 0, 3]) / np.sqrt(10))
    assert np.allclose(c, [1, 0, 0])


def test_normals_reject_cancelling():
    with pytest.raises(ValueError):
        fit_plane_normals([[0, 0, 1], [0, 0, -1]], [1, 1], [[0, 0, 0], [0, 0, 0]])


@pytest.mark.parametrize(
    "a,b",
    [((0, 0, 1), (0, 0, 1)), ((1, 0, 0), (0, 0, 1)), ((0.3, -0.2, 0.9), (0, 0, 1)),
     ((0, 0, 1), (0, 0, -1)), ((1, 0, 0), (-1, 0, 0))],
)
def test_rotation_between_is_proper_rotation(a, b):
    R = rotation_between(a, b)
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.isclose(np.linalg.det(R), 1.0)
    assert np.allclose(R @ (np.array(a) / np.linalg.norm(a)), np.array(b) / np.linalg.norm(b))


def test_alignment_levels_surface_and_moves_to_zero():
    pts = tilted_surface((0.15, -0.08, 1), noise=0.02)
    c, n = fit_plane_lstsq(pts)
    n = orient_normal(n, (0, 0, 1))
    M = alignment_matrix(c, n, (0, 0, 1), move_to_zero=True)
    out = pts @ M[:3, :3].T + M[:3, 3]

    c2, n2 = fit_plane_lstsq(out)
    assert angle_between(orient_normal(n2, (0, 0, 1)), (0, 0, 1)) < 1e-6
    assert abs(c2[2]) < 1e-9
    # x/y of centroid unchanged; shape preserved
    assert np.allclose(c2[:2], c[:2])
    assert np.allclose(np.linalg.norm(out - out[0], axis=1), np.linalg.norm(pts - pts[0], axis=1))


def test_alignment_without_move_keeps_centroid():
    pts = tilted_surface((1, 0.2, 0.1))
    c, n = fit_plane_lstsq(pts)
    M = alignment_matrix(c, n, (0, 0, 1), move_to_zero=False)
    assert np.allclose(M[:3, :3] @ c + M[:3, 3], c)


def test_subset_fit_aligns_whole_object():
    """Fit only on a patch, apply the transform to everything."""
    pts = tilted_surface((0.05, 0.1, 1), n=5000, noise=0.03)
    patch = pts[: len(pts) // 10]
    c, n = fit_plane_lstsq(patch)
    n = orient_normal(n, (0, 0, 1))
    M = alignment_matrix(c, n, (0, 0, 1))
    out = pts @ M[:3, :3].T + M[:3, 3]
    _, n_all = fit_plane_lstsq(out)
    assert angle_between(orient_normal(n_all, (0, 0, 1)), (0, 0, 1)) < 0.5
