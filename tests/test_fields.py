"""Tests of fields.py: the symmetry rules for vectors and nematic tensors are checked against
fields whose transformation is known independently (the gradient of a scalar, the tensor of
a director field)."""

import numpy as np
import pytest

from phasewall.fields import Field, mirror_x, random_transform, rotate90, transform

N = 16


def smooth_scalar(seed: int = 0, n: int = N) -> np.ndarray:
    """A random periodic field without any symmetry, indexed [x, y]."""
    rng = np.random.default_rng(seed)
    x = 2 * np.pi * np.arange(n) / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    f = np.zeros((n, n))
    for kx, ky in [(1, 0), (0, 1), (1, 1), (2, -1), (1, 3)]:
        f += rng.normal() * np.cos(kx * X + ky * Y + rng.uniform(0, 2 * np.pi))
    return f


def gradient(f: np.ndarray) -> np.ndarray:
    """Periodic central differences, as a vector field [x, y, component]."""
    gx = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / 2
    gy = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / 2
    return np.stack([gx, gy], axis=-1)


def tensor_of_director(theta: np.ndarray, S: float = 0.8) -> np.ndarray:
    """(Q_xx, Q_xy) = S (n_x^2 - 1/2, n_x n_y) for the director n = (cos theta, sin theta)."""
    nx, ny = np.cos(theta), np.sin(theta)
    return np.stack([S * (nx**2 - 0.5), S * nx * ny], axis=-1)


def test_field_checks_kind_and_shape():
    with pytest.raises(ValueError):
        Field("a", ("a",), kind="tensor")
    with pytest.raises(ValueError):
        Field("p", ("p",), kind="vector", relative=True)
    Field("rho", ("rho",)).check(np.zeros((4, 4)))
    with pytest.raises(ValueError):
        Field("rho", ("rho",)).check(np.zeros((4, 4, 2)))
    with pytest.raises(ValueError):
        Field("p", ("p",), kind="vector").check(np.zeros((4, 4, 3)))


def test_rotated_gradient_is_gradient_of_rotated_scalar():
    f = smooth_scalar()
    assert np.allclose(rotate90(gradient(f), "vector"), gradient(rotate90(f, "scalar")))


def test_mirrored_gradient_is_gradient_of_mirrored_scalar():
    f = smooth_scalar()
    assert np.allclose(mirror_x(gradient(f), "vector"), gradient(mirror_x(f, "scalar")))


def test_nematic_rules_follow_the_director():
    """Turning the director field as a vector field and then building Q gives the same as
    building Q and then applying the nematic rules (Q -> -Q, Q_xy -> -Q_xy)."""
    theta = smooth_scalar(1)
    n = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    Q = tensor_of_director(theta)
    n_rot = rotate90(n, "vector")
    assert np.allclose(rotate90(Q, "nematic"), tensor_of_director(np.arctan2(n_rot[..., 1], n_rot[..., 0])))
    n_mir = mirror_x(n, "vector")
    assert np.allclose(mirror_x(Q, "nematic"), tensor_of_director(np.arctan2(n_mir[..., 1], n_mir[..., 0])))


def test_group_relations():
    """Four turns and two mirrors are the identity; mirror . turn . mirror is the inverse turn."""
    fields = [Field("rho", ("rho",)), Field("p", ("p",), "vector"), Field("Q", ("Q",), "nematic")]
    f = smooth_scalar(2)
    snap = {"rho": f, "p": gradient(f), "Q": tensor_of_director(f)}
    same = transform(snap, fields, turns=4)
    twice = transform(transform(snap, fields, mirror=True), fields, mirror=True)
    for name, a in snap.items():
        assert np.array_equal(same[name], a)
        assert np.array_equal(twice[name], a)
    for fl in fields:
        a = snap[fl.name]
        left = mirror_x(rotate90(mirror_x(a, fl.kind), fl.kind), fl.kind)
        right = rotate90(rotate90(rotate90(a, fl.kind), fl.kind), fl.kind)
        assert np.allclose(left, right)


def test_transform_shift_and_dropped_fields():
    fields = [Field("rho", ("rho",))]
    f = smooth_scalar(3)
    out = transform({"rho": f, "other": f}, fields, shift=(3, -2))
    assert set(out) == {"rho"}
    assert np.array_equal(out["rho"], np.roll(f, (3, -2), axis=(0, 1)))


def test_random_transform_keeps_values_and_needs_a_square_grid():
    fields = [Field("p", ("p",), "vector")]
    p = gradient(smooth_scalar(4))
    rng = np.random.default_rng(0)
    for _ in range(10):
        q = random_transform({"p": p}, fields, rng)["p"]
        # a symmetry only moves the vectors around and turns them: their lengths are kept
        assert np.allclose(np.sort(np.hypot(q[..., 0], q[..., 1]).ravel()),
                           np.sort(np.hypot(p[..., 0], p[..., 1]).ravel()))
    with pytest.raises(ValueError):
        random_transform({"p": np.zeros((8, 6, 2))}, fields, rng)
