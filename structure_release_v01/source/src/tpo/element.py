from __future__ import annotations

import numpy as np


def q4_plane_stress_stiffness(
    width: float,
    height: float,
    young: float = 1.0,
    nu: float = 0.3,
    thickness: float = 1.0,
) -> np.ndarray:
    for name, value in (
        ("width", width),
        ("height", height),
        ("young", young),
        ("thickness", thickness),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be a positive finite value")
    if not np.isfinite(nu) or not (-1.0 < nu < 0.5):
        raise ValueError("nu must be finite and satisfy -1.0 < nu < 0.5")

    coords = np.asarray(
        [
            [0.0, 0.0],
            [width, 0.0],
            [width, height],
            [0.0, height],
        ],
        dtype=float,
    )
    constitutive = young / (1.0 - nu**2) * np.asarray(
        [
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - nu) / 2.0],
        ],
        dtype=float,
    )
    points = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
    ke = np.zeros((8, 8), dtype=float)

    for xi in points:
        for eta in points:
            dndnatural = 0.25 * np.asarray(
                [
                    [-(1.0 - eta), -(1.0 - xi)],
                    [+(1.0 - eta), -(1.0 + xi)],
                    [+(1.0 + eta), +(1.0 + xi)],
                    [-(1.0 + eta), +(1.0 - xi)],
                ],
                dtype=float,
            )
            jac = dndnatural.T @ coords
            det_jac = float(np.linalg.det(jac))
            if det_jac <= 0.0:
                raise ValueError("element Jacobian must be positive")
            dndxy = dndnatural @ np.linalg.inv(jac)

            bmat = np.zeros((3, 8), dtype=float)
            for node in range(4):
                bmat[0, 2 * node] = dndxy[node, 0]
                bmat[1, 2 * node + 1] = dndxy[node, 1]
                bmat[2, 2 * node] = dndxy[node, 1]
                bmat[2, 2 * node + 1] = dndxy[node, 0]
            ke += thickness * (bmat.T @ constitutive @ bmat) * det_jac

    return 0.5 * (ke + ke.T)
