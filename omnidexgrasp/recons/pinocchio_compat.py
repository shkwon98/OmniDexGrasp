"""Minimal Pinocchio compatibility used by the MegaPose pose-estimation path."""
from __future__ import annotations

import sys
import types

import numpy as np


class Quaternion:
    def __init__(self, *args):
        if len(args) == 1:
            self._wxyz = self._from_rotation_matrix(np.asarray(args[0], dtype=float))
        elif len(args) == 4:
            self._wxyz = np.asarray(args, dtype=float)
        else:
            raise ValueError("Quaternion expects a rotation matrix or w, x, y, z")

    @staticmethod
    def _from_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
        trace = float(np.trace(rotation))
        if trace > 0:
            scale = np.sqrt(trace + 1.0) * 2.0
            w = 0.25 * scale
            x = (rotation[2, 1] - rotation[1, 2]) / scale
            y = (rotation[0, 2] - rotation[2, 0]) / scale
            z = (rotation[1, 0] - rotation[0, 1]) / scale
        else:
            idx = int(np.argmax(np.diag(rotation)))
            if idx == 0:
                scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
                w = (rotation[2, 1] - rotation[1, 2]) / scale
                x = 0.25 * scale
                y = (rotation[0, 1] + rotation[1, 0]) / scale
                z = (rotation[0, 2] + rotation[2, 0]) / scale
            elif idx == 1:
                scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
                w = (rotation[0, 2] - rotation[2, 0]) / scale
                x = (rotation[0, 1] + rotation[1, 0]) / scale
                y = 0.25 * scale
                z = (rotation[1, 2] + rotation[2, 1]) / scale
            else:
                scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
                w = (rotation[1, 0] - rotation[0, 1]) / scale
                x = (rotation[0, 2] + rotation[2, 0]) / scale
                y = (rotation[1, 2] + rotation[2, 1]) / scale
                z = 0.25 * scale
        return np.asarray([w, x, y, z], dtype=float)

    def normalize(self) -> None:
        norm = np.linalg.norm(self._wxyz)
        if norm == 0:
            raise ValueError("Cannot normalize a zero quaternion")
        self._wxyz = self._wxyz / norm

    def matrix(self) -> np.ndarray:
        self.normalize()
        w, x, y, z = self._wxyz
        return np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=float,
        )

    def coeffs(self) -> np.ndarray:
        w, x, y, z = self._wxyz
        return np.asarray([x, y, z, w], dtype=float)


class SE3:
    def __init__(self, rotation: np.ndarray, translation: np.ndarray):
        self.rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
        self.translation = np.asarray(translation, dtype=float).reshape(3)

    @property
    def homogeneous(self) -> np.ndarray:
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = self.rotation
        matrix[:3, 3] = self.translation
        return matrix

    def inverse(self) -> "SE3":
        rotation = self.rotation.T
        translation = -rotation @ self.translation
        return SE3(rotation, translation)

    def __mul__(self, other: "SE3") -> "SE3":
        rotation = self.rotation @ other.rotation
        translation = self.rotation @ other.translation + self.translation
        return SE3(rotation, translation)

    def __str__(self) -> str:
        return str(self.homogeneous)


def install_pinocchio_compat() -> None:
    module = types.ModuleType("pinocchio")
    module.SE3 = SE3
    module.Quaternion = Quaternion
    module.seed = np.random.seed
    sys.modules["pinocchio"] = module
