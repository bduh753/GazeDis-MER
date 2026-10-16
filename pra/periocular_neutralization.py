from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LandmarkRegionSpec:
    """Landmark indices defining alignment anchors and two periocular regions."""

    left_eye: tuple[int, ...]
    right_eye: tuple[int, ...]
    regions: tuple[tuple[int, ...], ...]
    anchors: tuple[int, ...]


def _validate_inputs(
    onset: np.ndarray,
    apex: np.ndarray,
    onset_landmarks: np.ndarray,
    apex_landmarks: np.ndarray,
    spec: LandmarkRegionSpec,
) -> None:
    if onset.shape != apex.shape:
        raise ValueError(f"Onset and apex shapes differ: {onset.shape} != {apex.shape}")
    if onset.ndim != 3 or onset.shape[2] != 3:
        raise ValueError(f"Expected RGB HWC images, got {onset.shape}")
    for name, landmarks in (
        ("onset_landmarks", onset_landmarks),
        ("apex_landmarks", apex_landmarks),
    ):
        if landmarks.ndim != 2 or landmarks.shape[1] != 2:
            raise ValueError(f"{name} must have shape (count, 2), got {landmarks.shape}")
    required = spec.left_eye + spec.right_eye + spec.anchors + tuple(
        index for region in spec.regions for index in region
    )
    if not required:
        raise ValueError("The landmark region specification is empty")
    if min(required) < 0:
        raise ValueError("Landmark indices must be non-negative")
    if max(required) >= min(len(onset_landmarks), len(apex_landmarks)):
        raise ValueError("The landmark region specification references unavailable landmarks")


def neutralize_periocular_motion(
    onset: np.ndarray,
    apex: np.ndarray,
    onset_landmarks: np.ndarray,
    apex_landmarks: np.ndarray,
    spec: LandmarkRegionSpec,
    margin_ratio: float = 0.10,
    feather_ratio: float = 0.04,
    strength: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace the apex periocular content with aligned onset content.

    Images must be RGB uint8 arrays. Landmark coordinates are pixel coordinates.
    The returned tuple contains the neutralized apex and the soft replacement mask.
    """

    _validate_inputs(onset, apex, onset_landmarks, apex_landmarks, spec)
    if margin_ratio < 0.0 or feather_ratio < 0.0:
        raise ValueError("margin_ratio and feather_ratio must be non-negative")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")

    anchors = np.asarray(spec.anchors, dtype=np.int32)
    transform, _ = cv2.estimateAffinePartial2D(
        onset_landmarks[anchors].astype(np.float32),
        apex_landmarks[anchors].astype(np.float32),
        method=cv2.LMEDS,
    )
    if transform is None:
        raise RuntimeError("Could not align onset landmarks to apex landmarks")

    height, width = apex.shape[:2]
    aligned_onset = cv2.warpAffine(
        onset,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    aligned_landmarks = cv2.transform(
        onset_landmarks.astype(np.float32)[None, :, :], transform
    )[0]

    left_center = apex_landmarks[np.asarray(spec.left_eye)].mean(axis=0)
    right_center = apex_landmarks[np.asarray(spec.right_eye)].mean(axis=0)
    interocular_distance = float(np.linalg.norm(left_center - right_center))
    margin = max(0, int(round(interocular_distance * margin_ratio)))
    feather = max(0, int(round(interocular_distance * feather_ratio)))

    mask = np.zeros((height, width), dtype=np.uint8)
    for region in spec.regions:
        indices = np.asarray(region, dtype=np.int32)
        points = np.concatenate(
            (apex_landmarks[indices], aligned_landmarks[indices]), axis=0
        )
        hull = cv2.convexHull(np.rint(points).astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 255)

    if margin:
        kernel_size = 2 * margin + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.dilate(mask, kernel)
    if feather:
        blur_size = 2 * feather + 1
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

    alpha = strength * mask.astype(np.float32)[..., None] / 255.0
    neutralized = apex.astype(np.float32) * (1.0 - alpha)
    neutralized += aligned_onset.astype(np.float32) * alpha
    return np.clip(np.rint(neutralized), 0, 255).astype(np.uint8), mask

