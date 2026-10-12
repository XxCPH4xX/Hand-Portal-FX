"""Convex four-corner geometry, temporal smoothing, and local compositing."""

from collections.abc import Callable

import cv2
import numpy as np

import config
from filters import GEOMETRIC_COLORS


def order_quad(points: np.ndarray, min_area: float = config.MIN_PORTAL_AREA):
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return None
    hull = cv2.convexHull(points, clockwise=False).reshape(-1, 2)
    if len(hull) != 4 or cv2.contourArea(hull) < min_area:
        return None
    if np.min(np.linalg.norm(hull - np.roll(hull, 1, axis=0), axis=1)) < 5:
        return None
    # A hull rejects concavity and prevents crossed thumb/index connections.
    start = np.lexsort((hull[:, 0], hull[:, 1]))[0]
    return np.roll(hull, -start, axis=0)


class PortalState:
    def __init__(self, smoothing: float = config.PORTAL_SMOOTHING,
                 hold: float = config.TRACKING_HOLD,
                 fade: float = config.TRACKING_FADE):
        self.smoothing, self.hold, self.fade = smoothing, hold, fade
        self.reset()

    def reset(self) -> None:
        self.corners = None
        self.last_seen = None
        self.last_update = None

    def update(self, points, now: float):
        candidate = order_quad(points) if points is not None else None
        if candidate is not None:
            if self.corners is not None:
                # Match cyclic corners by distance, independent of handedness.
                choices = [np.roll(candidate, k, axis=0) for k in range(4)]
                candidate = min(choices, key=lambda p: np.sum((p - self.corners) ** 2))
                dt = max(0.001, now - self.last_update)
                weight = 1 - (1 - self.smoothing) ** (dt * 30)
                smoothed = self.corners + weight * (candidate - self.corners)
                if (cv2.isContourConvex(smoothed)
                        and cv2.contourArea(smoothed) >= config.MIN_PORTAL_AREA):
                    candidate = smoothed
            self.corners = candidate
            self.last_seen = now
        self.last_update = now
        if self.corners is None:
            return None, 0.0
        age = now - self.last_seen
        alpha = max(0.0, 1.0 - max(0.0, age - self.hold) / self.fade)
        if alpha == 0:
            self.corners = None
        return self.corners, alpha


def render_portal(frame: np.ndarray, corners: np.ndarray, opacity: float,
                  effect: Callable, now: float, decorate: bool = True) -> np.ndarray:
    """Filter a padded bounding box; leave all other camera pixels untouched."""
    output = frame.copy()
    if opacity <= 0:
        return output
    height, width = frame.shape[:2]
    padding = 24
    x0, y0 = np.maximum(np.floor(corners.min(axis=0)) - padding, 0).astype(int)
    x1, y1 = np.minimum(np.ceil(corners.max(axis=0)) + padding + 1,
                       [width, height]).astype(int)
    if x0 >= x1 or y0 >= y1:
        return output
    roi = frame[y0:y1, x0:x1]
    local = np.rint(corners - [x0, y0]).astype(np.int32)
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, local, 255, lineType=cv2.LINE_AA)
    feather = cv2.GaussianBlur(mask, (5, 5), 0).astype(np.float32) / 255
    feather *= mask.astype(np.float32) / 255
    alpha = (feather * opacity)[..., None]
    if effect in GEOMETRIC_COLORS:
        filtered = effect(roi, now, corners=corners - [x0, y0])
    else:
        filtered = effect(roi, now)
    blended = roi.astype(np.float32) * (1 - alpha) + filtered * alpha
    if decorate:
        light = np.zeros_like(roi)
        border_color = GEOMETRIC_COLORS.get(effect, (255, 185, 100))
        cv2.polylines(light, [local], True, border_color, 3, cv2.LINE_AA)
        glow = cv2.GaussianBlur(light, (0, 0), 5)
        blended += glow.astype(np.float32) * (config.GLOW_INTENSITY * 2 * opacity)
        # Blend the colored core so bright camera backgrounds cannot wash it out.
        core_alpha = (light.max(axis=2).astype(np.float32) / 255 * opacity)[..., None]
        core_color = np.array(border_color, dtype=np.float32)
        blended = blended * (1 - core_alpha) + core_color * core_alpha
        particles = np.zeros_like(roi)
        edges = np.roll(corners, -1, axis=0) - corners
        lengths = np.linalg.norm(edges, axis=1)
        cumulative = np.cumsum(lengths)
        perimeter = cumulative[-1]
        for i in range(config.PARTICLE_COUNT):
            distance = (i / config.PARTICLE_COUNT + now * 0.075) % 1 * perimeter
            edge = min(int(np.searchsorted(cumulative, distance)), 3)
            start = 0 if edge == 0 else cumulative[edge - 1]
            point = corners[edge] + edges[edge] * ((distance - start) / lengths[edge])
            point += 2 * np.array([np.sin(now * 2 + i), np.cos(now * 2 + i)])
            center = tuple(np.rint(point - [x0, y0]).astype(int))
            cv2.circle(particles, center, 1 + (i % 3 == 0), (255, 235, 210), -1,
                       cv2.LINE_AA)
        blended += particles.astype(np.float32) * opacity
    output[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
    return output
