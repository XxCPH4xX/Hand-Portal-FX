"""Scale-independent thumb/index pinch detection with a release latch."""

import numpy as np

import config


def pinch_ratio(points: np.ndarray) -> float | None:
    """Measure fingertip separation relative to palm size in image pixels."""
    if points.shape != (21, 2) or not np.isfinite(points).all():
        return None
    palm = max(np.linalg.norm(points[9] - points[0]),
               np.linalg.norm(points[17] - points[5]))
    if palm < 5:
        return None
    return float(np.linalg.norm(points[4] - points[8]) / palm)


class PinchSwitch:
    def __init__(self, dwell: float = config.GESTURE_DWELL,
                 cooldown: float = config.GESTURE_COOLDOWN,
                 release: float = config.GESTURE_RELEASE):
        self.dwell = dwell
        self.cooldown = cooldown
        self.release = release
        self.reset()

    def reset(self) -> None:
        self.started = None
        self.released = None
        self.latched = False
        self.pinching = False
        self.required_hands = 0
        self.last_switch = float("-inf")
        self.progress = 0.0

    def update(self, hands: list[np.ndarray], now: float) -> bool:
        ratios = [ratio for points in hands if (ratio := pinch_ratio(points)) is not None]
        self.progress = 0.0
        if not ratios:
            # Occlusion is not a deliberate release.
            self.started = self.released = None
            self.pinching = False
            return False
        touching = min(ratios) <= config.PINCH_CLOSE_RATIO
        separated = min(ratios) >= config.PINCH_RELEASE_RATIO
        # Separate thresholds prevent fingertip jitter from flickering the state.
        self.pinching = touching or (self.pinching and not separated)
        if separated:
            self.started = None
            if len(ratios) < self.required_hands:
                self.released = None
                return False
            if self.released is None:
                self.released = now
            if now - self.released >= self.release:
                self.latched = False
                self.required_hands = 0
            return False
        self.released = None
        if self.latched:
            return False
        if not touching:
            self.started = None
            return False
        if self.started is None:
            self.started = now
        self.progress = min(1.0, (now - self.started) / self.dwell)
        if now - self.started < self.dwell:
            return False
        self.latched = True
        self.required_hands = len(ratios)
        self.started = None
        # Consume pinches inside cooldown so they cannot switch later while held.
        if now - self.last_switch < self.cooldown:
            return False
        self.last_switch = now
        return True
