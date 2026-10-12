from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions, RunningMode,
)

import config


@dataclass
class Hand:
    points: np.ndarray
    handedness: str


class HandTracker:
    def __init__(self):
        if not config.MODEL_PATH.is_file():
            raise RuntimeError("Hand model missing. Run: python download_model.py")
        self.detector = HandLandmarker.create_from_options(HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=str(config.MODEL_PATH),
                delegate=BaseOptions.Delegate.CPU,
            ),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=config.HAND_CONFIDENCE,
            min_hand_presence_confidence=config.HAND_CONFIDENCE,
            min_tracking_confidence=config.HAND_CONFIDENCE,
        ))
        self.timestamp = -1

    def detect(self, frame: np.ndarray, now: float) -> list[Hand]:
        height, width = frame.shape[:2]
        scale = min(1.0, config.DETECTION_WIDTH / width)
        small = cv2.resize(frame, (round(width * scale), round(height * scale)),
                           interpolation=cv2.INTER_AREA) if scale < 1 else frame
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.timestamp = max(self.timestamp + 1, int(now * 1000))
        result = self.detector.detect_for_video(image, self.timestamp)
        hands = []
        for landmarks, labels in zip(result.hand_landmarks, result.handedness):
            points = np.array([(p.x * width, p.y * height) for p in landmarks],
                              dtype=np.float32)
            hands.append(Hand(points, labels[0].category_name if labels else ""))
        return hands

    def close(self) -> None:
        self.detector.close()

