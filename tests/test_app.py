import argparse
from contextlib import ExitStack
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import app


class LifecycleTests(unittest.TestCase):
    def test_camera_open_failure_releases_device(self):
        camera = MagicMock()
        camera.isOpened.return_value = False
        with (patch.object(app.cv2, "VideoCapture", return_value=camera),
              patch.object(app.sys.stdin, "isatty", return_value=False)):
            with self.assertRaisesRegex(RuntimeError, "Could not open camera"):
                app.open_camera(0, 1280, 720)
        camera.release.assert_called_once()

    def test_macos_permission_retry_opens_after_user_response(self):
        first, second = MagicMock(), MagicMock()
        first.isOpened.return_value = False
        second.isOpened.return_value = True
        with (patch.object(app.cv2, "VideoCapture", side_effect=[first, second]),
              patch.object(app.sys, "platform", "darwin"),
              patch.object(app.sys.stdin, "isatty", return_value=True),
              patch("builtins.input", return_value="") as prompt):
            camera = app.open_camera(0, 1280, 720)
        self.assertIs(camera, second)
        first.release.assert_called_once()
        second.release.assert_not_called()
        prompt.assert_called_once()

    def test_macos_permission_retry_is_bounded_and_releases_both_attempts(self):
        first, second = MagicMock(), MagicMock()
        first.isOpened.return_value = second.isOpened.return_value = False
        with (patch.object(app.cv2, "VideoCapture", side_effect=[first, second]),
              patch.object(app.sys, "platform", "darwin"),
              patch.object(app.sys.stdin, "isatty", return_value=True),
              patch("builtins.input", return_value="")):
            with self.assertRaisesRegex(RuntimeError, "Could not open camera"):
                app.open_camera(0, 1280, 720)
        first.release.assert_called_once()
        second.release.assert_called_once()

    def test_camera_failure_does_not_initialize_tracker(self):
        args = argparse.Namespace(camera=0, low_res=False)
        with (patch.object(app, "open_camera", side_effect=RuntimeError("No camera")),
              patch.object(app, "HandTracker") as tracker,
              patch.object(app.cv2, "destroyAllWindows")):
            with self.assertRaisesRegex(RuntimeError, "No camera"):
                app.run(args)
        tracker.assert_not_called()

    def run_mock_camera(self, reads, keys, frames=0):
        camera, tracker = MagicMock(), MagicMock()
        camera.read.side_effect = reads
        tracker.detect.return_value = []
        args = argparse.Namespace(camera=0, low_res=True, fps=False, frames=frames)
        patches = (
            patch.object(app, "open_camera", return_value=camera),
            patch.object(app, "HandTracker", return_value=tracker),
            patch.object(app.cv2, "namedWindow"),
            patch.object(app.cv2, "resizeWindow"),
            patch.object(app.cv2, "imshow"),
            patch.object(app.cv2, "waitKey", side_effect=keys),
            patch.object(app.cv2, "getWindowProperty", return_value=1),
            patch.object(app.cv2, "destroyAllWindows"),
        )
        with ExitStack() as stack:
            mocks = [stack.enter_context(p) for p in patches]
            try:
                app.run(args)
            finally:
                camera.release.assert_called_once()
                tracker.close.assert_called_once()
                mocks[-1].assert_called_once()
        return tracker

    def test_one_transient_failure_then_keyboard_exit(self):
        frame = np.zeros((1080, 1920, 3), np.uint8)
        frame[:, :960] = 255
        tracker = self.run_mock_camera([(False, None), (True, frame)], [-1, ord("q")])
        tracked = tracker.detect.call_args.args[0]
        self.assertEqual(tracked.shape, (360, 640, 3))
        self.assertEqual(int(tracked[0, 0, 0]), 0)
        self.assertEqual(int(tracked[0, -1, 0]), 255)

    def test_persistent_failure_closes_resources(self):
        with self.assertRaisesRegex(RuntimeError, "stopped returning frames"):
            self.run_mock_camera([(False, None)] * 30, [-1] * 29)


if __name__ == "__main__":
    unittest.main()
