import itertools
import unittest

import cv2
import numpy as np

from filters import GEOMETRIC_COLORS, inversion
from portal import PortalState, order_quad, render_portal

QUAD = np.array([[20, 20], [180, 25], [170, 120], [25, 110]], np.float32)


class GeometryTests(unittest.TestCase):
    def test_order_ignores_hand_and_point_order(self):
        expected = order_quad(QUAD)
        for permutation in itertools.permutations(QUAD):
            actual = order_quad(permutation)
            np.testing.assert_array_equal(actual, expected)
            self.assertTrue(cv2.isContourConvex(actual))

    def test_degenerate_concave_and_invalid_inputs(self):
        for points in (
            [[0, 0], [10, 0], [20, 0], [30, 0]],
            [[0, 0], [100, 0], [0, 100], [10, 10]],
            [[0, 0], [100, 0], [0, 100], [0, 100]],
            [[0, 0], [1, 0], [1, 1], [0, 1]],
            [[0, 0], [100, 0], [0, 100], [np.nan, 100]],
        ):
            self.assertIsNone(order_quad(points))

    def test_rotation_translation_and_crossing_order(self):
        state = PortalState()
        for i, angle in enumerate(np.linspace(0, 2 * np.pi, 80)):
            rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]])
            points = (QUAD - QUAD.mean(axis=0)) @ rotation + [300 + i, 200]
            if i % 2:
                points = points[[2, 3, 0, 1]]
            corners, opacity = state.update(points, i / 30)
            self.assertTrue(cv2.isContourConvex(corners))
            self.assertGreater(cv2.contourArea(corners), 700)
            self.assertEqual(opacity, 1)

    def test_missing_hand_hold_fade_and_recovery(self):
        state = PortalState(hold=0.2, fade=0.4)
        corners, _ = state.update(QUAD, 0)
        held, opacity = state.update(None, 0.1)
        np.testing.assert_array_equal(corners, held)
        self.assertEqual(opacity, 1)
        _, opacity = state.update(None, 0.4)
        self.assertAlmostEqual(opacity, 0.5)
        self.assertEqual(state.update(None, 0.7), (None, 0.0))
        recovered, opacity = state.update(QUAD + 50, 0.8)
        np.testing.assert_array_equal(recovered, order_quad(QUAD + 50))
        self.assertEqual(opacity, 1)

    def test_smoothing_moves_partway_and_reset_clears(self):
        state = PortalState(smoothing=0.5)
        original, _ = state.update(QUAD, 0)
        moved, _ = state.update(QUAD + 10, 1 / 30)
        np.testing.assert_allclose(moved, original + 5)
        state.reset()
        self.assertEqual(state.update(None, 1), (None, 0.0))

    def test_composite_only_changes_inside_mask(self):
        frame = np.full((160, 220, 3), 30, dtype=np.uint8)
        result = render_portal(frame, QUAD, 1, inversion, 0, decorate=False)
        np.testing.assert_array_equal(result[60, 90], [225] * 3)
        mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, QUAD.astype(np.int32), 255, cv2.LINE_AA)
        np.testing.assert_array_equal(result[mask == 0], frame[mask == 0])
        np.testing.assert_array_equal(frame, np.full_like(frame, 30))

    def test_extreme_offscreen_and_zero_opacity(self):
        frame = np.zeros((100, 120, 3), np.uint8)
        for points in (QUAD * 100, QUAD - 1000, QUAD + 1000):
            out = render_portal(frame, points, 1, inversion, 0)
            self.assertEqual(out.shape, frame.shape)
        np.testing.assert_array_equal(render_portal(frame, QUAD, 0, inversion, 0), frame)

    def test_geometric_effects_follow_rotated_portal_and_stay_inside(self):
        frame = np.full((160, 220, 3), 30, dtype=np.uint8)
        diamond = np.array([[110, 10], [205, 80], [110, 145], [15, 80]], np.float32)
        mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, diamond.astype(np.int32), 255, cv2.LINE_AA)
        for effect in GEOMETRIC_COLORS:
            actual = render_portal(frame, diamond, 1, effect, 0, decorate=False)
            np.testing.assert_array_equal(actual[mask == 0], frame[mask == 0])
            # Portal-local projection must agree with projection on the full frame.
            expected = effect(frame, 0, corners=diamond)
            interior = cv2.erode(mask, np.ones((9, 9), np.uint8)) == 255
            np.testing.assert_allclose(actual[interior], expected[interior], atol=1)


if __name__ == "__main__":
    unittest.main()

