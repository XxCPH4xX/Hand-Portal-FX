import unittest

import cv2
import numpy as np

from filters import (
    EFFECTS, GEOMETRIC_COLORS, comic_halftone, inversion, monochrome, neon_edges,
    pixel_art, rgb_glitch, sepia, vhs,
)


class FilterTests(unittest.TestCase):
    def test_contract_and_input_preserved(self):
        rng = np.random.default_rng(3)
        for shape in ((1, 1, 3), (1, 9, 3), (11, 1, 3), (48, 64, 3)):
            frame = rng.integers(0, 256, shape, dtype=np.uint8)
            original = frame.copy()
            for name, effect in EFFECTS:
                with self.subTest(effect=name, shape=shape):
                    output = effect(frame, 1.2)
                    self.assertEqual(output.shape, shape)
                    self.assertEqual(output.dtype, np.uint8)
                    np.testing.assert_array_equal(frame, original)

    def test_inversion_and_bgr_sepia(self):
        colors = np.array([[[255, 0, 0], [0, 0, 255], [255, 255, 255]]], np.uint8)
        np.testing.assert_array_equal(inversion(colors), 255 - colors)
        np.testing.assert_allclose(sepia(colors)[0, 0], [33, 43, 48], atol=1)
        np.testing.assert_allclose(sepia(colors)[0, 1], [69, 89, 100], atol=1)

    def test_gray_channels_match_luminance(self):
        frame = np.array([[[20, 100, 230]]], dtype=np.uint8)
        out = monochrome(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for channel in range(3):
            np.testing.assert_array_equal(out[:, :, channel], gray)

    def test_glitch_is_time_controlled(self):
        frame = np.random.default_rng(8).integers(0, 256, (80, 100, 3), dtype=np.uint8)
        np.testing.assert_array_equal(rgb_glitch(frame, 2), rgb_glitch(frame, 2))
        self.assertFalse(np.array_equal(rgb_glitch(frame, 2), rgb_glitch(frame, 3)))

    def test_pixel_art_has_flat_blocks_and_limited_palette(self):
        frame = np.random.default_rng(4).integers(0, 256, (32, 48, 3), dtype=np.uint8)
        output = pixel_art(frame)
        for y in range(0, 32, 8):
            for x in range(0, 48, 8):
                block = output[y:y + 8, x:x + 8]
                self.assertTrue(np.all(block == block[0, 0]))
        self.assertTrue(np.all(output % 32 == 16))

    def test_neon_highlights_edges_without_lighting_flat_black(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        np.testing.assert_array_equal(neon_edges(frame), frame)
        frame[:, 32:] = 180
        output = neon_edges(frame)
        self.assertGreater(output[:, 30:34].mean(), output[:, 50:60].mean() * 3)

    def test_halftone_puts_more_ink_in_shadows(self):
        dark = comic_halftone(np.full((60, 60, 3), 40, dtype=np.uint8))
        bright = comic_halftone(np.full((60, 60, 3), 220, dtype=np.uint8))
        ink = np.array([35, 25, 40], dtype=np.uint8)
        self.assertGreater(np.all(dark == ink, axis=2).sum(),
                           np.all(bright == ink, axis=2).sum())

    def test_vhs_is_time_controlled_without_large_frame_flashes(self):
        frame = np.full((48, 64, 3), 128, dtype=np.uint8)
        first = vhs(frame, 1)
        np.testing.assert_array_equal(first, vhs(frame, 1))
        second = vhs(frame, 1.05)
        self.assertFalse(np.array_equal(first, second))
        self.assertLess(np.abs(first.astype(float) - second).mean(), 3)

    def test_geometric_designs_draw_on_blank_frames_and_animate(self):
        frame = np.zeros((200, 240, 3), dtype=np.uint8)
        for effect in GEOMETRIC_COLORS:
            with self.subTest(effect=effect.__name__):
                first = effect(frame, 0)
                self.assertGreater(np.count_nonzero(first), 500)
                np.testing.assert_array_equal(first, effect(frame, 0))
                self.assertFalse(np.array_equal(first, effect(frame, 2)))

    def test_geometric_design_follows_translated_corners(self):
        frame = np.zeros((240, 300, 3), dtype=np.uint8)
        corners = np.array([[40, 40], [180, 40], [180, 180], [40, 180]], np.float32)
        for effect in GEOMETRIC_COLORS:
            first = effect(frame, 0, corners=corners)
            moved = effect(frame, 0, corners=corners + [30, 0])
            np.testing.assert_array_equal(first[:, :-30], moved[:, 30:])


if __name__ == "__main__":
    unittest.main()

