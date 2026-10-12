import unittest

import numpy as np

from gestures import PinchSwitch, pinch_ratio


def hand_points(gap=80.0):
    points = np.zeros((21, 2), np.float32)
    points[0], points[9] = [100, 200], [100, 100]
    points[5], points[17] = [60, 130], [140, 130]
    points[8], points[4] = [70, 60], [70 + gap, 60]
    return points


class GestureTests(unittest.TestCase):
    def test_ratio_is_rotation_scale_and_mirror_independent(self):
        points = hand_points(10)
        for angle in np.linspace(0, 2 * np.pi, 16):
            rotation = np.array([[np.cos(angle), -np.sin(angle)],
                                 [np.sin(angle), np.cos(angle)]])
            self.assertAlmostEqual(pinch_ratio(points @ rotation * 0.6 + 100), 0.1)
        self.assertAlmostEqual(pinch_ratio(points * [-1, 1]), 0.1)

    def test_invalid_and_tiny_hands_do_not_switch(self):
        switch = PinchSwitch()
        invalid = [np.zeros((21, 2)), np.full((21, 2), np.nan), np.zeros((4, 2))]
        for points in invalid:
            self.assertIsNone(pinch_ratio(points))
        for now in (0, 1):
            self.assertFalse(switch.update(invalid, now))

    def test_separated_fingers_do_not_switch(self):
        switch = PinchSwitch()
        for now in (0, 0.5, 1):
            self.assertFalse(switch.update([hand_points()], now))
            self.assertFalse(switch.pinching)

    def test_either_hand_switches_once_regardless_of_detection_order(self):
        for hands in ([hand_points(10), hand_points()],
                      [hand_points(), hand_points(10)],
                      [hand_points(10)], [hand_points(10), hand_points(10)]):
            with self.subTest(hands=len(hands)):
                switch = PinchSwitch()
                self.assertFalse(switch.update(hands, 0))
                self.assertTrue(switch.update(hands[::-1], 0.09))
                self.assertFalse(switch.update(hands, 1))
                self.assertFalse(switch.update(hands, 5))

    def test_release_then_new_pinch_switches_again(self):
        switch = PinchSwitch()
        switch.update([hand_points(10)], 0)
        self.assertTrue(switch.update([hand_points(10)], 0.09))
        switch.update([hand_points()], 0.2)
        switch.update([hand_points()], 0.33)
        self.assertFalse(switch.update([hand_points(10)], 0.5))
        self.assertTrue(switch.update([hand_points(10)], 0.59))

    def test_jitter_between_thresholds_and_brief_release_do_not_rearm(self):
        switch = PinchSwitch()
        switch.update([hand_points(10)], 0)
        self.assertTrue(switch.update([hand_points(10)], 0.09))
        for now in (0.5, 0.8, 1):
            self.assertFalse(switch.update([hand_points(30)], now))
            self.assertTrue(switch.pinching)
        switch.update([hand_points()], 1.1)
        self.assertFalse(switch.update([hand_points(10)], 1.15))
        self.assertFalse(switch.update([hand_points(10)], 2))

    def test_hidden_pinched_hand_cannot_be_replaced_by_remaining_open_hand(self):
        switch = PinchSwitch()
        hands = [hand_points(10), hand_points()]
        switch.update(hands, 0)
        self.assertTrue(switch.update(hands, 0.09))
        for now in (0.5, 1):
            self.assertFalse(switch.update([hand_points()], now))
        self.assertFalse(switch.update(hands, 2))
        self.assertFalse(switch.update([], 3))
        self.assertFalse(switch.update(hands, 4))
        switch.update([hand_points(), hand_points()], 5)
        switch.update([hand_points(), hand_points()], 5.13)
        switch.update(hands, 5.2)
        self.assertTrue(switch.update(hands, 5.3))

    def test_interrupted_or_very_short_contact_does_not_switch(self):
        switch = PinchSwitch()
        switch.update([hand_points(10)], 0)
        switch.update([], 0.05)
        self.assertFalse(switch.update([hand_points(10)], 0.2))
        self.assertFalse(switch.update([hand_points(10)], 0.24))
        switch.update([hand_points()], 0.25)
        self.assertFalse(switch.update([hand_points(10)], 0.5))
        self.assertTrue(switch.update([hand_points(10)], 0.59))

    def test_cooldown_consumes_contact_without_delayed_switch(self):
        switch = PinchSwitch(cooldown=1)
        switch.update([hand_points(10)], 0)
        self.assertTrue(switch.update([hand_points(10)], 0.09))
        switch.update([hand_points()], 0.2)
        switch.update([hand_points()], 0.33)
        switch.update([hand_points(10)], 0.4)
        self.assertFalse(switch.update([hand_points(10)], 0.49))
        self.assertFalse(switch.update([hand_points(10)], 2))
        switch.update([hand_points()], 2.1)
        switch.update([hand_points()], 2.23)
        switch.update([hand_points(10)], 2.4)
        self.assertTrue(switch.update([hand_points(10)], 2.49))


if __name__ == "__main__":
    unittest.main()
