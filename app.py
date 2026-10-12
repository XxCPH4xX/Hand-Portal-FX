import argparse
import sys
import time

import cv2
import numpy as np

import config
from filters import EFFECTS
from gestures import PinchSwitch
from hand_tracking import HandTracker
from portal import PortalState, render_portal

WINDOW = "Hand Portal FX"
CAMERA_HELP = (
    "Check System Settings > Privacy & Security > Camera and enable the app "
    "launching Python (Terminal, VS Code, or Codex). Permission for one app "
    "does not authorize the others. Quit and reopen that app after granting "
    "access if needed. Close other camera apps, then retry. "
    "Use --camera INDEX to select another camera."
)


def open_camera(index: int, width: int, height: int):
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    camera = cv2.VideoCapture(index, backend)
    if not camera.isOpened():
        camera.release()
        if sys.platform == "darwin" and sys.stdin.isatty():
            # AVFoundation requests permission asynchronously on the first open.
            print("Camera unavailable. If macOS asks for camera access, choose Allow.\n"
                  "Otherwise enable your terminal app in System Settings > "
                  "Privacy & Security > Camera.")
            try:
                input("After allowing access, press Enter to retry, or Ctrl-C to quit: ")
            except EOFError:
                raise RuntimeError(f"Could not open camera {index}. {CAMERA_HELP}") from None
            camera = cv2.VideoCapture(index, backend)
            if not camera.isOpened():
                camera.release()
                raise RuntimeError(f"Could not open camera {index}. {CAMERA_HELP}")
        else:
            raise RuntimeError(f"Could not open camera {index}. {CAMERA_HELP}")
    try:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Best effort; backend may ignore it.
    except Exception:
        camera.release()
        raise
    return camera


def draw_text(frame, text, position, scale=0.5, color=(235, 235, 235)):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (12, 12, 20), 3, cv2.LINE_AA)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


def run(args) -> None:
    camera = tracker = None
    try:
        width, height = (640, 480) if args.low_res else (config.FRAME_WIDTH, config.FRAME_HEIGHT)
        camera = open_camera(args.camera, width, height)
        tracker = HandTracker()
        portal, gesture = PortalState(), PinchSwitch()
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, width, height)
        effect_index, show_help = 0, True
        changed_until = time.monotonic() + 2
        feedback, feedback_until = "", 0.0
        fps, previous = 0.0, time.monotonic()
        failures, frame_count = 0, 0
        reported_size = False
        while True:
            ok, frame = camera.read()
            now = time.monotonic()
            if not ok or frame is None or frame.size == 0:
                failures += 1
                if failures >= 30:
                    raise RuntimeError(f"Camera stopped returning frames. {CAMERA_HELP}")
                if cv2.waitKey(20) & 0xFF in (27, ord("q"), ord("Q")):
                    break
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                continue
            failures = 0
            if not reported_size:
                actual_h, actual_w = frame.shape[:2]
                print(f"Camera {args.camera}: {actual_w} x {actual_h}; press Q or Esc to quit.")
                if (actual_w, actual_h) != (width, height):
                    print(f"Requested {width} x {height}; fitting frames within that size.")
                reported_size = True
            scale = min(1.0, width / frame.shape[1], height / frame.shape[0])
            if scale < 1:
                frame = cv2.resize(frame, (round(frame.shape[1] * scale),
                                          round(frame.shape[0] * scale)),
                                   interpolation=cv2.INTER_AREA)
            frame = cv2.flip(frame, 1)
            hands = tracker.detect(frame, now)
            if gesture.update([hand.points for hand in hands], now):
                effect_index = (effect_index + 1) % len(EFFECTS)
                changed_until = feedback_until = now + 1.5
                feedback = "Switched - separate fingertips to re-arm"
            points = None
            if len(hands) == 2 and not gesture.pinching:
                points = np.concatenate([hand.points[[4, 8]] for hand in hands])
                # Clamp only portal coordinates; pinch detection uses raw geometry.
                points = np.clip(points, [0, 0], [frame.shape[1] - 1, frame.shape[0] - 1])
            corners, opacity = portal.update(points, now)
            name, effect = EFFECTS[effect_index]
            output = frame
            if corners is not None:
                output = render_portal(frame, corners, opacity, effect, now)
            if config.SHOW_LANDMARKS:
                for hand in hands:
                    for point in hand.points:
                        cv2.circle(output, tuple(point.astype(int)), 2, (100, 230, 140), -1)
            elapsed = now - previous
            if elapsed > 0:
                fps = 0.9 * fps + 0.1 / elapsed if fps else 1 / elapsed
            previous = now
            draw_text(output, f"{effect_index + 1:02d} / {len(EFFECTS):02d}  {name}", (20, 32),
                      0.7 if now < changed_until else 0.55, (255, 220, 160))
            if args.fps or config.SHOW_FPS:
                draw_text(output, f"{fps:.0f} FPS", (output.shape[1] - 100, 30))
            if now < feedback_until:
                draw_text(output, feedback, (20, 60))
            elif gesture.pinching:
                message = ("Separate fingertips to re-arm" if gesture.latched
                           else "Pinch thumb + index to switch")
                draw_text(output, message, (20, 60))
                if gesture.progress > 0:
                    cv2.line(output, (20, 70), (20 + int(180 * gesture.progress), 70),
                             (255, 220, 160), 3, cv2.LINE_AA)
            if show_help:
                bottom = output.shape[0] - 18
                draw_text(output, "Two hands: spread thumbs + index fingers to shape the portal",
                          (20, bottom - 25), 0.45)
                draw_text(output,
                          "Pinch either hand: next | N/P: filters  R: reset  H: help  Q/Esc: quit",
                          (20, bottom), 0.45)
            cv2.imshow(WINDOW, output)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("n"), ord("N"), ord("p"), ord("P")):
                step = 1 if key in (ord("n"), ord("N")) else -1
                effect_index = (effect_index + step) % len(EFFECTS)
                changed_until = now + 1.5
            elif key in (ord("r"), ord("R")):
                portal.reset()
                gesture.reset()
                tracker.close()
                tracker = None
                tracker = HandTracker()
                feedback, feedback_until = "Tracking reset", now + 1.5
            elif key in (ord("h"), ord("H")):
                show_help = not show_help
            frame_count += 1
            if args.frames and frame_count >= args.frames:
                break
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if camera is not None:
            camera.release()
        try:
            if tracker is not None:
                tracker.close()
        finally:
            cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description="A local, hand-controlled camera portal.")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    parser.add_argument("--low-res", action="store_true", help="Request 640 x 480 capture")
    parser.add_argument("--fps", action="store_true", help="Show measured loop FPS")
    parser.add_argument("--frames", type=int, default=0,
                        help="Exit after N frames (startup check)")
    args = parser.parse_args()
    if args.frames < 0 or args.camera < 0:
        parser.error("--frames and --camera must be nonnegative")
    try:
        run(args)
    except KeyboardInterrupt:
        pass
    except (RuntimeError, OSError, ValueError, cv2.error) as error:
        print(f"Hand Portal FX: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

