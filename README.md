# Hand Portal FX

A local Python camera experiment: use two hands to frame a glowing portal, with a different camera effect inside it.

The portal follows the thumb and index fingertips of both hands. A convex hull orders the four corners, while a time-adjusted moving average reduces jitter. Spread your thumbs and index fingers. Move, rotate, or separate your hands to reshape the portal. Touch the thumb and index fingertips of either hand together to advance the effect; separate them before pinching again.

- Sixteen effects: thermal, pencil sketch, RGB glitch, monochrome, pink/yellow dual-tone, inversion, sepia, blur, cartoon, neon edges, pixel art, comic halftone, VHS, mystic seal, wireframe grid, and astral lattice.
- Feathered interior edges, a cyan glow, and small moving boundary particles.
- Brief hold and fade when tracking is lost or a switching gesture is shown.
- Mirrored preview, keyboard fallback, optional FPS and landmark displays.
- CPU inference and local camera processing. No camera images are saved or sent anywhere.

## Setup on Apple Silicon

Use native ARM64 Python 3.12 (Python 3.11 is also supported by the selected wheels), on macOS 13 or later. Validation here used Python 3.12.14 on an Apple M2. Do not use macOS's `/usr/bin/python3` if it still reports Python 3.9.

With an existing native Python 3.12 installation:

```bash
cd /Users/fahimshahrier/HandGestureFilter
python3.12 -c "import platform; print(platform.machine())"
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python download_model.py
python app.py
```

The architecture check should print `arm64`. An existing Python 3.11 installation can replace `python3.12` in those commands. No Conda environment or global Python packages need to change.

This checkout already has a working `.venv`, created with Codex's bundled ARM64 Python 3.12.14, and the downloaded model. To use it now:

```bash
cd /Users/fahimshahrier/HandGestureFilter
source .venv/bin/activate
python app.py
```

The local environment depends on that bundled interpreter staying installed; recreate it with your own Python 3.12 for long-term use. In VS Code, select `.venv/bin/python` as the interpreter.

Dependencies are pinned to MediaPipe 0.10.32, OpenCV contrib 4.11.0.86, and NumPy 2.2.6, which publish native Apple Silicon wheels. MediaPipe requires the contrib distribution of OpenCV and brings in Matplotlib and audio support as transitive dependencies; this application does not use audio. Avoid installing `opencv-python` or a headless OpenCV package alongside `opencv-contrib-python`, since they share the `cv2` module.

The one-time download fetches Google's approximately 8 MB [Hand Landmarker model](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker) and verifies its SHA-256 before installing it. The application does not download anything at startup and runs offline afterward. The model and virtual environment are excluded from Git.

## Run and controls

```bash
python app.py --fps
python app.py --low-res           # request 640 x 480
python app.py --camera 1          # explicitly select this Mac's FaceTime HD camera
python app.py --frames 120        # bounded startup check
```

| Control | Action |
| --- | --- |
| Thumb–index pinch on either hand | Next effect |
| N / P | Next / previous effect |
| R | Reset portal, gesture state, and hand tracker |
| H | Toggle instructions |
| Q / Esc / window close | Quit and release the camera |

After Blur: **Cartoon** adds flat colors and ink outlines; **Neon Edges** lights contours in cyan and magenta; **Pixel Art** uses chunky blocks and reduced colors; **Comic Halftone** adds print dots; **VHS** adds soft color bleed, scanlines, and subtle moving tape interference.

Three animated geometric looks follow VHS: **Mystic Seal** layers amber polygon rings, rotating squares, and small angular marks; **Wireframe Grid** creates a cyan lattice with moving scan lines; **Astral Lattice** connects rotating violet hexagons with straight lines. These original, Doctor Strange–inspired designs follow the portal's perspective and rotation, with matching border colors. The camera remains visible beneath the lines. Pinch or press N/P to cycle through all sixteen. From the initial Thermal filter, press P three times to reach Mystic Seal quickly.

Keep both hands in view, with some space between each thumb and index fingertip. Touch those fingertips together on either hand to switch once (about 0.08 seconds of stable contact). A pinch briefly holds the portal in its last shape before fading, so it does not collapse into a thin triangle. Separate the fingertips for 0.12 seconds before pinching again; holding a pinch does not repeatedly switch. If both hands were visible when you pinched, both must be visible with fingertips separated to rearm. This prevents a disappearing pinched hand from looking like a release. Pinching also works with just one hand visible. Simultaneous pinches advance one effect, and a 0.35-second cooldown prevents rapid accidental changes.

On this Mac, index `0` is **OBS Virtual Camera** and index `1` is **FaceTime HD Camera**, so `config.py` defaults to `1`. An OBS logo or placeholder means you selected its virtual camera. Camera indices can change when devices are added or removed; use `--camera INDEX` to override the default.

## Camera permissions

Enable the app launching Python under **System Settings → Privacy & Security → Camera**. This may be Terminal, Visual Studio Code, or Codex. Restart that app after granting permission if necessary. Close FaceTime, Zoom, or other software using the camera. The app prefers AVFoundation on macOS and reports the actual camera resolution when the requested size is unavailable. Larger camera frames are downscaled to fit the selected size while preserving aspect ratio (for example, 1920 × 1080 becomes 640 × 360 in low-resolution mode).

Temporary read failures are retried up to 30 times. Persistent failure exits with a short message. Camera, tracker, and OpenCV windows are closed on normal exit, Ctrl-C, or errors.

On the first launch, OpenCV can report `not authorized ... (status 0)` immediately after requesting permission. In an interactive terminal, the application now waits for you to choose **Allow** and press Enter, then retries once. If access remains unavailable, it exits with instructions. Permission granted to Codex does not grant permission to Terminal or VS Code: enable the app you are actually using. Noninteractive launches exit without waiting for input.

## Implementation notes

`hand_tracking.py` uses MediaPipe's [Hand Landmarker Tasks API](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python) in video mode, with the CPU delegate and at most two hands. The already-mirrored frame is resized to 640 pixels wide for inference; normalized landmarks map back to the full preview. Handedness is retained but does not determine portal corner order.

`portal.py` rejects degenerate shapes and any four points that cannot form a convex quadrilateral. Cyclic corner matching makes ordering independent of detection order and handedness changes. Missing geometry holds for 0.25 seconds and fades over another 0.35 seconds. Filtering, feathering, glow, and particles are restricted to a padded portal bounding box. Geometric effects project normalized line designs onto the four tracked corners while leaving the camera image unwarped. Only the decorative glow extends outside the portal mask.

`gestures.py` measures thumb–index distance relative to palm size. Separate contact and release thresholds, a short debounce, cooldown, and a release latch prevent repeated switching. Handedness and detection order do not determine which hand can switch. `filters.py` contains independent BGR uint8 image functions. Adjust resolution, confidence, smoothing, gesture timing, particles, and optional landmarks in `config.py`.

The camera loop is synchronous and requests a one-frame capture buffer where supported. AVFoundation can ignore that buffer setting. There is no application frame queue or background capture thread. The thermal effect maps brightness to a palette; it does not measure temperature.

## Validation and limitations

```bash
python -m unittest discover -s tests -v
python -m pip check
```

The test suite covers filter shape/type and input preservation (including tiny images), BGR sepia, deterministic glitch timing, corner permutations and rotation, degenerate polygons, smoothing, missing-hand fading, compositing boundaries, gesture dwell/release/cooldown, the macOS permission retry, and camera cleanup after failures. Tests do not validate real camera interaction. CPU inference on Google's sample photograph returned two hands with 21 landmarks each; `pip check` found no broken requirements.

Initial startup checks captured and displayed frames from index `0`, later identified as OBS Virtual Camera, so those checks did not validate the physical camera feed. The default is now index `1`, this Mac's FaceTime HD camera; a subsequent 10-frame application run captured 1280 × 720 frames and exited successfully. Live two-hand gestures, perceived latency, keyboard interaction, and sustained end-to-end FPS still require a manual check:

1. Enable camera permission and run `python app.py --fps`. Confirm a mirrored preview.
2. Make a portal pose with both hands; translate, rotate, and resize it. Cross hands briefly and return them to view.
3. Hide one hand briefly, then for a second. Check hold, fade, and recovery.
4. Touch thumb and index fingertips together on the left hand, then test the right hand. Each pinch should switch once. Hold the pinch to check it does not repeat, then separate and pinch again.
5. Use N/P to inspect every effect, R to reset, and H to hide instructions.
6. Quit with Q, Esc, and the close button in separate runs. Confirm the camera indicator turns off.

The target is 24–30 FPS at 720p, not a verified live measurement. Try `--low-res` if needed. Synthetic 720p benchmarks with a roughly 640 × 420 portal measured about 15–22 ms per render across the original eight effects, 17–23 ms across the five image effects, and 24–26 ms across the geometric effects on this M2, excluding inference, capture, and display.

Low light, motion blur, fingers pointing toward the camera, and overlapping hands can reduce tracking quality. Pinch thresholds in `config.py` may need adjustment for your hand pose; increase `PINCH_CLOSE_RATIO` slightly if contact is missed, or decrease it if switching happens too early. If one corner falls inside the other three, the portal holds/fades until all four form a valid boundary again. Filtering a changing bounding box can slightly change blur and glitch appearance near edges.

## Inspiration

The interaction is inspired by this [reference demonstration](https://youtube.com/shorts/XqYy4Qovh5c) and the linked [python-handtrack project](https://github.com/OdekkNanda/python-handtrack). This project implements its own geometry, effects, gesture rules, and rendering pipeline; source from that repository was not used.
