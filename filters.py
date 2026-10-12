"""BGR uint8 effects. Inputs are never modified; time is in seconds."""

import cv2
import numpy as np


def thermal(frame: np.ndarray, t: float = 0.0) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


def pencil_sketch(frame: np.ndarray, t: float = 0.0) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(255 - gray, (0, 0), 3.0)
    sketch = cv2.divide(gray, np.maximum(255 - blurred, 1), scale=256)
    # A little original shading keeps uniformly dark areas from going black.
    sketch = cv2.addWeighted(sketch, 0.88, gray, 0.12, 15)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


def rgb_glitch(frame: np.ndarray, t: float = 0.0) -> np.ndarray:
    height, width = frame.shape[:2]
    shift = max(1, round(width * 0.012 * (0.7 + 0.3 * np.sin(t * 2))))
    out = frame.copy()
    out[:, :, 0] = np.roll(frame[:, :, 0], -shift, axis=1)
    out[:, :, 2] = np.roll(frame[:, :, 2], shift, axis=1)
    band = int((0.5 + 0.45 * np.sin(t * 0.8)) * (height - 1))
    end = min(height, band + max(1, height // 18))
    out[band:end] = np.roll(out[band:end], shift * 2, axis=1)
    out[::5] = (out[::5].astype(np.uint16) * 9 // 10).astype(np.uint8)
    return out


def monochrome(frame: np.ndarray, t: float = 0.0) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def dual_tone(frame: np.ndarray, t: float = 0.0) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    weight = np.clip((gray - 0.15) / 0.7, 0, 1)[..., None]
    pink = np.array([132, 25, 225], dtype=np.float32)
    yellow = np.array([65, 235, 255], dtype=np.float32)
    return (pink + weight * (yellow - pink)).astype(np.uint8)


def inversion(frame: np.ndarray, t: float = 0.0) -> np.ndarray:
    return cv2.bitwise_not(frame)


def sepia(frame: np.ndarray, t: float = 0.0) -> np.ndarray:
    # Both input and output use OpenCV's BGR channel order.
    matrix = np.array([[0.131, 0.534, 0.272],
                       [0.168, 0.686, 0.349],
                       [0.189, 0.769, 0.393]], dtype=np.float32)
    return cv2.transform(frame, matrix)


def blur(frame: np.ndarray, t: float = 0.0) -> np.ndarray:
    return cv2.GaussianBlur(frame, (19, 19), 0)


def cartoon(frame: np.ndarray, t: float = 0.0) -> np.ndarray:
    smooth = cv2.bilateralFilter(frame, 5, 55, 4)
    poster = (smooth // 32) * 32 + 16
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    return cv2.bitwise_and(poster, poster, mask=255 - edges)


def neon_edges(frame: np.ndarray, t: float = 0.0) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    strength = np.clip(cv2.magnitude(dx, dy) * 0.012, 0, 1)
    tone = gray.astype(np.float32) / 255
    # Dark contours glow cyan; brighter contours lean toward magenta.
    color = np.stack((np.full_like(tone, 255), 235 - 180 * tone,
                      70 + 185 * tone), axis=2)
    edges = color * strength[..., None]
    glow = cv2.GaussianBlur(edges, (0, 0), 2)
    return np.clip(frame * 0.10 + edges * 0.85 + glow * 0.6, 0, 255).astype(np.uint8)


def pixel_art(frame: np.ndarray, t: float = 0.0) -> np.ndarray:
    height, width = frame.shape[:2]
    small = cv2.resize(frame, (max(1, width // 8), max(1, height // 8)),
                       interpolation=cv2.INTER_AREA)
    palette = (small // 32) * 32 + 16
    return cv2.resize(palette, (width, height), interpolation=cv2.INTER_NEAREST)


def comic_halftone(frame: np.ndarray, t: float = 0.0) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    luminance = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32) / 255
    y, x = np.ogrid[:frame.shape[0], :frame.shape[1]]
    # Dot area grows with darkness, like printed ink on a six-pixel grid.
    distance_sq = ((x % 6) - 2.5) ** 2 + ((y % 6) - 2.5) ** 2
    ink = distance_sq < (1 - luminance) * (36 / np.pi)
    paper = (frame // 64) * 64 + 32
    out = cv2.addWeighted(paper, 0.8, np.full_like(frame, 245), 0.2, 0)
    out[ink | (cv2.Canny(gray, 80, 180) != 0)] = (35, 25, 40)
    return out


def vhs(frame: np.ndarray, t: float = 0.0) -> np.ndarray:
    height, width = frame.shape[:2]
    ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    chroma = cv2.GaussianBlur(ycc[:, :, 1:], (7, 3), 0)
    offset = int(round(2 * np.sin(t * 1.4)))
    columns = np.clip(np.arange(width) - offset, 0, width - 1)
    ycc[:, :, 1:] = chroma[:, columns]
    faded = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR).astype(np.float32) * 0.88 + 14
    rows = np.arange(height, dtype=np.float32)[:, None, None]
    scanlines = 1 - 0.07 * (rows % 3 == 0)
    # Slow tracking band and low-amplitude grain avoid random frame flashes.
    band_y = (t * 35) % (height + 40) - 20
    tracking = 5 * np.exp(-((rows - band_y) / 7) ** 2)
    columns = np.arange(width, dtype=np.float32)[None, :, None]
    grain = 2 * np.sin(columns * 1.7 + rows * 2.3 + t * 8)
    return np.clip(faded * scanlines + tracking + grain, 0, 255).astype(np.uint8)


def _polygon(sides: int, radius: float, angle: float) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, sides + 1) + angle
    return np.column_stack((np.cos(angles), np.sin(angles))) * radius


def _geometry_overlay(frame: np.ndarray, paths: list, color: tuple,
                      corners: np.ndarray | None) -> np.ndarray:
    """Project a line design onto the hand plane without warping the camera image."""
    height, width = frame.shape[:2]
    if corners is None:
        corners = np.array([[0, 0], [width - 1, 0],
                            [width - 1, height - 1], [0, height - 1]], np.float32)
    source = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float32)
    transform = cv2.getPerspectiveTransform(source, np.asarray(corners, np.float32))
    ink = np.zeros((height, width), np.uint8)
    thickness = max(1, min(height, width) // 260)
    for path in paths:
        points = cv2.perspectiveTransform(np.asarray(path, np.float32)[None], transform)
        cv2.polylines(ink, [np.rint(points[0]).astype(np.int32)], False,
                      255, thickness, cv2.LINE_AA)
    glow = cv2.GaussianBlur(ink, (0, 0), 2).astype(np.float32) / 255
    alpha = (ink.astype(np.float32) / 255 * 0.9)[..., None]
    tint = np.array(color, np.float32)
    background = frame.astype(np.float32) * 0.72
    output = background * (1 - alpha) + tint * alpha
    output += glow[..., None] * tint * 0.65
    return np.clip(output, 0, 255).astype(np.uint8)


def mystic_seal(frame: np.ndarray, t: float = 0.0,
                corners: np.ndarray | None = None) -> np.ndarray:
    paths = [_polygon(12, 0.93, t * 0.08), _polygon(12, 0.83, -t * 0.06),
             _polygon(4, 0.76, t * 0.16), _polygon(4, 0.76, -t * 0.12 + np.pi / 4),
             _polygon(6, 0.32, -t * 0.2)]
    for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False) + t * 0.08:
        direction = np.array([np.cos(angle), np.sin(angle)])
        tangent = np.array([-direction[1], direction[0]])
        paths.append([direction * 0.85, direction * 0.9])
        paths.append([direction * 0.8 - tangent * 0.025,
                      direction * 0.76, direction * 0.8 + tangent * 0.025])
    return _geometry_overlay(frame, paths, (45, 170, 255), corners)


def wireframe_grid(frame: np.ndarray, t: float = 0.0,
                   corners: np.ndarray | None = None) -> np.ndarray:
    paths = []
    for position in np.linspace(-1, 1, 9):
        paths.extend(([[-1, position], [1, position]],
                      [[position, -1], [position, 1]]))
    for position in np.linspace(-1, 1, 5):
        paths.extend(([[-1, position], [-position, 1]],
                      [[position, -1], [1, -position]]))
    for radius in (0.25, 0.5, 0.75):
        paths.append(_polygon(4, radius, np.pi / 4 + t * 0.05))
    sweep = np.sin(t * 0.6) * 0.9
    paths.extend(([[-1, sweep], [1, sweep]], [[sweep, -1], [sweep, 1]]))
    return _geometry_overlay(frame, paths, (255, 215, 80), corners)


def astral_lattice(frame: np.ndarray, t: float = 0.0,
                   corners: np.ndarray | None = None) -> np.ndarray:
    rings = [_polygon(6, radius, t * speed + np.pi / 6)
             for radius, speed in ((0.94, 0.09), (0.64, -0.13), (0.34, 0.18))]
    paths = list(rings)
    for outer, inner in zip(rings, rings[1:]):
        for i in range(6):
            paths.extend(([outer[i], inner[i]], [outer[i], inner[(i + 1) % 6]]))
    paths.append(_polygon(3, 0.22, -t * 0.2))
    return _geometry_overlay(frame, paths, (255, 110, 210), corners)


GEOMETRIC_COLORS = {
    mystic_seal: (45, 170, 255),
    wireframe_grid: (255, 215, 80),
    astral_lattice: (255, 110, 210),
}


EFFECTS = (
    ("Thermal", thermal),
    ("Pencil Sketch", pencil_sketch),
    ("RGB Glitch", rgb_glitch),
    ("Monochrome", monochrome),
    ("Dual-Tone", dual_tone),
    ("Color Inversion", inversion),
    ("Sepia", sepia),
    ("Blur", blur),
    ("Cartoon", cartoon),
    ("Neon Edges", neon_edges),
    ("Pixel Art", pixel_art),
    ("Comic Halftone", comic_halftone),
    ("VHS", vhs),
    ("Mystic Seal", mystic_seal),
    ("Wireframe Grid", wireframe_grid),
    ("Astral Lattice", astral_lattice),
)

