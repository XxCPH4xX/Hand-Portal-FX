"""Fetch the official model once; camera operation itself stays offline."""

import hashlib
import sys
import urllib.error
import urllib.request

import config

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"


def download() -> None:
    path = config.MODEL_PATH
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == MODEL_SHA256:
        print(f"Model ready: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".download")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != MODEL_SHA256:
            raise RuntimeError("Model checksum mismatch; download was not installed.")
        temporary.write_bytes(data)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Model ready: {path}")


if __name__ == "__main__":
    try:
        download()
    except (OSError, urllib.error.URLError, RuntimeError) as error:
        print(f"Model download failed: {error}", file=sys.stderr)
        sys.exit(1)

