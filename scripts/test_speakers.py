#!/usr/bin/env python3
"""Speaker test — plays wav files from assets/ on the USB PnP Sound Device (card 2)."""

import subprocess
import sys
from pathlib import Path

USB_CARD = 2  # C-Media USB PnP Sound Device
VOLUME = 80   # percent (0-100), adjust to taste

ASSETS = Path(__file__).parent.parent / "assets"


def set_volume(pct):
    # Convert percent to 0-151 range
    val = int(151 * pct / 100)
    subprocess.run(["amixer", "-c", str(USB_CARD), "cset", "numid=4", f"{val},{val}"], capture_output=True)
    subprocess.run(["amixer", "-c", str(USB_CARD), "sset", "PCM Playback Switch", "on"], capture_output=True)


def play(wav_path):
    result = subprocess.run(
        ["aplay", "-D", f"plughw:{USB_CARD},0", "-q", str(wav_path)],
        timeout=10,
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr.decode().strip()}")
        return False
    return True


set_volume(VOLUME)

wav_files = sorted(ASSETS.glob("*.wav"))
if not wav_files:
    print(f"No .wav files found in {ASSETS}")
    sys.exit(1)

# If a filename arg is given, play just that one
if len(sys.argv) > 1:
    target = ASSETS / sys.argv[1]
    if not target.exists():
        print(f"File not found: {target}")
        sys.exit(1)
    print(f"Playing {target.name} ...")
    play(target)
    sys.exit(0)

# Otherwise play them all
print(f"Playing {len(wav_files)} wav files from assets/ (volume {VOLUME}%):\n")
for wav in wav_files:
    print(f"  {wav.name} ... ", end="", flush=True)
    ok = play(wav)
    if ok:
        print("OK")

print("\nDone.")
