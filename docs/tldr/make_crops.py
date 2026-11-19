"""Panels of the one-page summary (docs/TLDR.pdf), cut from the example figures of docs/examples/.

Usage (from the repository): python docs/tldr/make_crops.py, then in docs/tldr:
pdflatex TLDR.tex, and copy TLDR.pdf to docs/ (pdftoppm -r 200 -png -singlefile TLDR.pdf ../TLDR
for the PNG). Rerun after the examples change.
"""

from pathlib import Path

from PIL import Image

HERE = Path(__file__).parent
EX = HERE.parent / "examples"
CROPS = {   # file: (source, box in pixels: left, top, right, bottom)
    "barkley_map.png": ("barkley/phase_map_hand.png", (735, 0, 1350, 690)),
    "barkley_ref.png": ("barkley/phase_map_hand.png", (1355, 0, 1980, 690)),
    "barkley_states.png": ("barkley/states.png", (0, 0, 1095, 410)),
    "gs_map.png": ("gray-scott/phase_map_hand.png", (720, 0, 1350, 690)),
    "gs_phases.png": ("gray-scott/phases.png", (0, 0, 2935, 450)),
}

if __name__ == "__main__":
    for name, (src, box) in CROPS.items():
        Image.open(EX / src).crop(box).save(HERE / name)
