"""Derive the site's web-ready brand files from the delivered art.

The originals in ``site/assets/brand`` are what Ben was handed: a 957px icon, a
1348px lockup and a tagline lockup, all at full resolution and all PNG-heavy -
525KB for a flat-colour icon. Shipping those straight into the header would send
a megabyte of logo with every page load, so this reduces them to the sizes the
site actually renders and quantizes each one to a palette, which for flat colour
is visually lossless and roughly a tenth of the bytes.

Two things here are more than resizing.

*The dark-text lockup.* The wordmark is white, and the site follows the reader's
theme - it has a real light mode. A white wordmark on a light background is an
invisible wordmark, so this recolours it to the light theme's ink and leaves the
icon alone. The icon is a red field with a white glyph inside it and carries its
own contrast, so it must not be touched; the red field ends at x=412 in the
original and the recolour starts well clear of it at 440.

*The share card.* Transparency is not honoured in a link preview - Slack,
iMessage and the rest composite onto whatever they please, which for a white
wordmark usually means white on white. So the card is flattened onto the dark
plane rather than shipped with an alpha channel, at the 1200x630 that the
scrapers crop to.

The derived files are committed rather than generated during the site build. The
build copies bytes, which keeps it byte-identical run to run without depending
on a particular Pillow version producing a particular PNG encoding.

Run:  python3 scripts/build_brand_assets.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "site" / "assets" / "brand"
OUT = ROOT / "site" / "assets" / "web"

# The light theme's ink and plane, from STYLES in site.py. Kept in sync by hand;
# a mismatch shows up as a wordmark that is nearly but not quite the body text.
LIGHT_INK = (11, 11, 11)
DARK_PLANE = (13, 13, 13)

# Everything right of this column in the lockup is wordmark and divider. The red
# icon field ends at 412, so this clears it with room to spare.
WORDMARK_FROM = 440

# The header renders the lockup about 34px tall; this is a shade over 2x for
# retina, which is where the file stops getting visibly better.
HEADER_HEIGHT = 76
CARD = (1200, 630)


def _save(image: Image.Image, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    if image.mode == "RGBA":
        # Quantize with an alpha-aware palette. Flat colour art has a handful of
        # real colours plus antialiasing, so 128 entries is more than it needs.
        image = image.quantize(colors=128, method=Image.Quantize.FASTOCTREE)
    image.save(path, optimize=True)
    print(f"  {name}: {path.stat().st_size / 1024:.0f} KB")


def _fit(image: Image.Image, height: int) -> Image.Image:
    width = round(image.width * height / image.height)
    return image.resize((width, height), Image.LANCZOS)


def dark_text_lockup(lockup: Image.Image) -> Image.Image:
    """The same lockup with the wordmark in ink rather than white."""
    pixels = np.array(lockup.convert("RGBA"))
    region = pixels[:, WORDMARK_FROM:, :]
    # Only the colour changes; alpha is what carries the letterforms and their
    # antialiased edges, so recolouring under it keeps the shapes exactly.
    visible = region[..., 3] > 0
    region[..., 0][visible] = LIGHT_INK[0]
    region[..., 1][visible] = LIGHT_INK[1]
    region[..., 2][visible] = LIGHT_INK[2]
    pixels[:, WORDMARK_FROM:, :] = region
    return Image.fromarray(pixels, "RGBA")


def share_card(tagline: Image.Image) -> Image.Image:
    """The tagline lockup centred on the dark plane, at link-preview size."""
    card = Image.new("RGB", CARD, DARK_PLANE)
    art = tagline.convert("RGBA")
    scale = min(CARD[0] * 0.72 / art.width, CARD[1] * 0.62 / art.height)
    art = art.resize((round(art.width * scale), round(art.height * scale)), Image.LANCZOS)
    card.paste(art, ((CARD[0] - art.width) // 2, (CARD[1] - art.height) // 2), art)
    return card


def main() -> None:
    lockup = Image.open(SRC / "mri-lockup.png").convert("RGBA")
    tagline = Image.open(SRC / "mri-lockup-with-tagline.png").convert("RGBA")
    icon = Image.open(SRC / "mri-icon-1024.png").convert("RGBA")

    print("deriving web assets:")
    _save(_fit(lockup, HEADER_HEIGHT), "mri-lockup.png")
    _save(_fit(dark_text_lockup(lockup), HEADER_HEIGHT), "mri-lockup-ink.png")

    # 192 and 512 are what a web app manifest asks for; 180 is Apple's touch
    # icon. All three come from the 1024 rather than from each other, so none of
    # them is a resize of a resize.
    for size, name in ((512, "icon-512.png"), (192, "icon-192.png"),
                       (180, "apple-touch-icon.png")):
        _save(icon.resize((size, size), Image.LANCZOS), name)

    _save(share_card(tagline), "mri-card.png")

    (OUT / "favicon.ico").write_bytes((SRC / "favicon.ico").read_bytes())
    print(f"  favicon.ico: {(OUT / 'favicon.ico').stat().st_size / 1024:.0f} KB (copied)")


if __name__ == "__main__":
    main()
