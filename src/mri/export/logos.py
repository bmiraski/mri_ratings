"""Cache team logos into the site so it does not depend on someone else's CDN.

Hotlinking 138 images means the site is slower than it needs to be, breaks if
the CDN moves or rate-limits, and leaks every visitor to a third party. The
files are a few kilobytes each, so there is no reason not to carry them.

Logos are trademarks of their schools. Using them to identify the team whose
rating is being shown is ordinary editorial use; they are not modified, and the
site is not selling anything.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import requests

TIMEOUT = 30
LOGO_DIR = "logos"

# CFBD serves each logo at several widths under /logos/<width>/<id>.png and
# defaults to 500px. The site never renders one larger than 46px, so the 128px
# variant is already generous and roughly a twentieth of the bytes.
PREFERRED_WIDTH = 128


def cache_logos(payload: dict, out_dir: Path, *, refresh: bool = False) -> dict:
    """Download each team's logo and rewrite the payload to point at the copy.

    Returns a summary. Teams whose logo cannot be fetched keep their remote URL,
    and the site's colour-chip fallback covers them if that fails too - a
    missing logo is never allowed to break a page.
    """
    target = out_dir / LOGO_DIR
    target.mkdir(parents=True, exist_ok=True)

    fetched = cached = failed = 0
    session = requests.Session()

    for team in payload["teams"]:
        url = team.get("logo")
        if not url:
            continue
        url = _prefer_small(url)

        suffix = Path(url.split("?")[0]).suffix or ".png"
        if suffix.lower() not in {".png", ".svg", ".jpg", ".jpeg", ".webp"}:
            suffix = ".png"
        name = f"{hashlib.sha1(url.encode()).hexdigest()[:12]}{suffix}"
        path = target / name

        if path.exists() and not refresh:
            cached += 1
        else:
            try:
                response = session.get(url, timeout=TIMEOUT)
                response.raise_for_status()
                if not response.content:
                    raise ValueError("empty body")
                path.write_bytes(_shrink(response.content, suffix))
                fetched += 1
            except Exception as exc:  # noqa: BLE001 - a logo is never fatal
                print(f"  logo failed for {team['team']}: {exc}")
                failed += 1
                # Cleared rather than left pointing at a URL we just failed to
                # fetch. The page falls back to the colour chip on its own; a
                # remote URL that 403s is 365 broken requests per page load and
                # a flash of missing images before the onerror handler runs.
                team["logo"] = None
                continue

        team["logoRemote"] = url
        team["logo"] = f"{LOGO_DIR}/{name}"

    return {"fetched": fetched, "cached": cached, "failed": failed}


def _prefer_small(url: str) -> str:
    """Swap CFBD's 500px default for a size the site actually renders."""
    return re.sub(r"/logos/\d+/", f"/logos/{PREFERRED_WIDTH}/", url)


def _shrink(data: bytes, suffix: str) -> bytes:
    """Downscale a raster logo to the width the site renders.

    CFBD serves whatever size you ask for in the path. ESPN - which is where the
    basketball logos come from, because CFBD's CDN carries football schools only
    - serves 500px and ignores every size hint. At 63KB each, 365 of those is
    23MB committed to the repository and pushed to every visitor, for marks the
    site never draws larger than 46px.

    Failure here is never fatal: a logo that will not resize is stored as it
    arrived, which costs bytes and loses nothing.
    """
    if suffix.lower() == ".svg":
        return data
    try:
        import io

        from PIL import Image

        image = Image.open(io.BytesIO(data))
        if image.width <= PREFERRED_WIDTH:
            return data
        image = image.convert("RGBA")
        image.thumbnail((PREFERRED_WIDTH, PREFERRED_WIDTH), Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()
    except Exception:  # noqa: BLE001 - a logo is never fatal
        return data
