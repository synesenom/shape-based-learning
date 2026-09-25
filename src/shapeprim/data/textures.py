"""Phase 3a appearance: textures, palettes, clutter, occlusion, noise, blur.

PLAN.md section 7, step 3a: take the Phase 2 drawings and change only how
they *look*. Geometry and ground truth are untouched -- the same primitives
are drawn, filled with a texture instead of a flat colour -- so H4
("the primitive model is more robust to texture, colour and style
changes") can be tested cleanly: any change in accuracy is an appearance
effect, and the oracle extractor is by construction blind to it.

Every dial is a ``GenerationConfig`` field and off by default:

- ``texture``: ``flat`` or a list of kinds drawn per primitive from
  ``noise``, ``stripes``, ``dots``, ``photo`` (a crop of a natural photo);
- ``palette``: ``default`` (Phase 1's dark grey), ``seen`` (hues in
  [0, 180) degrees) or ``unseen`` (hues in [180, 360)), so a model can be
  trained on one set of colours and tested on colours it never saw;
- ``clutter``: a textured background with random shapes and lines
  (``True``, or a probability per image);
- ``occlusion_range``: fraction of the object covered by random occluders
  drawn on top (they are not ground-truth primitives);
- ``noise_std`` (pixel noise, 0-255 scale) and ``blur_radius``, each a
  number or a ``[lo, hi]`` range sampled per image.

Randomising per image matters for anything trained on these images: a
detector trained with clutter on *every* image never saw a clean
background and failed on plain white ones (F1 0.47), while reaching 0.998
on clutter in colours it had never seen.

Part outlines are drawn in white as in Phase 1: they are the drawing
convention that separates touching parts, not an appearance choice.
"""

from __future__ import annotations

import colorsys
import math
import random
from functools import lru_cache
from typing import List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .primitives import OUTLINE_COLOR, Primitive

TEXTURE_KINDS = ("noise", "stripes", "dots", "photo")
PALETTES = ("default", "seen", "unseen")

Color = Tuple[int, int, int]


@lru_cache(maxsize=1)
def _photo_pool() -> Tuple[np.ndarray, ...]:
    """Natural photos bundled with scikit-image (no download needed)."""
    import skimage.data as sd

    photos = []
    for name in ("astronaut", "coffee", "chelsea", "rocket", "immunohistochemistry"):
        try:
            img = getattr(sd, name)()
        except Exception:  # pragma: no cover - a missing optional image
            continue
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        photos.append(np.ascontiguousarray(img[..., :3]).astype(np.uint8))
    if not photos:  # pragma: no cover
        raise RuntimeError("no scikit-image sample photos available for the photo texture")
    return tuple(photos)


def palette_color(rng: random.Random, palette: str, base: Color = (40, 40, 40)) -> Color:
    """One fill colour from ``palette``."""
    if palette == "default":
        return base
    if palette == "seen":
        hue = rng.uniform(0.0, 0.5)
    elif palette == "unseen":
        hue = rng.uniform(0.5, 1.0)
    else:
        raise ValueError(f"unknown palette {palette!r}; known: {PALETTES}")
    sat = rng.uniform(0.55, 1.0)
    val = rng.uniform(0.35, 0.8)
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return (int(r * 255), int(g * 255), int(b * 255))


def _lighter(c: Color, amount: float = 0.55) -> Color:
    return tuple(int(v + (255 - v) * amount) for v in c)


def make_texture(kind: str, size: int, color: Color, rng: random.Random, np_rng: np.random.Generator) -> np.ndarray:
    """A size x size x 3 uint8 texture patch whose dominant colour is ``color``."""
    base = np.array(color, dtype=np.float32)
    if kind == "flat":
        return np.broadcast_to(base, (size, size, 3)).astype(np.uint8)
    if kind == "noise":
        noise = np_rng.normal(0, 1, (size // 4 + 1, size // 4 + 1, 1)).astype(np.float32)
        noise = np.kron(noise, np.ones((4, 4, 1), dtype=np.float32))[:size, :size]
        fine = np_rng.normal(0, 0.5, (size, size, 1)).astype(np.float32)
        out = base + 45 * (noise + fine)
        return np.clip(out, 0, 255).astype(np.uint8)
    if kind == "stripes":
        period = rng.uniform(3.0, 8.0)
        angle = rng.uniform(0, math.pi)
        ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
        phase = (xs * math.cos(angle) + ys * math.sin(angle)) / period
        on = (np.floor(phase) % 2 == 0)[..., None]
        other = np.array(_lighter(color), dtype=np.float32)
        return np.where(on, base, other).astype(np.uint8)
    if kind == "dots":
        spacing = rng.randint(4, 8)
        radius = spacing * rng.uniform(0.2, 0.35)
        ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
        ox, oy = rng.uniform(0, spacing), rng.uniform(0, spacing)
        dx = (xs + ox) % spacing - spacing / 2
        dy = (ys + oy) % spacing - spacing / 2
        on = ((dx**2 + dy**2) <= radius**2)[..., None]
        other = np.array(_lighter(color, 0.7), dtype=np.float32)
        return np.where(on, other, base).astype(np.uint8)
    if kind == "photo":
        photos = _photo_pool()
        photo = photos[rng.randrange(len(photos))]
        crop = rng.randint(size, max(size, min(photo.shape[0], photo.shape[1]) // 2))
        y0 = rng.randint(0, photo.shape[0] - crop)
        x0 = rng.randint(0, photo.shape[1] - crop)
        patch = Image.fromarray(photo[y0 : y0 + crop, x0 : x0 + crop]).resize((size, size), Image.BILINEAR)
        arr = np.asarray(patch, dtype=np.float32)
        # Tint toward the palette colour so colour shift still applies.
        return np.clip(0.55 * arr + 0.45 * base, 0, 255).astype(np.uint8)
    raise ValueError(f"unknown texture {kind!r}; known: flat, {', '.join(TEXTURE_KINDS)}")


def clutter_background(size: int, rng: random.Random, np_rng: np.random.Generator) -> Image.Image:
    """A light textured background with random faint shapes and lines."""
    base = palette_color(rng, rng.choice(["seen", "unseen"]))
    base = _lighter(base, rng.uniform(0.6, 0.85))
    bg = make_texture(rng.choice(["noise", "stripes", "dots", "photo"]), size, base, rng, np_rng)
    # Keep the background light overall: blend toward white.
    bg = (0.45 * bg.astype(np.float32) + 0.55 * np.array(base, dtype=np.float32)).astype(np.uint8)
    img = Image.fromarray(bg)
    draw = ImageDraw.Draw(img)
    for _ in range(rng.randint(4, 10)):
        c = _lighter(palette_color(rng, rng.choice(["seen", "unseen"])), rng.uniform(0.3, 0.6))
        x0, y0 = rng.uniform(0, size), rng.uniform(0, size)
        w, h = rng.uniform(0.05, 0.3) * size, rng.uniform(0.05, 0.3) * size
        kind = rng.random()
        if kind < 0.35:
            draw.ellipse((x0, y0, x0 + w, y0 + h), fill=c)
        elif kind < 0.7:
            draw.rectangle((x0, y0, x0 + w, y0 + h), fill=c)
        else:
            draw.line((x0, y0, x0 + rng.uniform(-1, 1) * size / 2, y0 + rng.uniform(-1, 1) * size / 2),
                      fill=c, width=rng.randint(1, 3))
    return img


def _sample(value, rng: random.Random) -> float:
    """A number as-is, or a uniform draw from a ``[lo, hi]`` range."""
    if isinstance(value, (list, tuple)):
        lo, hi = value
        return rng.uniform(float(lo), float(hi))
    return float(value)


def appearance_value_positive(value) -> bool:
    if isinstance(value, (list, tuple)):
        return max(float(v) for v in value) > 0
    return bool(value) and float(value) > 0


def _mask_of(p: Primitive, size: int) -> Image.Image:
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).polygon(p.boundary_points(), fill=255)
    return m


def render_styled(
    primitives: Sequence[Primitive],
    size: int,
    rng: random.Random,
    texture: Sequence[str] | str = "flat",
    palette: str = "default",
    base_color: Color = (40, 40, 40),
    background: Color = (255, 255, 255),
    clutter=False,
    occlusion_range: Tuple[float, float] = (0.0, 0.0),
    noise_std=0.0,
    blur_radius=0.0,
) -> Image.Image:
    """Draw ``primitives`` (in order) with the requested appearance."""
    np_rng = np.random.default_rng(rng.getrandbits(32))
    if isinstance(clutter, bool):
        use_clutter = clutter
    else:
        use_clutter = rng.random() < float(clutter)
    noise_std = _sample(noise_std, rng)
    blur_radius = _sample(blur_radius, rng)
    img = clutter_background(size, rng, np_rng) if use_clutter else Image.new("RGB", (size, size), background)
    kinds = [texture] if isinstance(texture, str) else list(texture)
    object_mask = Image.new("L", (size, size), 0)
    # One colour per object (as in Phase 1, where every part of an object
    # shares its jittered grey), one texture kind per primitive.
    obj_color = palette_color(rng, palette, base_color)
    for p in primitives:
        color = obj_color if not p.is_distractor else palette_color(rng, palette, base_color)
        kind = rng.choice(kinds)
        tex = Image.fromarray(np.ascontiguousarray(make_texture(kind, size, color, rng, np_rng)))
        mask = _mask_of(p, size)
        img.paste(tex, (0, 0), mask)
        ImageDraw.Draw(img).polygon(p.boundary_points(), outline=OUTLINE_COLOR, width=p._outline_width())
        if not p.is_distractor:
            object_mask.paste(255, (0, 0), mask)

    lo, hi = occlusion_range
    if hi > 0:
        _occlude(img, np.asarray(object_mask) > 0, rng.uniform(lo, hi), rng, np_rng, palette, base_color)

    if noise_std > 0:
        arr = np.asarray(img, dtype=np.float32) + np_rng.normal(0, noise_std, (size, size, 3))
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur_radius))
    return img


def _occlude(img, obj: np.ndarray, target: float, rng, np_rng, palette, base_color) -> float:
    """Cover about ``target`` of the object's pixels with random occluders."""
    size = img.size[0]
    area = obj.sum()
    if area == 0 or target <= 0:
        return 0.0
    ys, xs = np.nonzero(obj)
    covered = np.zeros_like(obj)
    draw_mask = Image.new("L", (size, size), 0)
    for _ in range(12):
        if (covered & obj).sum() / area >= target:
            break
        # Anchor each occluder on an object pixel so it actually occludes.
        k = rng.randrange(len(xs))
        cx, cy = xs[k], ys[k]
        r = math.sqrt(area * rng.uniform(0.08, 0.25))
        w, h = r * rng.uniform(0.6, 1.6), r * rng.uniform(0.6, 1.6)
        box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        m = Image.new("L", (size, size), 0)
        (ImageDraw.Draw(m).ellipse if rng.random() < 0.5 else ImageDraw.Draw(m).rectangle)(box, fill=255)
        covered |= np.asarray(m) > 0
        draw_mask.paste(255, (0, 0), m)
    color = _lighter(palette_color(rng, rng.choice(["seen", "unseen"])), rng.uniform(0.0, 0.5))
    tex = Image.fromarray(np.ascontiguousarray(make_texture(rng.choice(["flat", "noise", "stripes"]), size, color, rng, np_rng)))
    img.paste(tex, (0, 0), draw_mask)
    return float((covered & obj).sum() / area)
