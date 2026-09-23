"""Learned primitive extractor: a small CenterNet-style detector.

PLAN.md section 4: "small detector trained on synthetic images to output
primitive type + pose". One heatmap per canonical primitive type marks
primitive centres; at each centre the network regresses a sub-cell offset,
the log width/height, and the orientation. It is trained once,
class-agnostically, from primitive-level labels only -- it never sees an
object class. The number of primitive-annotated images it was trained on
is recorded with the weights, because that annotation budget is the
hidden cost of the primitive pipeline (docs/research_plan.md, WP3.3) and
has to be reported next to any few-shot result.

Pose conventions (so that equivalent descriptions get one target):

- ellipse, quadrilateral, line: width >= height (swap and add 90 degrees
  otherwise) and orientation is 180-degree periodic, regressed as
  (cos 2t, sin 2t);
- triangle: orientation is the apex direction, 360-degree periodic,
  regressed as (cos t, sin t).

Decoded quadrilaterals and triangles are the rectangle and isosceles
triangle with the predicted pose; under perspective the true shapes are
general, and the gap shows up in the IoU-based F1 like any other error.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ..data.primitives import CANONICAL_TYPE, CANONICAL_TYPES, Primitive
from .base import PrimitiveExtractor

STRIDE = 2
NUM_T = len(CANONICAL_TYPES)
# Output channels: heatmaps, offset (2), log size (2), angle (2).
OUT_CHANNELS = NUM_T + 6


def _block(cin: int, cout: int, stride: int = 1, dilation: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride=stride, padding=dilation, dilation=dilation, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class CenterNetSmall(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        w = width
        self.enc1 = nn.Sequential(_block(3, w), _block(w, w))  # 1/1
        self.enc2 = nn.Sequential(_block(w, 2 * w, 2), _block(2 * w, 2 * w))  # 1/2
        self.enc3 = nn.Sequential(_block(2 * w, 4 * w, 2), _block(4 * w, 4 * w, dilation=2), _block(4 * w, 4 * w, dilation=4))  # 1/4
        self.up = _block(4 * w + 2 * w, 2 * w)
        self.head = nn.Sequential(_block(2 * w, 2 * w), nn.Conv2d(2 * w, OUT_CHANNELS, 1))
        # CenterNet's prior: heatmaps start near 0.1 so focal loss is stable.
        nn.init.constant_(self.head[-1].bias[:NUM_T], -2.19)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        u = F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(self.up(torch.cat([u, e2], dim=1)))


def _canonical_pose(p: Primitive):
    ctype = CANONICAL_TYPE[p.type]
    w, h, t = p.width, p.height, p.rotation
    if ctype == "triangle":
        return ctype, w, h, (math.cos(t), math.sin(t))
    if h > w:
        w, h, t = h, w, t + math.pi / 2
    return ctype, w, h, (math.cos(2 * t), math.sin(2 * t))


def encode_targets(primitives: Sequence[Primitive], image_size: int) -> Dict[str, np.ndarray]:
    """Dense training targets for one image."""
    g = image_size // STRIDE
    heat = np.zeros((NUM_T, g, g), dtype=np.float32)
    reg = np.zeros((6, g, g), dtype=np.float32)
    mask = np.zeros((g, g), dtype=np.float32)
    ys, xs = np.mgrid[0:g, 0:g]
    for p in primitives:
        ctype, w, h, (ca, sa) = _canonical_pose(p)
        k = CANONICAL_TYPES.index(ctype)
        fx, fy = p.cx / STRIDE, p.cy / STRIDE
        ix, iy = int(fx), int(fy)
        if not (0 <= ix < g and 0 <= iy < g):
            continue
        sigma = min(max(min(w, h) / (STRIDE * 6.0), 0.6), 2.0)
        gauss = np.exp(-((xs - ix) ** 2 + (ys - iy) ** 2) / (2 * sigma**2))
        heat[k] = np.maximum(heat[k], gauss)
        reg[:, iy, ix] = [fx - ix, fy - iy, math.log(max(w, 1.0)), math.log(max(h, 1.0)), ca, sa]
        mask[iy, ix] = 1.0
    return {"heat": heat, "reg": reg, "mask": mask}


def image_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def focal_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = torch.sigmoid(pred_logits).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    neg = 1 - pos
    pos_loss = torch.log(pred) * (1 - pred) ** 2 * pos
    neg_loss = torch.log(1 - pred) * pred**2 * (1 - target) ** 4 * neg
    n_pos = pos.sum().clamp(min=1)
    return -(pos_loss.sum() + neg_loss.sum()) / n_pos


def detector_loss(out: torch.Tensor, heat: torch.Tensor, reg: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    hm_loss = focal_loss(out[:, :NUM_T], heat)
    m = mask.unsqueeze(1)
    n = m.sum().clamp(min=1)
    reg_pred = out[:, NUM_T:]
    reg_loss = (F.l1_loss(reg_pred, reg, reduction="none") * m).sum() / n
    return hm_loss + reg_loss


def decode(out: torch.Tensor, threshold: float = 0.3, max_det: int = 24, color=(40, 40, 40)) -> List[Primitive]:
    """One image's output map (C, g, g) -> primitives in pixel coordinates."""
    heat = torch.sigmoid(out[:NUM_T])
    peaks = heat == F.max_pool2d(heat.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)
    scores = heat * peaks
    flat = scores.flatten()
    k = min(max_det, flat.numel())
    vals, idx = torch.topk(flat, k)
    g = heat.shape[-1]
    prims: List[Primitive] = []
    for v, i in zip(vals.tolist(), idx.tolist()):
        if v < threshold:
            break
        t, rem = divmod(i, g * g)
        iy, ix = divmod(rem, g)
        ox, oy, lw, lh, ca, sa = out[NUM_T:, iy, ix].tolist()
        ctype = CANONICAL_TYPES[t]
        w, h = math.exp(lw), math.exp(lh)
        ang = math.atan2(sa, ca)
        rot = ang if ctype == "triangle" else ang / 2
        prims.append(
            Primitive(
                type=ctype, cx=(ix + ox) * STRIDE, cy=(iy + oy) * STRIDE,
                width=max(w, 1.0), height=max(h, 1.0), rotation=rot, color=color,
            )
        )
    return prims


class LearnedExtractor(PrimitiveExtractor):
    """Loads a trained ``CenterNetSmall`` and decodes its detections."""

    def __init__(self, weights: str | Path, threshold: float = 0.3):
        weights = Path(weights)
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        self.model = CenterNetSmall(width=ckpt.get("width", 32))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.threshold = threshold
        self.meta = ckpt.get("meta", {})
        # The weights file identifies the extractor for the cache key.
        self.name = f"learned:{weights.stem}"

    @torch.no_grad()
    def extract(self, image: Image.Image, ground_truth: Optional[List[Primitive]] = None) -> List[Primitive]:
        out = self.model(image_tensor(image).unsqueeze(0))[0]
        return decode(out, threshold=self.threshold)


def train_detector(
    train_samples,
    val_samples,
    image_size: int,
    epochs: int = 20,
    lr: float = 2e-3,
    batch_size: int = 32,
    width: int = 32,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[CenterNetSmall, dict]:
    """Train on (image, primitives) samples; returns the model and a log."""
    torch.manual_seed(seed)

    def tensors(samples):
        imgs, heats, regs, masks = [], [], [], []
        for s in samples:
            imgs.append(image_tensor(s.image))
            t = encode_targets([p for p in s.primitives], image_size)
            heats.append(torch.from_numpy(t["heat"]))
            regs.append(torch.from_numpy(t["reg"]))
            masks.append(torch.from_numpy(t["mask"]))
        return torch.stack(imgs), torch.stack(heats), torch.stack(regs), torch.stack(masks)

    x, hm, rg, mk = tensors(train_samples)
    vx, vhm, vrg, vmk = tensors(val_samples)
    model = CenterNetSmall(width=width)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * math.ceil(len(x) / batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    log = {"history": []}
    best, best_state = float("inf"), None
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(x))
        total = 0.0
        for b in range(0, len(x), batch_size):
            i = perm[b : b + batch_size]
            loss = detector_loss(model(x[i]), hm[i], rg[i], mk[i])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            total += loss.item() * len(i)
        model.eval()
        with torch.no_grad():
            vl = sum(
                detector_loss(model(vx[b : b + 128]), vhm[b : b + 128], vrg[b : b + 128], vmk[b : b + 128]).item()
                * len(vx[b : b + 128])
                for b in range(0, len(vx), 128)
            ) / len(vx)
        log["history"].append({"epoch": epoch, "train_loss": total / len(x), "val_loss": vl})
        if vl < best:
            best = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"  epoch {epoch:2d} train {total / len(x):.4f} val {vl:.4f} ({time.time() - t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    log["best_val_loss"] = best
    log["train_seconds"] = time.time() - t0
    return model, log
