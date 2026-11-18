"""Defect regions (bounding boxes) from the two domain-trained classifiers.

How it works
------------
The CODEBRIM and SDNET2018 models classify a whole image; they do not output boxes. This module
scores overlapping 224 px tiles of the (down-scaled) photo with both models, keeps the tiles whose
score reaches the detection threshold, groups touching tiles, and reports the union of each group
as a box. The score of a box is the highest tile score in it.

What it is NOT (say this when presenting)
-----------------------------------------
* It is not a trained object detector. Boxes are tile-aligned (coarse, about 112-224 px) and
  inherit the classifiers' known weaknesses (domain shift, uncalibrated scores).
* Tile scores are not probabilities of a defect at a location; they are classifier outputs on a
  crop. The photo is down-scaled so that its longest side is at most ``WORK_MAX_SIDE`` px; the
  effect of that scale on the models is untested.
* SDNET2018 and CODEBRIM boxes are kept separate and never merged or averaged.

The grouping logic is pure Python/NumPy so it can be unit-tested without torch; only
``score_tiles`` needs torch and the checkpoints.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

import cv2
import numpy as np

METHOD = "tile_score_v1"
TILE = 224
STRIDE = 112
MAX_TILES = 80
WORK_MAX_SIDE = 1024
MAX_REGIONS = 8
MAX_REGION_FRACTION = 0.35   # a grouped box larger than this share of the image is split into tile boxes
NMS_IOU = 0.3                # same overlap limit as the training repo's sliding-window config
DIFFUSE_FRACTION = 0.8       # firing on >= 80 % of tiles = image-wide, not localized

NOTE = (
    "Regions come from classifier scores on 224 px tiles (tile-aligned, coarse); crack boxes are "
    "tightened with a classical dark-line search inside flagged tiles. Not a trained object "
    "detector and not validated; expect misses and false positives."
)

Tile = tuple[int, int, int, int]  # x1, y1, x2, y2

SOURCE_TAG = {"codebrim": "CB", "sdnet2018": "SD"}   # legend is shown in the app

COLOURS = {
    "crack": (230, 40, 40),
    "spallation": (255, 140, 0),
    "efflorescence": (60, 140, 255),
    "exposed_bars": (200, 0, 160),
    "corrosion_stain": (160, 90, 20),
}


def resize_for_work(image_rgb: np.ndarray, max_side: int = WORK_MAX_SIDE) -> tuple[np.ndarray, float]:
    """Down-scale (never up-scale) so the longest side is <= ``max_side``.

    Returns ``(work_image, scale)`` where ``scale = original_size / work_size`` (>= 1).
    """
    height, width = image_rgb.shape[:2]
    longest = max(height, width)

    if longest <= max_side:
        return image_rgb, 1.0

    factor = max_side / float(longest)
    work = cv2.resize(
        image_rgb,
        (max(1, int(round(width * factor))), max(1, int(round(height * factor)))),
        interpolation=cv2.INTER_AREA,
    )

    return work, longest / float(max(work.shape[:2]))


def tile_grid(
    height: int,
    width: int,
    tile: int = TILE,
    stride: int = STRIDE,
    max_tiles: int = MAX_TILES,
) -> list[Tile]:
    """Overlapping tiles covering every pixel. Small images give one whole-image tile."""
    if height < tile or width < tile:
        return [(0, 0, width, height)]

    def starts(length: int) -> list[int]:
        values = list(range(0, length - tile + 1, stride))

        if values[-1] != length - tile:
            values.append(length - tile)

        return values

    ys, xs = starts(height), starts(width)

    if len(ys) * len(xs) > max_tiles and stride < tile:
        return tile_grid(height, width, tile, tile, max_tiles)  # no overlap

    return [(x, y, x + tile, y + tile) for y in ys for x in xs]


def score_tiles(
    models: tuple[Any, Any],
    work_rgb: np.ndarray,
    tiles: Sequence[Tile],
    batch_size: int = 16,
) -> dict[str, Any]:
    """Run both models on every tile (needs torch). Returns plain Python lists.

    ``{"codebrim": {class_name: [score per tile]}, "sdnet": [crack score per tile]}``
    """
    import torch  # lazy: keeps the grouping logic importable without torch

    from rodic_inference.inference import CODEBRIM_CLASSES, prepare_tensor

    codebrim_model, sdnet_model = models

    codebrim = {name: [] for name in CODEBRIM_CLASSES}
    sdnet: list[float] = []

    for start in range(0, len(tiles), batch_size):
        batch = tiles[start : start + batch_size]
        tensor = torch.cat(
            [prepare_tensor(work_rgb[y1:y2, x1:x2]) for x1, y1, x2, y2 in batch]
        )

        with torch.inference_mode():
            cb = torch.sigmoid(codebrim_model(tensor)).cpu().numpy()
            sd = torch.sigmoid(sdnet_model(tensor)).reshape(-1).cpu().numpy()

        for row in cb:
            for name, value in zip(CODEBRIM_CLASSES, row):
                codebrim[name].append(float(value))

        sdnet.extend(float(v) for v in sd)

    return {"codebrim": codebrim, "sdnet": sdnet}


def _touch(a: Tile, b: Tile) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _group(indices: list[int], tiles: Sequence[Tile]) -> list[list[int]]:
    parent = {i: i for i in indices}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for pos, i in enumerate(indices):
        for j in indices[pos + 1 :]:
            if _touch(tiles[i], tiles[j]):
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}

    for i in indices:
        groups.setdefault(find(i), []).append(i)

    return list(groups.values())


def _iou(a: Tile, b: Tile) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter

    return inter / union if union > 0 else 0.0


def _nms(indices: list[int], tiles: Sequence[Tile], scores: Sequence[float]) -> list[int]:
    kept: list[int] = []

    for i in sorted(indices, key=lambda k: float(scores[k]), reverse=True):
        if all(_iou(tiles[i], tiles[j]) <= NMS_IOU for j in kept):
            kept.append(i)

    return kept


def find_regions(
    tiles: Sequence[Tile],
    scores_by_label: dict[str, tuple[str, Sequence[float]]],
    threshold: float,
    image_area: float,
    max_regions: int = MAX_REGIONS,
) -> list[dict[str, Any]]:
    """Boxes (WORK coordinates) from tiles at or above ``threshold``.

    * Touching hot tiles are grouped and reported as one box (score = highest tile score).
    * A group whose box covers more than ``MAX_REGION_FRACTION`` of the image (for example a long
      diagonal crack) is split into non-overlapping tile boxes by NMS, so boxes stay informative.
    * A label that fires on at least ``DIFFUSE_FRACTION`` of all tiles is reported as ONE
      ``image_wide`` region: the classifier does not localize it.
    ``scores_by_label`` maps a label to ``(source, [score per tile])``; sources are never merged.
    Returns regions sorted by score (highest first), at most ``max_regions``.
    """
    regions: list[dict[str, Any]] = []
    all_x2 = max((t[2] for t in tiles), default=0)
    all_y2 = max((t[3] for t in tiles), default=0)

    for label, (source, scores) in scores_by_label.items():
        hot = [i for i, s in enumerate(scores) if float(s) >= threshold]

        if not hot:
            continue

        if len(tiles) > 1 and len(hot) / len(tiles) >= DIFFUSE_FRACTION:
            regions.append(
                {
                    "label": label,
                    "source": source,
                    "score": max(float(scores[i]) for i in hot),
                    "tile_count": len(hot),
                    "box": (0, 0, all_x2, all_y2),
                    "image_wide": True,
                }
            )
            continue

        for group in _group(hot, tiles):
            x1 = min(tiles[i][0] for i in group)
            y1 = min(tiles[i][1] for i in group)
            x2 = max(tiles[i][2] for i in group)
            y2 = max(tiles[i][3] for i in group)

            if len(group) > 1 and (x2 - x1) * (y2 - y1) > MAX_REGION_FRACTION * image_area:
                for i in _nms(group, tiles, scores):
                    regions.append(
                        {
                            "label": label,
                            "source": source,
                            "score": float(scores[i]),
                            "tile_count": 1,
                            "box": tuple(tiles[i]),
                            "image_wide": False,
                        }
                    )
                continue

            regions.append(
                {
                    "label": label,
                    "source": source,
                    "score": max(float(scores[i]) for i in group),
                    "tile_count": len(group),
                    "box": (x1, y1, x2, y2),
                    "image_wide": False,
                }
            )

    regions.sort(key=lambda r: r["score"], reverse=True)

    return regions[:max_regions]


def refine_crack_box(
    work_rgb: np.ndarray,
    box: Tile,
    pad: int = 10,
    kernel: int = 15,
    min_contrast: float = 25.0,
    min_aspect: float = 3.0,
) -> Tile | None:
    """Tighten a crack tile box around dark, elongated structures inside it (OpenCV black-hat).

    Heuristic, not a detector: it only runs inside tiles the classifier already flagged, and
    returns ``None`` (keep the tile box) when no elongated dark structure is found.
    """
    x1, y1, x2, y2 = box
    crop = work_rgb[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    gray = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), (3, 3), 0)
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, se)

    if float(blackhat.max()) < min_contrast:
        return None

    _, mask = cv2.threshold(blackhat, min_contrast, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    xs: list[int] = []
    ys: list[int] = []

    for contour in contours:
        (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
        short, long_ = sorted((rw, rh))

        if long_ < 15 or long_ / max(short, 1.0) < min_aspect:
            continue

        cx, cy, cw, ch = cv2.boundingRect(contour)
        xs += [cx, cx + cw]
        ys += [cy, cy + ch]

    if not xs:
        return None

    height, width = crop.shape[:2]

    return (
        x1 + max(0, min(xs) - pad),
        y1 + max(0, min(ys) - pad),
        x1 + min(width, max(xs) + pad),
        y1 + min(height, max(ys) + pad),
    )


def _dedupe(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []

    for region in sorted(regions, key=lambda r: r["score"], reverse=True):
        if all(
            not (k["label"] == region["label"] and k["source"] == region["source"])
            or _iou(tuple(k["box"]), tuple(region["box"])) <= 0.5
            for k in kept
        ):
            kept.append(region)

    return kept


def regions_from_bundle(
    bundle: dict[str, Any],
    threshold: float,
    max_regions: int = MAX_REGIONS,
    image_rgb: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Regions in ORIGINAL image coordinates from a cached CV bundle (cheap: no model calls).

    With ``image_rgb`` given, crack regions are tightened around dark elongated structures
    (``box_basis = "tile+blackhat"``); otherwise boxes are the tile boxes (``"tile"``).
    """
    tile_scores = bundle["tile_scores"]
    tiles = [tuple(t) for t in bundle["tiles"]]

    labelled: dict[str, tuple[str, Sequence[float]]] = {}

    for name, values in tile_scores["codebrim"].items():
        if name != "background":
            labelled[name] = ("codebrim", values)

    # SDNET2018 only knows "crack"; key it separately so it never merges with CODEBRIM's crack.
    labelled["crack#sdnet"] = ("sdnet2018", tile_scores["sdnet"])

    work_h, work_w = bundle.get("work_size") or (
        max(t[3] for t in tiles),
        max(t[2] for t in tiles),
    )

    regions = find_regions(
        tiles, labelled, threshold, float(work_h) * float(work_w), max_regions * 2
    )

    work_rgb = resize_for_work(image_rgb)[0] if image_rgb is not None else None

    for region in regions:
        region["label"] = region["label"].split("#")[0]
        region["box_basis"] = "tile"

        if work_rgb is not None and region["label"] == "crack" and not region.get("image_wide"):
            refined = refine_crack_box(work_rgb, tuple(region["box"]))

            if refined is not None:
                region["box"] = refined
                region["box_basis"] = "tile+blackhat"

    regions = _dedupe(regions)[:max_regions]

    scale = float(bundle["scale"])
    height, width = bundle["image_size"]

    for region in regions:
        x1, y1, x2, y2 = region["box"]
        region["box"] = (
            int(max(0, round(x1 * scale))),
            int(max(0, round(y1 * scale))),
            int(min(width, round(x2 * scale))),
            int(min(height, round(y2 * scale))),
        )

    return regions


def draw_regions(image_rgb: np.ndarray, regions: Iterable[dict[str, Any]]) -> np.ndarray:
    """Return a copy of ``image_rgb`` with labelled boxes."""
    canvas = image_rgb.copy()
    height, width = canvas.shape[:2]
    thickness = max(2, int(round(max(height, width) / 400)))
    font_scale = max(0.5, max(height, width) / 1600)

    for region in regions:
        x1, y1, x2, y2 = region["box"]
        colour = COLOURS.get(region["label"], (255, 255, 0))
        text = f'{region["label"]} {SOURCE_TAG.get(region["source"], region["source"])} {region["score"]:.2f}'

        if region.get("image_wide"):
            text += " image-wide"

        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness)

        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        top = max(0, y1 - th - base - 4)

        cv2.rectangle(canvas, (x1, top), (min(width, x1 + tw + 6), top + th + base + 4), colour, -1)
        cv2.putText(
            canvas,
            text,
            (x1 + 3, top + th + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return canvas


def compute_tile_bundle(
    models: tuple[Any, Any],
    image_rgb: np.ndarray,
    scorer: Callable[..., dict[str, Any]] = score_tiles,
) -> dict[str, Any]:
    """Tile scores for one image (the expensive step; cache this, not the boxes)."""
    work, scale = resize_for_work(image_rgb)
    tiles = tile_grid(*work.shape[:2])

    return {
        "tiles": tiles,
        "tile_scores": scorer(models, work, tiles),
        "scale": scale,
        "image_size": tuple(int(v) for v in image_rgb.shape[:2]),
        "work_size": tuple(int(v) for v in work.shape[:2]),
    }
