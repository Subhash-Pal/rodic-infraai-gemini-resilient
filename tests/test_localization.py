import numpy as np

from rodic_inference import localization as loc
from tests.helpers import blank_image, make_bundle


def test_tile_grid_covers_every_pixel_and_small_image_is_one_tile():
    tiles = loc.tile_grid(700, 900)
    covered = np.zeros((700, 900), bool)

    for x1, y1, x2, y2 in tiles:
        covered[y1:y2, x1:x2] = True

    assert covered.all()
    assert len(tiles) <= loc.MAX_TILES
    assert loc.tile_grid(100, 150) == [(0, 0, 150, 100)]


def test_resize_for_work_never_upscales_and_reports_scale():
    small, s1 = loc.resize_for_work(blank_image(300, 400))
    big, s2 = loc.resize_for_work(blank_image(2048, 4096))

    assert small.shape[:2] == (300, 400) and s1 == 1.0
    assert max(big.shape[:2]) == loc.WORK_MAX_SIDE and abs(s2 - 4.0) < 0.01


def test_touching_hot_tiles_make_one_box_and_far_tiles_make_two():
    tiles = [(0, 0, 224, 224), (112, 0, 336, 224), (700, 700, 924, 924), (0, 500, 224, 724)]
    scores = {"crack": ("codebrim", [0.9, 0.7, 0.8, 0.1])}
    regions = loc.find_regions(tiles, scores, 0.5, image_area=1000 * 1000)

    boxes = sorted(r["box"] for r in regions)
    assert boxes == [(0, 0, 336, 224), (700, 700, 924, 924)]
    assert regions[0]["score"] == 0.9 and regions[0]["tile_count"] == 2


def test_below_threshold_gives_no_regions():
    tiles = [(0, 0, 224, 224)] * 3
    assert loc.find_regions(tiles, {"crack": ("codebrim", [0.1, 0.2, 0.3])}, 0.5, 1e6) == []


def test_long_group_is_split_into_non_overlapping_tile_boxes():
    tiles = loc.tile_grid(768, 1024)
    hot = [1.0 if (abs((x1 / 1024) - (y1 / 768)) < 0.12) else 0.0 for x1, y1, _x2, _y2 in tiles]
    regions = loc.find_regions(tiles, {"crack": ("sdnet2018", hot)}, 0.5, 768 * 1024)

    assert 2 <= len(regions) <= loc.MAX_REGIONS
    assert all(r["tile_count"] == 1 for r in regions)

    for i, a in enumerate(regions):
        for b in regions[i + 1 :]:
            assert loc._iou(tuple(a["box"]), tuple(b["box"])) <= loc.NMS_IOU


def test_firing_on_almost_every_tile_is_reported_image_wide():
    tiles = loc.tile_grid(448, 448, max_tiles=80)
    regions = loc.find_regions(tiles, {"crack": ("sdnet2018", [0.99] * len(tiles))}, 0.5, 448 * 448)

    assert len(regions) == 1 and regions[0]["image_wide"] is True


def test_sources_are_never_merged_even_for_the_same_label():
    bundle = make_bundle(codebrim={"crack": 0.9}, sdnet=0.9)
    regions = loc.regions_from_bundle(bundle, 0.5)

    assert {r["source"] for r in regions if r["label"] == "crack"} == {"codebrim", "sdnet2018"}
    assert all(r["label"] != "background" for r in regions)


def test_regions_are_scaled_to_original_pixels_and_clipped():
    bundle = make_bundle(image_size=(900, 900), sdnet=0.9)
    bundle["scale"] = 2.0
    bundle["work_size"] = (450, 450)
    bundle["tiles"] = [(0, 0, 224, 224), (226, 226, 450, 450), (0, 226, 224, 450), (226, 0, 450, 224)]
    bundle["tile_scores"]["sdnet"] = [0.9, 0.0, 0.0, 0.0]
    box = loc.regions_from_bundle(bundle, 0.5)[0]["box"]

    assert box == (0, 0, 448, 448)

    bundle["tile_scores"]["sdnet"] = [0.0, 0.9, 0.0, 0.0]
    box = loc.regions_from_bundle(bundle, 0.5)[0]["box"]

    assert box[2] <= 900 and box[3] <= 900


def test_refine_crack_box_hugs_a_dark_line_and_ignores_blank_tiles():
    import cv2

    img = blank_image(300, 300)
    cv2.line(img, (60, 80), (200, 150), (20, 20, 20), 3)
    tile = (0, 0, 224, 224)
    refined = loc.refine_crack_box(img, tile)

    assert refined is not None
    x1, y1, x2, y2 = refined
    assert x1 <= 60 and y1 <= 80 and x2 >= 200 - 5 and y2 >= 150 - 5
    assert (x2 - x1) * (y2 - y1) < 224 * 224
    assert loc.refine_crack_box(blank_image(300, 300), tile) is None


def test_draw_regions_returns_a_copy_with_visible_boxes():
    img = blank_image(300, 300)
    original = img.copy()
    out = loc.draw_regions(img, [{"label": "crack", "source": "sdnet2018", "score": 0.9, "box": (20, 40, 120, 140)}])

    assert (img == original).all()
    assert (out != original).any() and out.shape == img.shape
