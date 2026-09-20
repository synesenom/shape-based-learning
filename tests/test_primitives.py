import math

import pytest
from PIL import Image, ImageDraw

from shapeprim.data.primitives import Primitive


def test_rejects_unknown_type():
    with pytest.raises(ValueError):
        Primitive(type="hexagon", cx=0, cy=0, width=1, height=1)


def test_rejects_nonpositive_size():
    with pytest.raises(ValueError):
        Primitive(type="circle", cx=0, cy=0, width=0, height=1)


def test_rectangle_bbox_axis_aligned():
    p = Primitive(type="rectangle", cx=50, cy=50, width=20, height=10, rotation=0.0)
    x0, y0, x1, y1 = p.bbox()
    assert x0 == pytest.approx(40)
    assert x1 == pytest.approx(60)
    assert y0 == pytest.approx(45)
    assert y1 == pytest.approx(55)


def test_rectangle_rotation_90_swaps_extent():
    p = Primitive(type="rectangle", cx=0, cy=0, width=20, height=10, rotation=math.pi / 2)
    x0, y0, x1, y1 = p.bbox()
    assert (x1 - x0) == pytest.approx(10, abs=1e-6)
    assert (y1 - y0) == pytest.approx(20, abs=1e-6)


def test_triangle_polygon_has_three_vertices():
    p = Primitive(type="triangle", cx=10, cy=10, width=8, height=6)
    assert len(p.polygon()) == 3


def test_circle_has_no_polygon():
    p = Primitive(type="circle", cx=0, cy=0, width=4, height=4)
    with pytest.raises(ValueError):
        p.polygon()


def test_to_dict_from_dict_roundtrip():
    p = Primitive(type="triangle", cx=1.5, cy=2.5, width=3, height=4, rotation=0.7, color=(10, 20, 30), is_distractor=True)
    d = p.to_dict()
    p2 = Primitive.from_dict(d)
    assert p2 == p


def test_draw_fills_center_pixel():
    size = 64
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    p = Primitive(type="circle", cx=32, cy=32, width=20, height=20, color=(10, 10, 10))
    p.draw(draw)
    assert img.getpixel((32, 32)) == (10, 10, 10)
    assert img.getpixel((0, 0)) == (255, 255, 255)


@pytest.mark.parametrize("ptype", ["rectangle", "triangle", "line"])
def test_polygon_draw_fills_center(ptype):
    size = 64
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    p = Primitive(type=ptype, cx=32, cy=32, width=20, height=10, rotation=0.3, color=(5, 5, 5))
    p.draw(draw)
    assert img.getpixel((32, 32)) == (5, 5, 5)
