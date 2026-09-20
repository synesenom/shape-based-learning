from shapeprim.data.objects import CLASS_NAMES, CLASS_TEMPLATES
from shapeprim.data.primitives import PRIMITIVE_TYPES


def test_all_expected_classes_present():
    expected = {
        "bicycle", "car", "truck", "cat_face", "house",
        "tree", "arrow_sign", "snowman", "person", "fish",
    }
    assert set(CLASS_NAMES) == expected


def test_every_template_has_at_least_two_parts():
    for name, template in CLASS_TEMPLATES.items():
        assert len(template.parts) >= 2, name


def test_all_part_types_are_known_primitives():
    for template in CLASS_TEMPLATES.values():
        for part in template.parts:
            assert part.type in PRIMITIVE_TYPES


def test_part_coords_normalized_within_object_bbox():
    for template in CLASS_TEMPLATES.values():
        for part in template.parts:
            assert 0.0 <= part.cx <= 1.0
            assert 0.0 <= part.cy <= 1.0
            assert 0.0 < part.width <= 1.0
            assert 0.0 < part.height <= 1.0


def test_tree_and_arrow_sign_share_shape_vocabulary_but_differ_in_arrangement():
    tree = CLASS_TEMPLATES["tree"]
    arrow = CLASS_TEMPLATES["arrow_sign"]

    assert {p.type for p in tree.parts} == {"rectangle", "triangle"}
    assert {p.type for p in arrow.parts} == {"rectangle", "triangle"}

    tree_rect = next(p for p in tree.parts if p.type == "rectangle")
    tree_tri = next(p for p in tree.parts if p.type == "triangle")
    arrow_rect = next(p for p in arrow.parts if p.type == "rectangle")
    arrow_tri = next(p for p in arrow.parts if p.type == "triangle")

    # tree: rectangle (trunk) is below the triangle (crown)
    assert tree_rect.cy > tree_tri.cy
    # arrow sign: rectangle is above the triangle -- inverted arrangement
    assert arrow_rect.cy < arrow_tri.cy


def test_car_and_truck_share_vocabulary_with_different_counts():
    car = CLASS_TEMPLATES["car"]
    truck = CLASS_TEMPLATES["truck"]
    car_counts = {t: sum(1 for p in car.parts if p.type == t) for t in PRIMITIVE_TYPES}
    truck_counts = {t: sum(1 for p in truck.parts if p.type == t) for t in PRIMITIVE_TYPES}
    assert car_counts["circle"] == 2
    assert truck_counts["circle"] == 3
    assert car_counts != truck_counts
