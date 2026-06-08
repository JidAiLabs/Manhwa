"""Pure unit tests for boxes_to_panels_norm — no ultralytics needed."""

from studio.detect.yolo_panels import boxes_to_panels_norm


def test_convert_order_and_sort():
    # (x1,y1,x2,y2) pixels on a 100x200 image, out of order
    # box A: x1=0,y1=150,x2=100,y2=200 → ymin=150/200=0.75, xmin=0/100=0.0, ymax=200/200=1.0, xmax=100/100=1.0
    # box B: x1=0,y1=0,  x2=100,y2=50  → ymin=0/200=0.0,  xmin=0/100=0.0, ymax=50/200=0.25, xmax=100/100=1.0
    px = [(0, 150, 100, 200), (0, 0, 100, 50)]  # out of top-to-bottom order
    out = boxes_to_panels_norm(px, w=100, h=200)
    assert out == [[0.0, 0.0, 0.25, 1.0], [0.75, 0.0, 1.0, 1.0]], f"got {out}"


def test_single_box():
    px = [(50, 10, 150, 90)]  # x1=50,y1=10,x2=150,y2=90 on 200x100 image
    out = boxes_to_panels_norm(px, w=200, h=100)
    assert out == [[0.1, 0.25, 0.9, 0.75]], f"got {out}"


def test_empty():
    assert boxes_to_panels_norm([], w=100, h=200) == []


def test_sort_is_ymin_ascending():
    px = [(0, 80, 10, 90), (0, 10, 10, 20), (0, 40, 10, 50)]
    out = boxes_to_panels_norm(px, w=10, h=100)
    ymins = [row[0] for row in out]
    assert ymins == sorted(ymins), f"not sorted: {ymins}"


def test_precision_rounded():
    # Ensure values are rounded to 6 decimal places (no floating-point surprises)
    px = [(1, 1, 2, 2)]
    out = boxes_to_panels_norm(px, w=3, h=3)
    for row in out:
        for v in row:
            assert v == round(v, 6), f"value {v} not rounded to 6dp"
