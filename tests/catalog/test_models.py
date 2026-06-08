from studio.catalog.models import STATUS_ORDER, next_status, fail_status

def test_status_order_is_linear():
    assert STATUS_ORDER == ["discovered","downloaded","stitched","detected","scened",
                            "visioned","grouped","beated","scripted","voiced","planned"]
def test_next_status_advances():
    assert next_status("downloaded") == "stitched"
    assert next_status("planned") is None
def test_fail_status():
    assert fail_status("stitched") == "stitched_failed"
