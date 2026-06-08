from studio.paths import resolve_rel


def test_resolve_rel_after_move(tmp_path):
    man = tmp_path / "a" / "manifest.stitch.json"
    man.parent.mkdir(parents=True)
    assert resolve_rel(man, "stitch_chunks/chunk_0001.jpg") == man.parent / "stitch_chunks/chunk_0001.jpg"
    assert resolve_rel(man, ".") == man.parent
    assert str(resolve_rel(man, "/abs/legacy.jpg")) == "/abs/legacy.jpg"
