from studio.sources import base


def test_register_and_get():
    @base.register
    class Dummy(base.SourceAdapter):
        id = "dummy"
        capabilities = base.Capability.DOWNLOAD

        def series_meta(self, u): ...

        def list_chapters(self, u):
            return []

        def download(self, ch, d):
            return []

    assert isinstance(base.get("dummy"), Dummy)
    assert base.Capability.DOWNLOAD in base.get("dummy").capabilities


def test_slugify():
    assert base.slugify("The Beginning: After/End!") == "the-beginning-after-end"
