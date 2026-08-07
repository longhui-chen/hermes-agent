from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def clear_media_capability_cache():
    """Keep one test's faked media catalog from leaking into the next.

    ``zettlab_media_client.get_capabilities`` caches successful responses per
    base URL, so without this a test that stubs a catalog would satisfy the
    next test's probe before its own stub was ever consulted.
    """
    from plugins import zettlab_media_client

    zettlab_media_client.invalidate_capability_cache()
    yield
    zettlab_media_client.invalidate_capability_cache()


@pytest.fixture
def patch_media_get(monkeypatch):
    """Route both media GET paths to one handler.

    Capability probes go through ``_CAPABILITY_TRANSPORT`` (an in-process,
    cancellable loopback GET) while job polling goes through ``_SESSION`` (the
    subprocess worker pool). Tests generally stub a single handler that serves
    both, so patch both rather than making every caller know which path it is
    exercising.
    """

    def _patch(client, handler):
        monkeypatch.setattr(client._SESSION, "get", handler)
        monkeypatch.setattr(client._CAPABILITY_TRANSPORT, "get", handler)

    return _patch
