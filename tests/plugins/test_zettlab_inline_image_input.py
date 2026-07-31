from __future__ import annotations

import base64
import os

import pytest


PNG = b"\x89PNG\r\n\x1a\ninline-image"
JPEG = b"\xff\xd8\xffinline-image"
WEBP = b"RIFF\x0c\x00\x00\x00WEBPinline-image"


def _capability(limit: int = 1024):
    return {
        "modalities": ["text", "image"],
        "_type_limits": {"max_inline_image_bytes": limit},
    }


@pytest.mark.parametrize(
    ("raw", "mime"),
    [(PNG, "image/png"), (JPEG, "image/jpeg"), (WEBP, "image/webp")],
)
def test_inline_image_input_encodes_supported_local_file(tmp_path, monkeypatch, raw, mime):
    from agent import file_safety
    from plugins import zettlab_media_client as client

    image_path = tmp_path / "source.bin"
    image_path.write_bytes(raw)
    checked = []
    monkeypatch.setattr(file_safety, "raise_if_read_blocked", checked.append)

    got = client.inline_image_input(str(image_path), None, _capability())

    assert checked == [str(image_path)]
    assert got == f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def test_inline_image_input_accepts_matching_data_uri():
    from plugins import zettlab_media_client as client

    value = f"data:image/png;base64,{base64.b64encode(PNG).decode('ascii')}"
    assert client.inline_image_input(value, None, _capability()) == value


def test_inline_image_input_normalizes_gateway_modality():
    from plugins import zettlab_media_client as client

    value = f"data:image/png;base64,{base64.b64encode(PNG).decode('ascii')}"
    capability = _capability()
    capability["modalities"] = [" IMAGE "]

    assert client.inline_image_input(value, None, capability) == value


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("https://example.com/source.png", "local image path or data URI"),
        ("data:image/jpeg;base64," + base64.b64encode(PNG).decode("ascii"), "does not match"),
        ("data:image/png;base64,not-base64!", "valid base64"),
    ],
)
def test_inline_image_input_rejects_unsafe_or_invalid_values(value, message):
    from plugins import zettlab_media_client as client

    with pytest.raises(client.ZettlabMediaError, match=message):
        client.inline_image_input(value, None, _capability())


def test_inline_image_input_rejects_oversize_and_multiple_inputs():
    from plugins import zettlab_media_client as client

    value = f"data:image/png;base64,{base64.b64encode(PNG).decode('ascii')}"
    with pytest.raises(client.ZettlabMediaError, match="exceeds maximum size"):
        client.inline_image_input(value, None, _capability(limit=len(PNG) - 1))
    with pytest.raises(client.ZettlabMediaError, match="exactly one image"):
        client.inline_image_input(value, [value], _capability())


def test_inline_image_input_fails_closed_without_gateway_limit():
    from plugins import zettlab_media_client as client

    value = f"data:image/png;base64,{base64.b64encode(PNG).decode('ascii')}"
    with pytest.raises(client.ZettlabMediaError, match="not enabled"):
        client.inline_image_input(value, None, {"modalities": ["text", "image"]})


def test_inline_image_input_clamps_gateway_limit_to_local_hard_cap():
    from plugins import zettlab_media_client as client

    assert client._inline_image_limit(_capability(limit=16 * 1024 * 1024)) == (
        client.MAX_INLINE_IMAGE_BYTES
    )


def test_inline_image_input_rejects_oversize_file_before_encoding(tmp_path, monkeypatch):
    from plugins import zettlab_media_client as client

    image_path = tmp_path / "oversize.png"
    image_path.write_bytes(PNG + b"x" * client.MAX_INLINE_IMAGE_BYTES)
    monkeypatch.setattr(
        base64,
        "b64encode",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("oversize file must be rejected before encoding")
        ),
    )

    with pytest.raises(client.ZettlabMediaError, match="exceeds maximum size"):
        client.inline_image_input(
            str(image_path),
            None,
            _capability(limit=16 * 1024 * 1024),
        )


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO is unavailable")
def test_inline_image_input_rejects_fifo_without_blocking(tmp_path):
    from plugins import zettlab_media_client as client

    fifo_path = tmp_path / "image.fifo"
    os.mkfifo(fifo_path)

    with pytest.raises(client.ZettlabMediaError, match="regular file"):
        client.inline_image_input(str(fifo_path), None, _capability())


def test_media_http_session_accepts_base64_sized_request(monkeypatch):
    from plugins import zettlab_media_client as client

    sentinel = object()
    monkeypatch.setattr(client._HTTP_WORKER, "request", lambda *args, **kwargs: sentinel)
    got = client._SESSION.post(
        "http://127.0.0.1:9090/media/generation-jobs",
        json={"input_image": "A" * (2 * 1024 * 1024)},
        timeout=1,
        allow_redirects=False,
    )
    assert got is sentinel

    with pytest.raises(client.ZettlabMediaError, match="request exceeds maximum size"):
        client._SESSION.post(
            "http://127.0.0.1:9090/media/generation-jobs",
            json={"input_image": "A" * client.MAX_MEDIA_REQUEST_BYTES},
            timeout=1,
            allow_redirects=False,
        )
