from __future__ import annotations

import importlib.util
import tarfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CHECK_SCRIPT = REPO_ROOT / "scripts" / "check_zpk_secrets.py"


def _module():
    spec = importlib.util.spec_from_file_location("check_zpk_secrets", CHECK_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_secret_scan_rejects_key_without_returning_value(tmp_path):
    secret = "sk-lf-12345678901234567890"
    (tmp_path / "config.env").write_text(f"HERMES_LANGFUSE_SECRET_KEY={secret}\n")

    findings = _module().scan_tree(tmp_path)

    assert (Path("config.env"), "langfuse-project-key") in findings
    assert secret not in repr(findings)


def test_secret_scan_accepts_relay_source_and_empty_assignment(tmp_path):
    (tmp_path / "config.yaml").write_text(
        'mode: relay\nbase_url: http://127.0.0.1:19092\ningestion_key: ""\n'
    )
    plugin = tmp_path / "plugin.py"
    plugin.write_text('PREFIX = "sk-lf-"\nPLACEHOLDER = "relay-secret"\n')

    assert _module().scan_tree(tmp_path) == []


def test_secret_scan_accepts_schema_objects_and_known_oauthlib_example(tmp_path):
    (tmp_path / "discovery.json").write_text(
        '{"public_key":{"description":"Schema field metadata"}}\n'
    )
    (tmp_path / "oauthlib_example.py").write_text(
        "Authorization: Basic czZCaGRSa3F0MzpnWDFmQmF0M2JW\n"
    )

    assert _module().scan_tree(tmp_path) == []


def test_secret_scan_rejects_basic_headers_and_common_config_suffixes(tmp_path):
    (tmp_path / "runtime.env").write_text("LANGFUSE_BASIC_AUTH=Basic YTpi\n")
    (tmp_path / "otel.properties").write_text(
        "OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic%20YTpi\n"
    )
    (tmp_path / "settings.toml").write_text('secret_key = "opaque-value"\n')
    (tmp_path / "settings.json").write_text('{"ingestion_key":"opaque-json"}\n')

    findings = _module().scan_tree(tmp_path)

    assert (Path("runtime.env"), "static-basic-auth") in findings
    assert (Path("otel.properties"), "static-basic-auth") in findings
    assert (Path("settings.toml"), "nonempty-monitoring-key-assignment") in findings
    assert (Path("settings.json"), "nonempty-monitoring-key-assignment") in findings


def test_secret_scan_inspects_final_zpk_archive(tmp_path):
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "config.env").write_text("LANGFUSE_BASIC_AUTH=Basic YTpi\n")
    archive_path = tmp_path / "test.zpk"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(payload / "config.env", arcname="payload/config.env")

    findings = _module().scan_path(archive_path)

    assert (Path("payload/config.env"), "static-basic-auth") in findings
