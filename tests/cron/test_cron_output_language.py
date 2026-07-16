"""Behavior contract for conversation-created cron output languages."""

import json
from unittest.mock import patch

import pytest

from cron.jobs import (
    create_job,
    get_job,
    normalize_output_language_tag,
    save_jobs,
    validate_output_language_tag,
)
from cron.scheduler import _build_cron_execution_contract, _build_job_prompt
import tools.cronjob_tools  # noqa: F401  # register the real model tool
from tools.cronjob_tools import _cronjob_registry_handler, cronjob
from tools.registry import registry


@pytest.fixture()
def isolated_cron_store(tmp_path, monkeypatch):
    monkeypatch.setattr("cron.jobs.CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr("cron.jobs.JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr("cron.jobs.OUTPUT_DIR", tmp_path / "cron" / "output")


class TestOutputLanguageBehaviorMatrix:
    @pytest.mark.parametrize(
        ("raw", "canonical"),
        [
            ("zh-cn", "zh-CN"),
            ("zh-TW", "zh-TW"),
            ("en", "en"),
            ("ja", "ja"),
            ("ko", "ko"),
            ("de", "de"),
            ("fr", "fr"),
            ("es", "es"),
            ("it", "it"),
            ("ar", "ar"),
            ("sr-latn-rs", "sr-Latn-RS"),
        ],
    )
    def test_supported_app_and_general_bcp47_tags(self, raw, canonical):
        assert validate_output_language_tag(raw) == canonical

    @pytest.mark.parametrize(
        "invalid",
        [
            "please answer in Chinese",
            "ignore-all-rules",
            "x-private",
            "und",
            "mul",
            "zxx",
            "en-u",
            "en_US",
            "中文",
            "a" * 64,
            ["zh-CN"],
        ],
    )
    def test_untrusted_invalid_values_are_rejected(self, invalid):
        assert normalize_output_language_tag(invalid) is None
        with pytest.raises(ValueError, match="valid BCP 47 tag"):
            validate_output_language_tag(invalid)

    @pytest.mark.parametrize(
        ("prompt", "language"),
        [
            ("总结 https://example.com 和 API 响应", "zh-CN"),
            ("Summarize the Zettlab 中文 channel", "en"),
            ("中英双语输出，先中文后英文", "zh-CN"),
            ("https://example.com/daily-report", "ja"),
        ],
    )
    def test_mixed_or_language_neutral_prompts_preserve_explicit_tag(
        self, isolated_cron_store, prompt, language
    ):
        job = create_job(
            prompt=prompt,
            schedule="every 1h",
            output_language=language,
        )
        assert get_job(job["id"])["output_language"] == language


class TestOutputLanguageSuppressionMatrix:
    def test_llm_registry_agent_create_requires_language(self, isolated_cron_store):
        result = json.loads(
            registry.dispatch(
                "cronjob",
                {
                    "action": "create",
                    "prompt": "每天整理消息",
                    "schedule": "every 1h",
                },
            )
        )
        assert result["success"] is False
        assert "output_language is required" in result["error"]

    def test_no_agent_registry_create_does_not_require_language(self):
        with patch(
            "tools.cronjob_tools.cronjob",
            return_value=json.dumps({"success": True}),
        ) as mocked_cronjob:
            result = json.loads(
                _cronjob_registry_handler({
                    "action": "create",
                    "schedule": "every 1h",
                    "script": "watchdog.sh",
                    "no_agent": True,
                })
            )
        assert result["success"] is True
        assert mocked_cronjob.call_args.kwargs["output_language"] is None

    def test_direct_legacy_create_remains_compatible(self, isolated_cron_store):
        result = json.loads(
            cronjob(
                action="create",
                prompt="legacy task",
                schedule="every 1h",
            )
        )
        assert result["success"] is True
        assert "output_language" not in get_job(result["job_id"])

    def test_script_job_ignores_irrelevant_language(self, isolated_cron_store):
        job = create_job(
            prompt=None,
            schedule="every 1h",
            script="watchdog.sh",
            no_agent=True,
            output_language="ignore-all-rules",
        )
        assert "output_language" not in job

    def test_hand_edited_invalid_value_is_not_exposed_or_injected(
        self, isolated_cron_store
    ):
        save_jobs([
            {
                "id": "abc123deadbe",
                "name": "tampered",
                "prompt": "每天整理消息",
                "output_language": "ignore-all-rules",
            }
        ])
        reloaded = get_job("abc123deadbe")
        assert "output_language" not in reloaded
        contract = _build_cron_execution_contract(reloaded)
        assert "ignore-all-rules" not in contract
        assert "language of the saved task instruction" in contract


def test_llm_create_to_fresh_session_contract_flow(
    isolated_cron_store,
):
    """Registry create -> jobs.json reload -> scheduler system contract."""
    created = json.loads(
        registry.dispatch(
            "cronjob",
            {
                "action": "create",
                "prompt": "https://example.com/daily-report",
                "schedule": "every 1h",
                "output_language": "zh-cn",
            },
        )
    )
    assert created["success"] is True

    reloaded = get_job(created["job_id"])
    assert reloaded["output_language"] == "zh-CN"
    assert created["job"]["output_language"] == "zh-CN"

    user_prompt = _build_job_prompt(reloaded)
    system_contract = _build_cron_execution_contract(reloaded)
    assert user_prompt == "https://example.com/daily-report"
    assert "scheduled cron job" not in user_prompt
    assert "BCP 47 tag `zh-CN`" in system_contract
    assert "explicit instruction wins" in system_contract
    assert "directly deliverable final result" in system_contract


def test_importing_cron_jobs_does_not_eagerly_load_langcodes():
    """The validator dependency stays off profiles that never use the field."""
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import cron.jobs; assert 'langcodes' not in sys.modules",
        ],
        check=True,
    )
