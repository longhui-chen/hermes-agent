"""Cron sessions persist a clean operator-typed prompt, not the LLM payload.

Before this change, ``_build_job_prompt``'s output (cron_hint preamble +
script-output framing + skill wrappers + the operator's prompt) landed
verbatim in ``sessions.messages`` role=user content. The App's home
list and per-session history then surfaced the entire
``[IMPORTANT: You are running as a scheduled cron job. ...]`` block as
if the user had typed it.

Fix: ``run_job`` now hands a clean operator-only prompt to
``run_conversation(persist_user_message=...)``.
``_apply_persist_user_message_override`` rewrites the in-memory messages
list before the persist flush so DB / JSONL log carry only the clean
string. LLM behavior is unchanged — it still sees the full LLM payload
as ``user_message``.

These tests pin ``_build_job_persist_prompt`` itself (the helper that
produces the clean string). End-to-end "history is clean after a real
cron run" coverage would need an integration harness with a live
SessionDB; this unit-level pin is the cheapest guard against the
preamble re-leaking on a future refactor.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cron.scheduler import _build_job_persist_prompt


class TestBuildJobPersistPrompt:
    def test_returns_user_prompt_verbatim(self):
        """Operator's job["prompt"] passes through untouched — no
        cron_hint preamble, no skill wrapper, no surrounding markdown.
        """
        job = {
            "id": "abc123",
            "name": "morning ping",
            "prompt": "提醒阿曾 ZZZ 例行节奏开始了",
        }
        assert _build_job_persist_prompt(job) == "提醒阿曾 ZZZ 例行节奏开始了"

    def test_strips_surrounding_whitespace(self):
        """Treat purely-whitespace prompts as effectively empty so the
        fallback kicks in instead of persisting a blank user bubble.
        """
        job = {"id": "abc123", "name": "blank", "prompt": "   \n\t  "}
        assert _build_job_persist_prompt(job) == "_(cron job: blank)_"

    def test_falls_back_to_name_when_prompt_empty(self):
        """Skill-only crons (no operator-supplied text) still get a
        non-empty persist string. job["name"] preferred for readability.
        """
        job = {"id": "abc123", "name": "weather check", "prompt": ""}
        assert _build_job_persist_prompt(job) == "_(cron job: weather check)_"

    def test_falls_back_to_id_when_name_missing(self):
        job = {"id": "abc123", "prompt": ""}
        assert _build_job_persist_prompt(job) == "_(cron job: abc123)_"

    def test_falls_back_to_unknown_when_id_and_name_missing(self):
        """Defensive: a malformed job dict (neither name nor id) should
        not crash run_job; the persist string just labels it 'unknown'.
        """
        assert _build_job_persist_prompt({"prompt": ""}) == "_(cron job: unknown)_"

    def test_skills_do_not_leak_into_persist_string(self):
        """Persist string must stay focused on the operator's text even
        when a skill is invoked. Skill content lives in the LLM payload
        only — surfacing it in the home list is what we're fixing.
        """
        job = {
            "id": "abc123",
            "name": "daily news",
            "prompt": "run today's digest",
            "skills": ["news-digest"],
        }
        result = _build_job_persist_prompt(job)
        assert result == "run today's digest"
        assert "news-digest" not in result
        assert "[IMPORTANT" not in result

    def test_script_output_does_not_leak_into_persist_string(self):
        """Likewise for prerun_script and context_from blocks: the LLM
        sees them framed in the user_message, but the home list shouldn't.
        ``_build_job_persist_prompt`` ignores those entirely.
        """
        job = {
            "id": "abc123",
            "name": "scripted",
            "prompt": "summarize the data",
            "script": "/some/script/path",
            "context_from": ["other-job-id"],
        }
        result = _build_job_persist_prompt(job)
        assert result == "summarize the data"
        assert "Script Output" not in result
        assert "Output from job" not in result
