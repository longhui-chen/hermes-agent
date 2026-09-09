from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from cron.execution_context import current_execution, execution_scope


def test_identity_is_private_immutable_and_restored(monkeypatch):
    monkeypatch.setenv("ZETTLAB_CAMERA_JOB_ID", "forged")
    monkeypatch.setenv("ZETTLAB_CAMERA_EXECUTION_ID", "forged")
    assert current_execution() is None
    with execution_scope("job", "run", Path("/profile-a")) as identity:
        assert current_execution() is identity
        with pytest.raises(FrozenInstanceError):
            identity.job_id = "other"
        with pytest.raises(RuntimeError):
            with execution_scope("nested", "run2", Path("/profile-b")):
                raise RuntimeError("interrupted")
        assert current_execution() is identity
    assert current_execution() is None


@pytest.mark.parametrize("invalid", ["", "../job", "a/b", "a\x00b", "x" * 129, None])
def test_invalid_identity_shadows_outer_scope(invalid):
    with execution_scope("outer", "run", Path("/profile")) as outer:
        for job, run in [(invalid, "run"), ("job", invalid)]:
            with execution_scope(job, run, Path("/profile")):
                assert current_execution() is None
        assert current_execution() is outer


def test_parallel_worker_contexts_do_not_mix_profiles():
    contexts = []
    for n in range(16):
        with execution_scope(f"job{n}", f"run{n}", Path(f"/profile{n}")):
            contexts.append(copy_context())
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda ctx: ctx.run(current_execution), contexts))
        assert pool.submit(current_execution).result() is None
    for n, identity in enumerate(results):
        assert (identity.job_id, identity.execution_id, identity.profile_home) == (
            f"job{n}", f"run{n}", Path(f"/profile{n}")
        )
    assert current_execution() is None
