import pytest

from gateway.platforms.zet_agent_camera_semantic_arguments import (
    semantic_arguments_allowed,
    semantic_execution_timeout,
)


BASE = ["observe", "--policy-id", "12345678-1234-1234-1234-123456789abc", "--timeout-seconds", "65"]


@pytest.mark.parametrize("mode", [None, "finite", "periodic"])
def test_camera_observation_mode_is_explicit_and_keeps_budget(mode):
    args = BASE + (["--mode", mode] if mode else [])
    assert semantic_arguments_allowed(args)
    assert semantic_execution_timeout(args, 75, 600, 80) == 70
    with pytest.raises(ValueError):
        semantic_execution_timeout(args, 69, 600, 80)


@pytest.mark.parametrize("suffix", [
    ["--mode", "auto"], ["--mode", "periodic_observation"], ["--mode", ""],
    ["--mode", "periodic", "--mode", "finite"], ["--mode", "periodic", "--job-id", "other"],
    ["--mode", "periodic", "--duration", "3600"], ["--mode", "periodic", "--output", "/other"],
])
def test_camera_observation_mode_does_not_expand_authority(suffix):
    assert not semantic_arguments_allowed(BASE + suffix)


def test_periodic_mode_is_not_a_candidate_parameter():
    assert not semantic_arguments_allowed(["candidate", *BASE[1:3], "--mode", "periodic"])
