"""Per-turn correlation for Zettlab preset commands run through terminal."""

from gateway import session_context
from tools.environments import local


def test_terminal_env_injects_current_zettlab_turn_id_and_clears_stale_value():
    try:
        session_context.set_zettlab_turn_id("turn-current-123")
        command = local._with_zettlab_turn_id("run-skill")
        assert command == "export ZETTLAB_TURN_ID=turn-current-123\nrun-skill"

        session_context.set_zettlab_turn_id("")
        assert local._with_zettlab_turn_id("run-skill") == "run-skill"
    finally:
        session_context.set_zettlab_turn_id("")


def test_local_terminal_process_receives_zettlab_turn_id_flow(tmp_path):
    try:
        session_context.set_zettlab_turn_id("turn-flow-456")
        env = local.LocalEnvironment(cwd=str(tmp_path), timeout=10)
        try:
            current = env.execute('printf "%s" "$ZETTLAB_TURN_ID"')
            session_context.set_zettlab_turn_id("")
            cleared = env.execute('printf "%s" "${ZETTLAB_TURN_ID-absent}"')
            session_context.set_zettlab_turn_id("turn-flow-789")
            next_turn = env.execute('printf "%s" "$ZETTLAB_TURN_ID"')
        finally:
            env.cleanup()
    finally:
        session_context.set_zettlab_turn_id("")

    assert current["returncode"] == 0
    assert current["output"] == "turn-flow-456"
    assert cleared["returncode"] == 0
    assert cleared["output"] == "absent"
    assert next_turn["returncode"] == 0
    assert next_turn["output"] == "turn-flow-789"
