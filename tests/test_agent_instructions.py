from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MAX_AGENTS_MD_BYTES = 16 * 1024


def test_root_agents_md_stays_within_autoload_budget():
    agents_md = REPO_ROOT / "AGENTS.md"
    actual_bytes = agents_md.stat().st_size

    assert actual_bytes <= MAX_AGENTS_MD_BYTES, (
        f"AGENTS.md is {actual_bytes} bytes; keep the automatically loaded entry "
        f"at or below {MAX_AGENTS_MD_BYTES} bytes and route details to focused docs"
    )
