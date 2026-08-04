"""Neutral SOUL.md templates seeded by the Hermes runtime.

Hermes is profile-agnostic: product personas such as Zettlab Memo are
materialized into a profile's ``SOUL.md`` by the profile owner (local-server,
an Agent Hub package, or the user). A missing SOUL therefore always gets the
same neutral identity slot, regardless of profile name.

Language selection mirrors ``agent.prompt_builder.get_agent_prompt_lang`` so
the seeded SOUL.md and runtime base prompt agree on language. Unset -> English.
"""

import os

DEFAULT_BASE_SOUL_MD_EN = (
    "# Agent SOUL\n\n"
    "This profile has not been given a specialized persona yet. Treat this file "
    "as an open identity slot: follow the user's current request, the shared "
    "Zettlab agent base prompt, and any future edits to this SOUL.md. Do not "
    "assume any named specialist identity unless this file, a template package, "
    "or the current user explicitly defines that identity."
)

DEFAULT_BASE_SOUL_MD_ZH = (
    "# Agent SOUL\n\n"
    "这个 profile 还没有写入专属人格。把这个文件视为开放的身份槽：遵循用户当前请求、"
    "共享的 Zettlab agent base prompt，以及之后写入本 SOUL.md 的内容。"
    "除非本文件、模板包或当前用户明确指定，否则不要默认自己是任何具名专家。"
)


def _resolve_lang(lang: str | None = None) -> str:
    """Resolve 'zh' or 'en' from an explicit arg or the env contract.

    Kept self-contained (no agent.* import) so the lightweight hermes_cli
    seeding path stays decoupled from the runtime agent package.
    """
    if lang:
        raw = lang.strip().lower()
    else:
        raw = (
            os.environ.get("HERMES_AGENT_LANG")
            or os.environ.get("ZETTLAB_AGENT_LANG")
            or ""
        ).strip().lower()
    if raw in ("zh", "cn", "zh-cn", "zh-hans", "chinese", "mandarin"):
        return "zh"
    return "en"


def base_soul_md(lang: str | None = None) -> str:
    """Return the neutral identity slot used by every unmaterialized profile."""
    return DEFAULT_BASE_SOUL_MD_ZH if _resolve_lang(lang) == "zh" else DEFAULT_BASE_SOUL_MD_EN


def default_soul_md(lang: str | None = None, profile: str | None = None) -> str:
    """Return the profile-agnostic default SOUL body.

    ``profile`` remains accepted for API compatibility, but names never select
    a product persona. A product owner must materialize that persona into the
    profile's SOUL.md before Hermes starts it.
    """
    _ = profile
    return base_soul_md(lang)


# Back-compat aliases used by installers and existing imports.
DEFAULT_SOUL_MD_EN = DEFAULT_BASE_SOUL_MD_EN
DEFAULT_SOUL_MD_ZH = DEFAULT_BASE_SOUL_MD_ZH
DEFAULT_SOUL_MD = DEFAULT_SOUL_MD_EN
