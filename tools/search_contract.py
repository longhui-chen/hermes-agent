"""Validation shared by the native search tool and its NAS HTTP client."""


def normalize_search_query(scope, pattern, modes, media_type="", region=""):
    if scope not in ("nas", "workspace"):
        raise ValueError("scope must be 'nas' or 'workspace'.")
    if not isinstance(pattern, str):
        raise ValueError("pattern must be a string; use '' for metadata-only NAS queries.")
    if not isinstance(modes, list) or any(
        not isinstance(mode, str) or mode not in ("name", "content", "semantic")
        for mode in modes
    ):
        raise ValueError("modes must be an explicit array of name/content/semantic, or [].")
    if not isinstance(region, str) or not isinstance(media_type, str):
        raise ValueError("region and media_type must be strings.")
    if media_type not in ("", "image", "video", "media"):
        raise ValueError("media_type must be image, video or media; omit it for all file types.")
    pattern = pattern.strip()
    region = region.strip()
    modes = [mode for mode in ("name", "content", "semantic") if mode in modes]
    if bool(pattern) != bool(modes):
        raise ValueError("Non-empty pattern requires explicit modes; empty pattern requires modes=[].")
    if "semantic" in modes and not media_type:
        raise ValueError("semantic requires media_type=image, video or media.")
    if scope == "workspace" and (
        modes not in (["name"], ["content"]) or media_type or region
    ):
        raise ValueError("workspace supports one of modes=['name'] or ['content'], without media_type or region.")
    return pattern, modes, media_type, region
