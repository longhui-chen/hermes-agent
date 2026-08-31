"""Helpers for OpenAI-compatible chat ``response_format`` handling."""

from __future__ import annotations

from typing import Any, Dict, Optional


class ResponseFormatValidationError(ValueError):
    """调用方给的 ``response_format`` 本身不合法 —— 这是**请求校验**错误。

    ⭐ 为什么要有这个类型:出站层原先靠 ``"response_format" in str(e)``(后来
    收窄成 ``startswith``)去判「这条 ValueError 是不是 400 类请求校验」。
    那是**形状判据**:``_run_agent`` 内部任何一条恰好提到 response_format 的
    ``ValueError``(例如 "response_format resolver crashed at /volume1/…")
    都会被当成请求校验错误、原文回显给用户。
    ⇒ 换成**类型**这个闭集判据:只有本模块**有意**抛出的才是 400。

    ⭐ 仍继承 ``ValueError``,所以既有的 ``except ValueError`` 调用点行为不变。
    """


def validate_chat_response_format(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, dict):
        return "Invalid 'response_format' field"
    fmt_type = value.get("type")
    if fmt_type in {"text", "json_object"}:
        return None
    if fmt_type != "json_schema":
        return "Unsupported 'response_format.type'"
    schema_payload = value.get("json_schema")
    if not isinstance(schema_payload, dict):
        return "'response_format.json_schema' must be an object"
    if not isinstance(schema_payload.get("schema"), dict):
        return "'response_format.json_schema.schema' must be an object"
    name = schema_payload.get("name")
    if name is not None and not isinstance(name, str):
        return "'response_format.json_schema.name' must be a string"
    strict = schema_payload.get("strict")
    if strict is not None and not isinstance(strict, bool):
        return "'response_format.json_schema.strict' must be a boolean"
    return None


def response_format_requires_structured_output(value: Any) -> bool:
    return isinstance(value, dict) and value.get("type") in {"json_object", "json_schema"}


def responses_text_format_from_chat_response_format(response_format: Any) -> Dict[str, Any] | None:
    if not isinstance(response_format, dict):
        raise ResponseFormatValidationError("response_format must be an object.")
    fmt_type = response_format.get("type")
    if fmt_type == "text":
        return None
    if fmt_type == "json_object":
        return {"type": "json_object"}
    if fmt_type != "json_schema":
        raise ResponseFormatValidationError("response_format.type must be 'text', 'json_object' or 'json_schema'.")
    schema_payload = response_format.get("json_schema")
    if not isinstance(schema_payload, dict):
        raise ResponseFormatValidationError("response_format.json_schema must be an object.")
    schema = schema_payload.get("schema")
    if not isinstance(schema, dict):
        raise ResponseFormatValidationError("response_format.json_schema.schema must be an object.")
    text_format: Dict[str, Any] = {
        "type": "json_schema",
        "name": schema_payload.get("name") or "structured_output",
        "schema": schema,
    }
    if "strict" in schema_payload:
        strict = schema_payload.get("strict")
        if not isinstance(strict, bool):
            raise ResponseFormatValidationError("response_format.json_schema.strict must be a boolean.")
        text_format["strict"] = strict
    return text_format
