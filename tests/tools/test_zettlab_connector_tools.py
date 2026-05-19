from tools.zettlab_connector_tools import _normalise_tool_schema


def test_normalise_tool_schema_drops_null_required_fields():
    schema = {
        "type": "object",
        "properties": {
            "filter": {
                "type": "object",
                "properties": {"state": {"type": "string"}},
                "required": None,
            }
        },
        "required": None,
    }

    assert _normalise_tool_schema(schema) == {
        "type": "object",
        "properties": {
            "filter": {
                "type": "object",
                "properties": {"state": {"type": "string"}},
            }
        },
    }
