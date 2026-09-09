import copy

import pytest

from tools.connector_setup_intent import normalize_connector_setup


VALID = {"resource_kind": "camera", "observation": {"camera_id": "cam-1", "duration_seconds": 60, "subject_kind": "person", "predicate": "appears"}}


def test_valid_observation_is_a_detached_proposal():
    source = copy.deepcopy(VALID)
    normalized = normalize_connector_setup(source)
    assert normalized == source
    source["observation"]["duration_seconds"] = 90
    assert normalized["observation"]["duration_seconds"] == 60


@pytest.mark.parametrize("field,value", [
    ("duration_seconds", 0), ("duration_seconds", True), ("duration_seconds", 0.5),
    ("duration_seconds", 2147483648), ("camera_id", "rtsp://secret@host"),
    ("subject_ref", ""), ("subject_ref", None), ("zone_id", "../zone"),
    ("min_duration_seconds", None), ("min_duration_seconds", 61),
    ("predicate", "enters_zone"), ("predicate", ["appears"]),
    ("owner_id", "owner"), ("origin_session_id", "memo"),
    ("background_vision_consent", True), ("schedule_ref", "job"),
])
def test_observation_rejects_invalid_or_authority_fields(field, value):
    source = copy.deepcopy(VALID)
    source["observation"][field] = value
    with pytest.raises(ValueError):
        normalize_connector_setup(source)


def test_ordinary_setup_remains_unchanged_and_observation_is_camera_only():
    assert normalize_connector_setup({"resource_kind": "camera"}) == {"resource_kind": "camera"}
    with pytest.raises(ValueError):
        normalize_connector_setup({**VALID, "resource_kind": "printer3d"})
