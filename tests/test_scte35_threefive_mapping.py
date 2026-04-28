from pathlib import Path

import pytest
import threefive


SPLICE_INSERT_OON_TRUE = "/DAvAAAAAAAA///wFAVIAACef+/+c2nALv4AUsz1AAAAAAAMAQpDVUVJAAABNWLbowo="
SPLICE_NULL = "/DARAAAAAAAAAP/wAAAAAHpPGuQ="


def test_threefive_imports_current_local_package():
    assert threefive.__version__ == "3.0.77"
    assert Path(threefive.__file__).as_posix().endswith(
        "/threefive_is_scte35/threefive/__init__.py"
    )

    for exported_name in ("Cue", "Stream", "TagParser", "Segment"):
        assert hasattr(threefive, exported_name), f"threefive missing {exported_name}"


@pytest.mark.parametrize(
    "payload, expected_command_name",
    [
        (SPLICE_NULL, "Splice Null"),
        (SPLICE_INSERT_OON_TRUE, "Splice Insert"),
    ],
)
def test_fixture_cues_decode_with_expected_command_names(payload, expected_command_name):
    cue = threefive.Cue(payload)

    assert cue.decode() is True
    assert cue.command.get()["name"] == expected_command_name


def test_splice_insert_fixture_exposes_command_fields_and_descriptors():
    cue = threefive.Cue(SPLICE_INSERT_OON_TRUE)

    assert cue.decode() is True

    command = cue.command.get()
    assert command["name"] == "Splice Insert"
    assert command["pts_time"] > 0
    assert command["break_duration"] == pytest.approx(60.293567)
    assert command["splice_event_id"] == 1207959710
    assert command["out_of_network_indicator"] is True
    assert cue.get_descriptors()
