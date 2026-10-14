from cfb.ingest.schema_check import compare, model_fields

from conftest import load_fixture


def test_reads_vendored_model_fields():
    fields = model_fields("passing_play")
    assert fields["playId"].required and not fields["playId"].nullable
    assert fields["airYards"].nullable
    assert "attributionStatus" not in fields  # brief assumed it; spec and live data say no
    assert "attributionStatus" in model_fields("rushing_play")


def test_detects_each_kind_of_mismatch():
    row = dict(load_fixture("plays_2024_401628322")[0])
    row.pop("wallclock")  # missing
    row["surprise"] = 1  # extra
    row["offense"] = None  # non-nullable null
    row["down"] = "1"  # wrong type
    diff = compare([row], "play")
    assert diff.missing == ["wallclock"]
    assert diff.extra == ["surprise"]
    assert diff.null_violations == {"offense": 1}
    assert diff.type_mismatches == {"down": "expected int, saw str"}
    assert not diff.clean


def test_enum_and_nested_fields_typed_correctly():
    assert compare(load_fixture("rushing_plays_2025_401752665"), "rushing_play").clean
    bad = dict(load_fixture("rushing_plays_2025_401752665")[0], clock="12:00")
    assert "clock" in compare([bad], "rushing_play").type_mismatches


def test_every_registry_endpoint_has_a_vendored_model():
    from cfb.ingest.registry import ENDPOINTS
    from cfb.ingest.schema_check import MODEL_BY_SLUG, VENDOR_MODELS

    assert {s.slug for s in ENDPOINTS} <= MODEL_BY_SLUG.keys()
    assert all((VENDOR_MODELS / f"{m}.py").is_file() for m in MODEL_BY_SLUG.values())
