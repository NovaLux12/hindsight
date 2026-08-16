"""Retain ingestion must not store null metadata values (issue #3209).

The retain API accepts arbitrary JSON metadata; a null value (e.g.
{"ocr_engine": null}) stored verbatim poisons the read path, which validates
MemoryFact.metadata as dict[str, str] and made every recall fail for the
affected rows. RetainContent drops null-valued keys at construction, so
facts extracted from it (metadata=content.metadata in fact_extraction) stay
canonical on the write side; the read path drops nulls again for legacy rows.
Non-string values are preserved as-is here and coerced by the read path.
"""

from hindsight_api.engine.retain.orchestrator import _build_contents
from hindsight_api.engine.retain.types import RetainContent


def test_retain_content_drops_null_metadata_values():
    content = RetainContent(content="hi", metadata={"ocr_engine": None, "source": "slack"})
    assert content.metadata == {"source": "slack"}


def test_retain_content_keeps_non_null_values_as_given():
    content = RetainContent(content="hi", metadata={"n": 5, "source": "slack"})
    assert content.metadata == {"n": 5, "source": "slack"}


def test_build_contents_normalizes_null_metadata_from_api():
    """The ingestion path accepts JSON null metadata; stored facts must not
    carry null values (regression for the reported retain-with-null case)."""
    contents = _build_contents(
        [{"content": "hi", "metadata": {"ocr_engine": None, "n": 5, "source": "slack"}}],
        None,
    )
    assert contents[0].metadata == {"n": 5, "source": "slack"}
