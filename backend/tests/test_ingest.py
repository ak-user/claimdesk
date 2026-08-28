from claimdesk.ingest import parse_booklet, split_sentences
from claimdesk.store import load_booklet

EXPECTED_SECTIONS = [
    "1.1", "1.2", "2.1", "2.2", "2.3", "3.1", "3.2", "3.3", "3.4",
    "4.1", "4.2", "4.3", "4.4", "4.5", "4.6", "5.1", "5.2", "5.3", "5.4",
    "6", "7", "8", "9",
]  # fmt: skip


def test_one_chunk_per_section_with_body_text():
    chunks = parse_booklet(load_booklet())
    assert [c.section for c in chunks] == EXPECTED_SECTIONS


def test_chunks_keep_titles_and_parents():
    chunks = {c.section: c for c in parse_booklet(load_booklet())}
    assert chunks["4.2"].title == "Physiotherapy"
    assert chunks["4.2"].parent_title == "Health benefits"
    assert chunks["6"].title == "Exclusions"
    assert chunks["6"].parent_title == ""


def test_disclaimer_and_front_matter_are_not_indexed():
    text = " ".join(c.text for c in parse_booklet(load_booklet()))
    assert "Fictional sample data" not in text
    assert "Plan sponsor:" not in text


def test_long_sections_are_split_on_sentences():
    sentence = "The plan pays eighty percent of eligible expenses for this service. "
    markdown = "## 1. Long\n\n" + sentence * 30
    chunks = parse_booklet(markdown)
    assert [c.section for c in chunks] == ["1", "1-2"]
    assert all(len(c.text.split()) <= 180 for c in chunks)


def test_split_sentences():
    assert split_sentences("One. Two? Three! 4 four.") == ["One.", "Two?", "Three!", "4 four."]
