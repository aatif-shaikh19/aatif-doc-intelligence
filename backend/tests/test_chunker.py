from app.models.schemas import ParsedPage
from app.services.chunker import CHUNK_OVERLAP, CHUNK_SIZE, chunk_document


def test_chunker_preserves_short_page_metadata():
    page = ParsedPage(
        doc_id="doc-1",
        filename="short.pdf",
        page_number=1,
        text="Short page text.",
        character_count=len("Short page text."),
    )

    chunks = chunk_document([page])

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].doc_id == page.doc_id
    assert chunks[0].filename == page.filename
    assert chunks[0].page_number == page.page_number
    assert chunks[0].text == page.text


def test_chunker_splits_long_page_with_overlap():
    long_text = "".join(f"{index:05d}" for index in range(300))
    page = ParsedPage(
        doc_id="doc-2",
        filename="long.pdf",
        page_number=1,
        text=long_text,
        character_count=len(long_text),
    )

    chunks = chunk_document([page])

    assert len(chunks) > 1
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert chunks[1].text.startswith(long_text[CHUNK_SIZE - CHUNK_OVERLAP : CHUNK_SIZE - CHUNK_OVERLAP + 20])


def test_chunker_keeps_chunk_order_across_pages():
    page_one = ParsedPage(
        doc_id="doc-3",
        filename="ordered.pdf",
        page_number=1,
        text="First page content.",
        character_count=len("First page content."),
    )
    page_two = ParsedPage(
        doc_id="doc-3",
        filename="ordered.pdf",
        page_number=2,
        text="Second page content.",
        character_count=len("Second page content."),
    )

    chunks = chunk_document([page_one, page_two])

    assert [chunk.page_number for chunk in chunks] == [1, 2]
    assert [chunk.chunk_index for chunk in chunks] == [0, 0]