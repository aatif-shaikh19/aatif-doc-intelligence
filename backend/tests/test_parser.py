import pytest

from app.services.parser import PDFParseError, extract_pages

from tests.helpers import create_corrupt_pdf, create_pdf


def test_extract_pages_from_valid_pdf(tmp_path):
    pdf_path = create_pdf(tmp_path, "valid.pdf", ["First page text", "Second page text"])

    pages = extract_pages(str(pdf_path), "doc-1", "valid.pdf")

    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert pages[0].doc_id == "doc-1"
    assert pages[0].filename == "valid.pdf"
    assert "First page text" in pages[0].text
    assert pages[1].page_number == 2
    assert "Second page text" in pages[1].text


def test_extract_pages_skips_empty_text_page(tmp_path):
    pdf_path = create_pdf(tmp_path, "mixed.pdf", ["Text on page 1", "", "Text on page 3"])

    pages = extract_pages(str(pdf_path), "doc-2", "mixed.pdf")

    assert len(pages) == 2
    assert [page.page_number for page in pages] == [1, 3]
    assert all(page.text.strip() for page in pages)


def test_extract_pages_rejects_corrupt_pdf(tmp_path):
    pdf_path = create_corrupt_pdf(tmp_path, "corrupt.pdf")

    with pytest.raises(PDFParseError):
        extract_pages(str(pdf_path), "doc-3", "corrupt.pdf")


def test_extract_pages_rejects_encrypted_pdf(tmp_path):
    try:
        pdf_path = create_pdf(tmp_path, "encrypted.pdf", ["Secret text"], encrypted=True)
    except RuntimeError:
        pytest.skip("Encrypted PDF generation is not supported in this environment")

    with pytest.raises(PDFParseError):
        extract_pages(str(pdf_path), "doc-4", "encrypted.pdf")