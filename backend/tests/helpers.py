from pathlib import Path

import fitz


def create_pdf(tmp_path: Path, filename: str, page_texts: list[str], encrypted: bool = False) -> Path:
    path = tmp_path / filename
    document = fitz.open()
    for page_text in page_texts:
        page = document.new_page()
        if page_text:
            page.insert_text((72, 72), page_text, fontsize=12)

    if encrypted:
        try:
            document.save(
                str(path),
                encryption=fitz.PDF_ENCRYPT_AES_256,
                owner_pw="owner",
                user_pw="user",
            )
        except Exception as exc:
            document.close()
            raise RuntimeError("encrypted PDF generation failed") from exc
    else:
        document.save(str(path))

    document.close()
    return path


def create_corrupt_pdf(tmp_path: Path, filename: str) -> Path:
    path = tmp_path / filename
    path.write_bytes(b"%PDF-1.4\n%corrupt\n1 0 obj\n<<")
    return path