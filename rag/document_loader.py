
"""Load text from uploaded files, page by page, with OCR fallback."""
from __future__ import annotations

from pathlib import Path


# Windows Tesseract installation path
TESSERACT_PATH = Path(
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def extract_pdf_with_ocr(path: Path) -> list[dict]:
    """Extract PDF text normally; use OCR when a page is image-based."""
    import fitz
    import pytesseract
    from PIL import Image
    import io

    if not TESSERACT_PATH.exists():
        raise FileNotFoundError(
            f"Tesseract not found at: {TESSERACT_PATH}"
        )

    pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_PATH)

    pages = []

    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()

            # OCR fallback for scanned/image-based pages
            if len(text) < 30:
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(2, 2),
                    alpha=False
                )
                image = Image.open(
                    io.BytesIO(pix.tobytes("png"))
                )

                text = pytesseract.image_to_string(
                    image,
                    lang="eng",
                    config="--psm 3"
                ).strip()

            if text:
                pages.append({
                    "page": i,
                    "text": text
                })

    return pages


def load_document(path: str | Path) -> list[dict]:
    """Returns [{"page": int, "text": str}, ...]."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return extract_pdf_with_ocr(path)

    if suffix in {".txt", ".md"}:
        text = path.read_text(
            encoding="utf-8-sig",
            errors="ignore"
        )

        parts = text.split("\f") if "\f" in text else [text]

        return [
            {"page": i, "text": p.strip()}
            for i, p in enumerate(parts, start=1)
            if p.strip()
        ]

    raise ValueError(
        f"Unsupported file type: {suffix} (use PDF, TXT or MD)"
    )


def pages_to_text(pages: list[dict]) -> str:
    return "\n\n".join(
        f"[page {p['page']}]\n{p['text']}"
        for p in pages
    )
