import pdfplumber
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data/raw"
PROCESSED = BASE / "data/processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

def pdf_to_text(pdf_path: Path, out_path: Path):
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
            else:
                text_parts.append("")
    full_text = "\n\n".join(text_parts)
    out_path.write_text(full_text, encoding="utf-8")
    print(f"Wrote {out_path} ({len(full_text)} chars)")

def main():
    for pdf in RAW.glob("*.pdf"):
        out = PROCESSED / (pdf.stem + ".txt")
        print("Processing", pdf)
        pdf_to_text(pdf, out)

if __name__ == "__main__":
    main()
