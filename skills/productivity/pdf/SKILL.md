---
name: pdf
description: "Create, merge, split, fill, and secure PDF files."
version: 1.0.0
author: Anthropic (adapted by Nous Research)
license: Proprietary. LICENSE.txt has complete terms
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [PDF, Documents, Forms, Office, Productivity]
    category: productivity
    related_skills: [nano-pdf, docx, xlsx]
---

# PDF Skill

Create, combine, split, transform, and secure PDF files — merging, page manipulation, form filling, watermarks, and encryption. PDF content reading is owned by the runtime document pipeline and exposed through `read_file`; this skill must not run a second extraction or OCR path.

## When to Use

Use this skill for PDF authoring and transformation: combining, splitting, rotating, watermarking, creating, filling forms, and encrypting/decrypting. Use `read_file` for document content; do not use pdfplumber, pdftotext, pytesseract, or another parser/OCR implementation.

## Prerequisites

```bash
pip install pypdf reportlab
which pdftoppm || sudo apt install -y poppler-utils   # visual verification
which qpdf || sudo apt install -y qpdf                 # CLI merge/split/decrypt
```

macOS: `brew install poppler qpdf`.

> Script paths below are relative to this skill's directory. Form filling has its own workflow — read [forms.md](forms.md) and follow it.

## Quick Reference

| Task | Best Tool | Command/Code |
|------|-----------|--------------|
| Merge PDFs | pypdf | `writer.add_page(page)` per page |
| Split PDFs | pypdf | One page per file |
| Read content | `read_file` | Runtime canonical Markdown |
| Create PDFs | reportlab | Canvas or Platypus |
| Command-line merge/split | qpdf | `qpdf --empty --pages ...` |
| Fill PDF forms | see [forms.md](forms.md) | `scripts/fill_fillable_fields.py` etc. |
| Edit existing text | `nano-pdf` skill | `nano-pdf edit file.pdf <page> "<instruction>"` |

## Common operations

### Merge / split / rotate (pypdf)

```python
from pypdf import PdfReader, PdfWriter

# Merge
writer = PdfWriter()
for pdf_file in ["doc1.pdf", "doc2.pdf"]:
    for page in PdfReader(pdf_file).pages:
        writer.add_page(page)
with open("merged.pdf", "wb") as f:
    writer.write(f)

# Split: one file per page
reader = PdfReader("input.pdf")
for i, page in enumerate(reader.pages):
    w = PdfWriter(); w.add_page(page)
    with open(f"page_{i+1}.pdf", "wb") as f:
        w.write(f)

# Rotate
page = reader.pages[0]
page.rotate(90)  # clockwise
```

### Create PDFs (reportlab)

```python
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet

doc = SimpleDocTemplate("report.pdf", pagesize=letter)
styles = getSampleStyleSheet()
story = [Paragraph("Report Title", styles["Title"]), Spacer(1, 12),
         Paragraph("Body text...", styles["Normal"]), PageBreak(),
         Paragraph("Page 2", styles["Heading1"])]
doc.build(story)
```

**Subscripts/superscripts:** never use Unicode sub/superscript characters (₀₁₂, ⁰¹²) — the built-in fonts lack the glyphs and render solid black boxes. Use `<sub>`/`<super>` markup inside `Paragraph` objects: `Paragraph("H<sub>2</sub>O", styles['Normal'])`. For canvas-drawn text, adjust font size and position manually.

### Command-line tools

```bash
qpdf --empty --pages file1.pdf file2.pdf -- merged.pdf     # merge
qpdf input.pdf --pages . 1-5 -- pages1-5.pdf               # split range
qpdf input.pdf output.pdf --rotate=+90:1                   # rotate page 1
qpdf --password=pw --decrypt encrypted.pdf decrypted.pdf   # remove password
pdfimages -j input.pdf img                                 # extract images
```

### Watermark

```python
from pypdf import PdfReader, PdfWriter

watermark = PdfReader("watermark.pdf").pages[0]
reader, writer = PdfReader("document.pdf"), PdfWriter()
for page in reader.pages:
    page.merge_page(watermark)
    writer.add_page(page)
with open("watermarked.pdf", "wb") as f:
    writer.write(f)
```

### Password protection

```python
writer.encrypt("userpassword", "ownerpassword")
```

## Form filling

Read [forms.md](forms.md) first — it distinguishes fillable (AcroForm) PDFs from flat scanned forms and walks through the helper scripts:

- `scripts/check_fillable_fields.py` — does the PDF have AcroForm fields?
- `scripts/extract_form_field_info.py` — enumerate AcroForm fields for editing
- `scripts/fill_fillable_fields.py` — fill AcroForm fields
- `scripts/fill_pdf_form_with_annotations.py` — overlay text on flat forms
- `scripts/check_bounding_boxes.py`, `scripts/create_validation_image.py` — verify placement visually

## Pitfalls

- pypdf preserves encryption flags: reading an encrypted PDF requires `PdfReader(path, password=...)` before pages are accessible.
- reportlab coordinates are bottom-left origin, points (1/72″) — not top-left.
- When filling flat forms by annotation overlay, always render a validation image and check the placement before delivering.

## Verification

1. Open the output with `PdfReader` and assert the expected page count.
2. Render the output and verify the intended content visually.
3. For anything visual (watermarks, filled forms, created reports): `pdftoppm -jpeg -r 100 output.pdf page` and inspect the images with `vision_analyze`.

## Related skills

`nano-pdf` (NL text edits in place), `docx` (Word), `xlsx` (spreadsheets), `powerpoint` (decks).
