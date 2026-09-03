# PDF form filling

Keep form authoring separate from document-content parsing. If you need the
document's text, use `read_file`; never add a PDF text extractor or OCR fallback
inside this skill.

## Fillable AcroForms

1. Detect fields:

   ```bash
   python scripts/check_fillable_fields.py input.pdf
   python scripts/extract_form_field_info.py input.pdf field_info.json
   ```

2. Build `field_values.json` using only field IDs reported by
   `extract_form_field_info.py`:

   ```json
   [
     {
       "field_id": "last_name",
       "description": "The user's last name",
       "page": 1,
       "value": "Simpson"
     }
   ]
   ```

3. Fill and verify:

   ```bash
   python scripts/fill_fillable_fields.py input.pdf field_values.json output.pdf
   python scripts/convert_pdf_to_images.py output.pdf verify-images
   ```

Field metadata inspection is part of editing the AcroForm structure; it must
not be used as an alternate path for answering questions about document
content.

## Flat forms

For a PDF without AcroForm fields, render pages and visually locate the blank
areas to annotate. Do not extract text, tables, or layout with another parser.

1. Render the pages:

   ```bash
   python scripts/convert_pdf_to_images.py input.pdf images
   ```

2. Inspect each page image and record the intended entry rectangles in pixel
   coordinates. Crop/zoom with ImageMagick when needed.

3. Create `fields.json` with the render dimensions and coordinates:

   ```json
   {
     "pages": [
       {"page_number": 1, "image_width": 1700, "image_height": 2200}
     ],
     "form_fields": [
       {
         "page_number": 1,
         "description": "Last name entry field",
         "entry_bounding_box": [255, 175, 720, 218],
         "entry_text": {"text": "Smith", "font_size": 10}
       }
     ]
   }
   ```

4. Validate, fill, render again, and visually inspect the result:

   ```bash
   python scripts/check_bounding_boxes.py fields.json
   python scripts/fill_pdf_form_with_annotations.py input.pdf fields.json output.pdf
   python scripts/convert_pdf_to_images.py output.pdf verify-images
   ```

Correct any overlap, clipping, or misplaced annotation before delivering the
file.
