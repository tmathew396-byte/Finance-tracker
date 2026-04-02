import pytesseract
from PIL import Image
import re

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\tmath\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

def clean_text(text):
    text = text.replace("\n", " ").strip()
    return re.sub(r"\s+", " ", text)

def extract_invoice_fields(image_file):
    image = Image.open(image_file)

    # OCR
    raw_text = pytesseract.image_to_string(image)
    cleaned_text = clean_text(raw_text)

    fields = {
        "TOTAL": "",
        "DATE": "",
        "COMPANY": "",
        "INVOICE NO": ""
    }

    # ✅ Extract TOTAL
    total_regex = r"(Grand\s?Total|Total Amount|Amount Due|Total)\D*([\d,\.]+)"
    match = re.search(total_regex, cleaned_text, re.IGNORECASE)
    if match:
        fields["TOTAL"] = match.group(2)

    # ✅ Extract DATE
    date_regex = r"(\d{2}[/-]\d{2}[/-]\d{2,4})"
    date_match = re.search(date_regex, cleaned_text)
    if date_match:
        fields["DATE"] = date_match.group(1)

    # ✅ Extract Invoice Number
    invoice_regex = r"(Invoice\s?(No|Number|#)\s*[:\-]?\s*)([A-Za-z0-9\-\/]+)"
    inv_match = re.search(invoice_regex, cleaned_text, re.IGNORECASE)
    if inv_match:
        fields["INVOICE NO"] = inv_match.group(3)

    # ✅ Extract Company Name (first line)
    lines = raw_text.split("\n")
    lines = [l.strip() for l in lines if l.strip()]
    if lines:
        fields["COMPANY"] = lines[0]

    return fields, cleaned_text
