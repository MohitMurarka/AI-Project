"""
PDF extraction module for PolarBrief AI.

This module provides the `pdf_extraction` function, which:
    - Extracts text from a PDF using pdfplumber.
    - Falls back to OCR with Tesseract if direct extraction fails.
    - Cleans extracted text to remove noise and formatting artifacts.
"""

import re
from typing import List, Dict

import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import pdfplumber


def pdf_extraction(pdf_path: str, poppler_path: str , tesseract_path: str) -> List[Dict]:
   
    """
    Extract text from a PDF, falling back to OCR if necessary.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        List[Dict]: A list of dictionaries containing:
            - "text": The extracted line text.
            - "page_no": Page and line identifier.
            - "method": Extraction method ("pdfplumber" or "ocr").
    """

    # Configure Tesseract path for OCR (Windows)
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

    def clean_text(text: str) -> str:
        """Clean extracted text by removing bullets, symbols, and extra spaces."""
        text = re.sub(r"[•·●♦▪•∙]", "", text)
        text = re.sub(r"[^\w\s,.:;()\"'-]", "", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    def is_noisy(text: str, threshold: float = 0.6) -> bool:
        """Check if text contains a high proportion of non-alphanumeric characters."""
        if not text:
            return True
        non_alpha = sum(1 for c in text if not c.isalnum())
        return (non_alpha / len(text)) > threshold

    def has_repeated_characters(text: str, repeat_threshold: int = 4) -> bool:
        """Check if text has repeated characters beyond the threshold."""
        return bool(re.search(r"(.)\1{" + str(repeat_threshold) + ",}", text))

    def extract_text_with_fallback(pdf_path: str) -> List[Dict]:
        """Attempt PDF text extraction; fallback to OCR if no usable text found."""
        final_output = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, start=1):
                    print(
                        f"Page {page_num}/{total_pages}: "
                        f"Trying PDF text extraction..."
                    )
                    page_line_no = 1
                    text = page.extract_text()
                    lines = text.split("\n") if text else []

                    if lines and sum(len(l.strip()) for l in lines) > 20:
                        for line in lines:
                            cleaned = line.strip()
                            if cleaned:
                                final_output.append({
                                    "text": cleaned,
                                    "page_no": f"[p{page_num} {page_line_no}]",
                                    "method": "pdfplumber"
                                })
                                page_line_no += 1
                        continue

                    print(
                        f" Page {page_num} has no usable text — "
                        f"fallback to OCR"
                    )
                    images = convert_from_path(
                        pdf_path,
                        dpi=300,
                        first_page=page_num,
                        last_page=page_num
                    )
                    img = images[0]
                    gray = img.convert("L")
                    bw = gray.point(lambda x: 0 if x < 180 else 255, "1")
                    ocr_text = pytesseract.image_to_string(bw, lang="eng")
                    lines = ocr_text.strip().split("\n")
                    line_no = 1

                    for line in lines:
                        cleaned_line = clean_text(line)
                        if (
                            cleaned_line
                            and not is_noisy(cleaned_line)
                            and not has_repeated_characters(cleaned_line)
                        ):
                            final_output.append({
                                "text": cleaned_line,
                                "page_no": f"[p{page_num} {line_no}]",
                                "method": "ocr"
                            })
                            line_no += 1
        except Exception as e:
            print(f"[ERROR] Failed during PDF processing: {e}")

        return final_output

    output = extract_text_with_fallback(pdf_path)
    return output

