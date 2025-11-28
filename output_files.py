"""
Output file generator for PolarBrief AI.

This module:
    - Exports extracted and analyzed arguments to JSON and PDF.
    - Creates summary PDFs for top arguments (Pro, Con, Neutral).
    - Bundles all outputs into a downloadable ZIP.
"""

import json
import unicodedata
import io
import zipfile
import os
from typing import List, Dict
from fpdf import FPDF


def clean_text(text: str) -> str:
    """
    Normalize and strip unsupported characters for PDF export.

    Args:
        text (str): Input text.

    Returns:
        str: Cleaned ASCII-only text.
    """
    if not isinstance(text, str):
        text = str(text)
    return unicodedata.normalize("NFKD", text).encode(
        "ascii", "ignore"
    ).decode("ascii")


class PDF(FPDF):
    """Custom PDF class for formatting argument data."""

    def header(self):
        """Define PDF header styling."""
        self.set_font("Helvetica", "B", 14)
        self.ln(5)

    def chapter_body(self, entry: Dict, selected_fields: List[str]):
        """
        Add a section of key-value data to the PDF.

        Args:
            entry (Dict): Data entry.
            selected_fields (List[str]): Fields to include in output.
        """
        self.set_font("Helvetica", "", 11)
        for field in selected_fields:
            label = clean_text(field.replace("_", " ").title())
            value = clean_text(entry.get(field, ""))
            self.multi_cell(0, 8, f"{label}: {value}")
            self.ln(1)
        self.ln(3)
        self.cell(0, 0, "-" * 80)
        self.ln(5)


def output_files(final_output: List[Dict]) -> io.BytesIO:
    """
    Generate JSON, PDF, and ZIP outputs from the final analysis results.

    Args:
        final_output (List[Dict]): Processed and scored argument data.

    Returns:
        io.BytesIO: In-memory ZIP file containing all outputs.
    """

    def save_json(filename: str, data):
        """Save data as a JSON file."""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # Save all arguments
    save_json("all_arguments.json", final_output)

    # Sorted lists
    sorted_output = sorted(final_output, key=lambda x: x["final_score"], reverse=True)
    top_neutral = [x for x in sorted_output if x["polarity"].lower() == "n/a"][:5]
    top_pro = [x for x in sorted_output if x["polarity"].lower() == "pro"][:5]
    top_con = [x for x in sorted_output if x["polarity"].lower() == "con"][:5]
    balanced = top_con + top_pro

    save_json("top_5_neutral.json", top_neutral)
    save_json("balanced_arguments.json", balanced)

    # Minimal index
    minimal = [
        {
            "page": i.get("page"),
            "citation_start": i.get("citation_start"),
            "citation_end": i.get("citation_end"),
            "heading": i.get("heading", "")
        }
        for i in final_output
    ]
    save_json("Index.json", minimal)

    # PDF export helper
    def pdfexport(json_file: str, fields: List[str], pdf_file: str):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        pdf = PDF()
        pdf.add_page()
        for item in data:
            pdf.chapter_body(item, fields)
        pdf.output(pdf_file)

    # Export PDFs
    pdfexport(
        "Index.json",
        ["page", "citation_start", "citation_end", "heading"],
        "Index.pdf"
    )
    pdfexport(
        "all_arguments.json",
        [
            "page", "citation_start", "citation_end", "heading", "summary",
            "polarity", "source", "timestamp"
        ],
        "all_arguments.pdf"
    )
    pdfexport(
        "top_5_neutral.json",
        [
            "page", "citation_start", "citation_end", "heading", "summary",
            "polarity", "source", "timestamp"
        ],
        "top_5_neutral.pdf"
    )

    # Balanced PDF
    def export_balanced_pdf():
        pdf = PDF()
        pdf.add_page()

        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "PRO ARGUMENTS (Top 5)", ln=1, align="C")
        pdf.set_font("Helvetica", "", 12)
        for arg in top_pro:
            pdf.multi_cell(0, 8, clean_text(
                f"Page: {arg['page']}\nScore: {arg['final_score']}\n"
                f"Citation_Start: {arg['citation_start']}\n"
                f"Citation_End: {arg['citation_end']}\n"
                f"Heading: {arg['heading']}\nSummary: {arg['summary']}\n"
            ))
            pdf.ln(2)

        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "CON ARGUMENTS (Top 5)", ln=1, align="C")
        pdf.set_font("Helvetica", "", 12)
        for arg in top_con:
            pdf.multi_cell(0, 8, clean_text(
                f"Page: {arg['page']}\nScore: {arg['final_score']}\n"
                f"Citation_Start: {arg['citation_start']}\n"
                f"Citation_End: {arg['citation_end']}\n"
                f"Heading: {arg['heading']}\nSummary: {arg['summary']}\n"
            ))
            pdf.ln(2)

        pdf.output("balanced_arguments.pdf")

    export_balanced_pdf()

    # Zip bundle
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for f in [
            "all_arguments.json", "top_5_neutral.json", "balanced_arguments.json",
            "Index.json", "Index.pdf", "all_arguments.pdf", "top_5_neutral.pdf",
            "balanced_arguments.pdf"
        ]:
            if os.path.exists(f):
                zf.write(f)

    zip_buffer.seek(0)
    return zip_buffer
