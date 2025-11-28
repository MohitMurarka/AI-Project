"""
Document processing module for PolarBrief AI.

This module defines the DocumentProcessor class, which handles:
    - PDF text extraction
    - Text chunking into paragraphs
    - Text preprocessing
    - LLM-based legal argument analysis
    - Sorting and preparing balanced pro/con arguments

It returns structured analysis results including metadata.
"""

import os
import json
from typing import List, Dict
from datetime import datetime

import pytesseract
from dotenv import load_dotenv

from pdf_extraction import pdf_extraction
from chunking import chunk_lines
from processing import processing
from llm_analysis import llm_analysis
from output_files import output_files


class DocumentProcessor:
    """
    Main class for processing legal documents in PolarBrief AI.

    This class encapsulates the complete pipeline for:
        - Extracting text from PDF
        - Chunking text into logical sections
        - Preprocessing the text for analysis
        - Analyzing arguments using an LLM
        - Returning structured and sorted results
    """

    def __init__(self, poppler_path: str = None, tesseract_path: str = None):
        """
        Initialize the DocumentProcessor.

        Args:
            poppler_path (str, optional): Path to Poppler executables for PDF processing.
            tesseract_path (str, optional): Path to Tesseract OCR executable.
        """
        load_dotenv()
        self.POLARBRIEF_VERSION = "PolarBrief v1.0"

        if poppler_path:
            self.poppler_path = poppler_path
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

    def process_document(
        self,
        pdf_path: str,
        poppler_path: str,
        tesseract_path: str,
        groq_api_key: str
    ) -> Dict:
        """
        Process the given PDF document and return analysis results.

        Steps:
            1. Extract text lines from the PDF.
            2. Chunk lines into paragraphs.
            3. Preprocess text chunks.
            4. Perform LLM-based analysis.
            5. Sort results by final score.
            6. Select top 5 pro and con arguments.

        Args:
            pdf_path (str): Path to the PDF file.
            poppler_path (str): Path to Poppler executables.
            tesseract_path (str): Path to Tesseract OCR executable.
            groq_api_key (str): API key for the LLM service.

        Returns:
            Dict: Dictionary containing full analysis, top arguments,
                  timestamp, and version info. Includes error details if failed.
        """
        try:
            # Step 1: Extract text from PDF
            extracted_lines = pdf_extraction(pdf_path, poppler_path, tesseract_path)

            # Step 2: Chunk lines into paragraphs
            chunks = chunk_lines(extracted_lines)
            print("chunked")

            # Step 3: Preprocess text
            processed_chunks = processing(chunks)
            print("processed")

            # Step 4: Analyze with LLM
            final_output = llm_analysis(processed_chunks, groq_api_key)
            print("analysed")

            # Step 5: Sort and prepare outputs
            final_output_sorted = sorted(
                final_output,
                key=lambda x: x["final_score"],
                reverse=True
            )

            # Step 6: Build top/balanced lists
            final_pro = [
                item for item in final_output_sorted
                if item["polarity"].lower() == "pro"
            ][:5]
            final_con = [
                item for item in final_output_sorted
                if item["polarity"].lower() == "con"
            ][:5]

            balanced_args = final_con + final_pro

            return {
                "full_analysis": final_output,
                "balanced_arguments": balanced_args,
                "timestamp": datetime.now().isoformat(),
                "version": self.POLARBRIEF_VERSION
            }

        except Exception as e:
            print(f"Processing error: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "version": self.POLARBRIEF_VERSION
            }
