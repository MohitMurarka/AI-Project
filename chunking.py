"""
Chunking module for PolarBrief AI.

This module provides the `chunk_lines` function, which:
    - Groups extracted PDF lines into chunks based on token limits.
    - Preserves citation start/end markers and page ranges.
"""

import re
from typing import List, Dict


def chunk_lines(output: List[Dict]) -> List[Dict]:
    """
    Chunk lines of extracted PDF text into token-limited sections.

    Args:
        output (List[Dict]): List of dictionaries containing:
            - "text": Extracted text line.
            - "page_no": Page and line identifier.

    Returns:
        List[Dict]: Chunked text with metadata:
            - "page": Page range.
            - "citation_start": First line of the chunk.
            - "citation_end": Last line of the chunk.
            - "text": Combined chunk text.
    """

    def count_tokens_simple(text: str) -> int:
        """Count tokens in text using simple whitespace splitting."""
        return len(text.split())

    def parse_page_line(page_str: str):
        """
        Parse a page/line identifier string.

        Args:
            page_str (str): Identifier like "[p12 3]".

        Returns:
            tuple[int | None, int | None]: Page number, line number.
        """
        match = re.match(r"\[p(\d+)\s+(\d+)\]", page_str)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None, None

    def format_page_range(start_page_str: str, end_page_str: str) -> str:
        """
        Format a range of pages/lines into a readable string.

        Args:
            start_page_str (str): Starting page/line identifier.
            end_page_str (str): Ending page/line identifier.

        Returns:
            str: Formatted page range string.
        """
        sp, sl = parse_page_line(start_page_str)
        ep, el = parse_page_line(end_page_str)
        if sp is None or ep is None:
            return f"{start_page_str}-{end_page_str}"
        if sp == ep:
            return f"[p{sp} {sl}-{el}]"
        return f"[p{sp} {sl}]-[p{ep} {el}]"

    def chunk_lines_simple_tokenizer(
        lines: List[Dict], max_tokens: int = 250
    ) -> List[Dict]:
        """
        Chunk lines into groups not exceeding `max_tokens`.

        Args:
            lines (List[Dict]): Extracted PDF lines with metadata.
            max_tokens (int, optional): Max token count per chunk. Defaults to 250.

        Returns:
            List[Dict]: Chunked text with citation metadata.
        """
        chunks = []
        current_lines = []
        current_tokens = 0
        start_page_info = None
        end_page_info = None

        for line in lines:
            text = line["text"].strip()
            tokens = count_tokens_simple(text)

            if current_tokens + tokens > max_tokens and current_lines:
                chunks.append({
                    "page": format_page_range(start_page_info, end_page_info),
                    "citation_start": current_lines[0]["text"].strip(),
                    "citation_end": current_lines[-1]["text"].strip(),
                    "text": "\n".join([l["text"] for l in current_lines])
                })
                current_lines = []
                current_tokens = 0
                start_page_info = None
                end_page_info = None

            if not current_lines and text:
                start_page_info = line["page_no"]

            end_page_info = line["page_no"]
            current_lines.append(line)
            current_tokens += tokens

        if current_lines:
            chunks.append({
                "page": format_page_range(start_page_info, end_page_info),
                "citation_start": current_lines[0]["text"].strip(),
                "citation_end": current_lines[-1]["text"].strip(),
                "text": "\n".join([l["text"] for l in current_lines])
            })

        return chunks

    return chunk_lines_simple_tokenizer(output, max_tokens=250)
