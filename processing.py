"""
Processing module for PolarBrief AI.

This module provides functions to:
    - Download required NLTK datasets.
    - Clean OCR artifacts from extracted text.
    - Preprocess text by tokenizing, removing stopwords, and lemmatizing/stemming.
"""

import re
from typing import List, Dict

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize


def downloads_nltk() -> None:
    """
    Download required NLTK datasets for tokenization, lemmatization, and stopwords.
    """
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("stopwords")
    nltk.download("wordnet")


def processing(chunks: List[Dict]) -> List[Dict]:
    """
    Preprocess extracted PDF text chunks.

    Steps:
        1. Download required NLTK datasets.
        2. Remove OCR artifacts and noise.
        3. Convert to lowercase.
        4. Remove non-alphabetic characters.
        5. Tokenize.
        6. Remove stopwords.
        7. Lemmatize and stem tokens.

    Args:
        chunks (List[Dict]): List of dictionaries containing "text" keys.

    Returns:
        List[Dict]: Updated chunks with preprocessed "text".
    """
    downloads_nltk()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()

    def clean_ocr_artifacts(text: str) -> str:
        """
        Remove OCR-related artifacts such as repeated dots, hyphens, and long junk strings.
        """
        if not text:
            return ""
        text = re.sub(r"[\.\-]{3,}", " ", text)
        text = re.sub(r"[a-zA-Z0-9]{1,}[a-zA-Z0-9\s]{0,}$", "", text)
        text = re.sub(r"([a-zA-Z0-9]){15,}", "", text)
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text

    def preprocess_text(text: str) -> str:
        """
        Perform full preprocessing: clean, lowercase, tokenize, remove stopwords,
        lemmatize, and stem.
        """
        text = clean_ocr_artifacts(text)
        text = text.lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [
            stemmer.stem(lemmatizer.lemmatize(word))
            for word in tokens
        ]
        return " ".join(tokens)

    for entry in chunks:
        if "text" in entry:
            entry["text"] = preprocess_text(entry["text"])

    return chunks
