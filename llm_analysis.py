"""
LLM analysis module for PolarBrief AI.

This module defines the `llm_analysis` function, which:
    - Sends text chunks to a Groq LLM for legal argument analysis
    - Validates and parses JSON responses using Pydantic
    - Assigns scores based on LLM output and TF-IDF centrality
    - Returns structured analysis with metadata
"""

import os
import json
import re
from datetime import datetime
from typing import List, Annotated, Dict

import numpy as np
from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    Field,
    StringConstraints,
    ValidationError,
    TypeAdapter,
)
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

POLARBRIEF_VERSION = "polarbrief v1"


def llm_analysis(chunks: List[Dict], groq_api_key: str) -> List[Dict]:
    """
    Perform LLM-based legal argument analysis on a list of text chunks.

    This function:
        1. Sends batches of text chunks to a Groq-hosted LLM.
        2. Extracts and validates structured JSON responses.
        3. Calculates combined scores from LLM output and TF-IDF centrality.
        4. Returns structured results with metadata.

    Args:
        chunks (List[Dict]): List of text chunks to analyze.
        groq_api_key (str): Groq API key for LLM access.

    Returns:
        List[Dict]: Structured analysis results with metadata and scores.

    Raises:
        ValueError: If the Groq API key is missing.
    """
    if not groq_api_key:
        raise ValueError("GROQ API key is required for analysis.")

    os.environ["GROQ_API_KEY"] = groq_api_key

    llm = ChatGroq(
        model_name="llama3-8b-8192",
        temperature=0,
    )

    class ArgumentAnalysis(BaseModel):
        """Schema for validated LLM argument analysis output."""

        heading: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
        contains_argument: Annotated[str, StringConstraints(pattern="^(yes|no)$")]
        summary: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
        polarity: Annotated[str, StringConstraints(pattern="^(Pro|Con|N/A)$")]
        score: Annotated[int, Field(ge=0, le=100)]

    # Adapter for validating a list of ArgumentAnalysis
    ArgumentListAdapter = TypeAdapter(List[ArgumentAnalysis])

    # ===== Shared System Prompt =====
    SYSTEM_PROMPT = """
    You are a legal assistant AI. You must read the given legal paragraph(s) and produce ONLY a valid JSON array.
    IMPORTANT:
    - You may be given multiple paragraphs.
    - For each paragraph, produce a separate JSON object following the schema below.
    - Return the results as a JSON array in the same order as the paragraphs are given (Paragraph 1 → First element, Paragraph 2 → Second element, etc.).
    - Do NOT merge paragraphs into a single analysis.
    EXAMPLE TEXT:
    "case cv z document file page pageid unit state district court northern district texa amarillo divis allianc hippocrat medicin et al plaintiff v case cv z u 
    food drug administr et al defend amicu curia brief mississippi alabama alaska arkansa florida georgia idaho indiana iowa kansa kentucki louisiana montana nebraska 
    ohio oklahoma south carolina south dakota tennesse texa utah wyom support plaintiff motion preliminari injunct case cv z document file page pageid tabl content page tabl author
    ii introduct interest amici curia summari argument background argument public interest equiti support injunct relief fda action mifepriston public interest equiti weigh strongli 
    fda action action defi feder law b fda action undermin public interest determin state feder agenc entitl make c fda action harm public interest undermin state abil protect citizen 
    forc state divert scarc resourc investig prosecut violat law conclus case cv z document file page pageid tabl author page case dobb v jackson woman health organ ct gonzal v oregon"
    EXAMPLE OUTPUT:
    [
    {
        "heading": "State Authority to Regulate Abortion Post-Dobbs",
        "contains_argument": "yes",
        "summary": "In Dobbs v. Jackson (2022), the Supreme Court returned abortion regulation to the states, allowing them to balance interests in unborn life, women’s health, and medical 
                    integrity. Twenty states, as amici, have enacted varying abortion laws—some restrictive, some permissive—with protections for a woman’s life and other exceptions.",
        "polarity": "Pro",
        "score": 85
    }
    ]
    TASKS:
    1. Write a concise summary of the paragraph.
    - Maximum 75 words.
    - Do NOT add or infer facts not explicitly stated.
    - Write in clear legal English.
    2. Create an appropriate heading for the paragraph.
    3. Indicate if it contains a legal argument: "yes" or "no".
    4. Classify polarity only as:
    - "Pro" = Supports plaintiff
    - "Con" = Opposes plaintiff
    - "N/A" = Neutral, lists, citations, or procedural content
    - Only use one of: "Pro", "Con", or "N/A"
    5. Assign a score (0–100) based on:
    - 0–20 = Irrelevant or no legal reasoning
    - 21–50 = Weak logic, lacks citations or clarity
    - 51–80 = Reasonable argument with support, but limited
    - 81–100 = Strong, well-structured, clearly supported legal argumen
    6. Do not hallucinate any facts .
    7. Ensure your summary exactly reflects the original paragraph.
    Output ONLY a JSON array starting with "[" and ending with "]".
    Do NOT output multiple standalone JSON objects.
    Each paragraph must be evaluated in isolation, without influence from others.
    JSON FORMAT (strict key order, quotes required):
    [
    {
        "heading": "string",
        "contains_argument": "yes/no",
        "summary": "string",
        "polarity": "Pro/Con/N/A",
        "score": int
    }
    ]
    OUTPUT_JSON:
    """

    # ===== JSON Extraction & Validation =====
    def extract_and_validate_json(text: str) -> List[ArgumentAnalysis] | None:
        """Extract the first JSON array from text and validate with Pydantic."""
        match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
        if not match:
            return None

        raw_json = match.group(0)
        raw_json = raw_json.replace("“", '"').replace("”", '"')
        raw_json = raw_json.replace("‘", "'").replace("’", "'")
        raw_json = re.sub(r",\s*([\]}])", r"\1", raw_json)

        try:
            parsed = json.loads(raw_json)
            return ArgumentListAdapter.validate_python(parsed)
        except (json.JSONDecodeError, ValidationError):
            return None

    # ===== Batch Processing =====
    def get_batch_argument_analysis(texts: List[str]) -> List[ArgumentAnalysis]:
        """Send a batch of paragraphs to the LLM and return validated analysis."""
        numbered_paragraphs = "\n".join(
            [f"{i+1}. {t}" for i, t in enumerate(texts)]
        )
        expected_count = len(texts)

        strict_prompt = f"""
        You will read the following {expected_count} paragraphs and produce exactly {expected_count} JSON objects.
        Do not merge or skip any paragraph. Output only the JSON array.
        PARAGRAPHS:
        {numbered_paragraphs}
        """

        try:
            response = llm(
                [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=strict_prompt),
                ]
            )
            parsed_list = extract_and_validate_json(response.content)
            if parsed_list and len(parsed_list) == expected_count:
                return parsed_list

            print(
                f"⚠ LLM returned {len(parsed_list) if parsed_list else 0} "
                f"items instead of {expected_count}"
            )
            return [
                ArgumentAnalysis(
                    heading="[No heading extracted]",
                    contains_argument="no",
                    summary="[No summary extracted]",
                    polarity="N/A",
                    score=0,
                )
                for _ in texts
            ]
        except Exception as e:
            print(f"Batch LLM request error: {e}")
            return [
                ArgumentAnalysis(
                    heading="[No heading extracted]",
                    contains_argument="no",
                    summary="[No summary extracted]",
                    polarity="N/A",
                    score=0,
                )
                for _ in texts
            ]

    # ===== Dynamic batching =====
    def dynamic_batches(data: List, batch_size: int = 5):
        """Yield batches of up to `batch_size` from a list."""
        for i in range(0, len(data), batch_size):
            yield data[i : i + batch_size]

    # ===== Normalization =====
    def normalize(arr: List[float]):
        """Normalize a numeric array to the range [0, 1]."""
        arr = np.array(arr)
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    # ===== Main Pipeline =====
    final_output = []
    texts = []
    llm_scores = []

    for batch_chunks in dynamic_batches(chunks, batch_size=5):
        batch_texts = [
            c.get("text", "").strip() or "[Empty text]" for c in batch_chunks
        ]
        batch_results = get_batch_argument_analysis(batch_texts)

        for chunk, parsed in zip(batch_chunks, batch_results):
            llm_scores.append(parsed.score)
            texts.append(chunk.get("text", "[Empty text]"))
            final_output.append(
                {
                    "page": chunk.get("page"),
                    "citation_start": chunk.get("citation_start"),
                    "citation_end": chunk.get("citation_end"),
                    "text": chunk.get("text", "[Empty text]"),
                    "heading": parsed.heading,
                    "contains_argument": parsed.contains_argument,
                    "summary": parsed.summary,
                    "polarity": parsed.polarity,
                    "source": POLARBRIEF_VERSION,
                    "timestamp": datetime.now().isoformat(),
                }
            )

    # ===== TF-IDF Centrality Scoring =====
    if any(t.strip() for t in texts):
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(texts)
        centrality_scores = cosine_similarity(tfidf_matrix, tfidf_matrix).mean(
            axis=1
        )
    else:
        centrality_scores = np.zeros(len(texts))

    # ===== Combine Scores =====
    llm_scores_norm = normalize(llm_scores)
    centrality_scores_norm = normalize(centrality_scores)
    combined_scores = (
        0.6 * llm_scores_norm + 0.4 * centrality_scores_norm
    )

    for i, item in enumerate(final_output):
        item["final_score"] = round(combined_scores[i] * 100, 2)

    return final_output


