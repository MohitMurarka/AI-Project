"""
PolarBrief AI - Legal Argument Analyzer (Streamlit App)

This is the main UI entry point for the PolarBrief AI project.
It allows the user to upload a PDF legal document and performs
argument analysis using the backend DocumentProcessor class.

The app:
    - Loads environment variables (e.g., API keys, paths)
    - Accepts PDF uploads
    - Processes the document with OCR and NLP
    - Displays balanced pro/con arguments
    - Provides downloadable results as a ZIP file
"""

import os
import json
import unicodedata

import streamlit as st
from dotenv import load_dotenv
from fpdf import FPDF

from logic import DocumentProcessor
from output_files import output_files

# Load environment variables
load_dotenv()

# === UI Setup ===
st.set_page_config(page_title="Legal Argument Analyzer", layout="wide")

BACKGROUND_IMAGE_URL = (
    "https://i.pinimg.com/736x/64/eb/ef/64ebefbbd558d77f1a1e0d01a4e050c1.jpg"
)

# Inject CSS for styling the app
st.markdown(
    f"""
    <style>
        html, body, [class*="css"] {{
            font-family: 'Segoe UI', sans-serif;
            font-size: 18px;
            color: #222;
        }}
        .stApp {{
            background-image: url('{BACKGROUND_IMAGE_URL}');
            background-size: cover;
            background-attachment: fixed;
        }}
        .main-box {{
            background-color: green;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
            text-align: center;
        }}
        .title-box {{
            background-color: #004466;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
        }}
        .title-box h1 {{
            color: white;
            font-size: 2.8em;
            margin: 0;
        }}
        .main-box h2 {{
            color: black;
            font-size: 1.4em;
            margin: 0;
        }}
        .section-header {{
            background-color: #F08080;
            padding: 12px 18px;
            border-left: 6px solid #007acc;
            border-radius: 6px;
            font-size: 1.4em;
            font-weight: bold;
            margin: 20px 0 10px 0;
        }}
        .stButton > button {{
            width: 100%;
            background-color: #007acc;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 10px;
            font-size: 1.05em;
        }}
        .stDownloadButton > button {{
            width: 100%;
            background-color: #007acc;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 10px;
            font-size: 1.05em;
        }}
        .stDownloadButton > button:hover {{
            background-color: #005f99;
            cursor: pointer;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# === Title ===
st.markdown(
    '<div class="title-box"><h1>PolarBrief AI - Legal Argument Analyzer</h1></div>',
    unsafe_allow_html=True,
)

# === Backend Setup ===
poppler_path = os.getenv("POPPLER_PATH")
tesseract_path = os.getenv("TESSERACT_PATH")
groq_api_key = os.getenv("GROQ_API_KEY")

# === Upload PDF ===
uploaded_file = st.file_uploader("📄 Upload PDF Document", type=["pdf"])

if uploaded_file is not None:
    """
    This block handles the document upload and triggers processing.

    Steps:
        1. Save the uploaded PDF temporarily
        2. When 'Analyze Document' is clicked:
            - Check if GROQ API key is set
            - Process the document with DocumentProcessor
            - Display extracted arguments in collapsible sections
            - Offer download of results as a ZIP file
        3. Clean up temporary files after processing
    """
    temp_file = "temp_upload.pdf"
    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("Analyze Document"):
        if not groq_api_key:
            st.error("Please provide your GROQ API key.")
            st.stop()

        processor = DocumentProcessor()

        with st.spinner("🔍 Processing document..."):
            try:
                results = processor.process_document(
                    pdf_path=temp_file,
                    poppler_path=poppler_path,
                    tesseract_path=tesseract_path,
                    groq_api_key=groq_api_key,
                )

                final_output = results["full_analysis"]
                balanced_args = results["balanced_arguments"]

                zip_buffer = output_files(final_output)

                st.success("Analysis complete!")

                # === Display Top Arguments ===
                st.markdown(
                    '<div class="section-header">'
                    'Balanced Arguments (Top 5 PRO & Top 5 CON)'
                    '</div>',
                    unsafe_allow_html=True,
                )

                for i, arg in enumerate(balanced_args, 1):
                    with st.expander(
                        f"Argument # (Polarity: {arg['polarity']}) - {arg['heading']}"
                    ):
                        st.markdown(f"**Page:** {arg['page']}")
                        st.markdown(
                            f"**Citation_start:** {arg['citation_start']}"
                        )
                        st.markdown(
                            f"**Citation_end:** {arg['citation_end']}"
                        )
                        st.markdown("**Summary:**")
                        st.write(arg["summary"])

                # === Download Section ===
                st.markdown(
                    '<div class="section-header">⬇Download Bundle</div>',
                    unsafe_allow_html=True,
                )
                st.download_button(
                    label="Download All Results as ZIP",
                    data=zip_buffer,
                    file_name="legal_argument_bundle.zip",
                    mime="application/zip",
                )

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
