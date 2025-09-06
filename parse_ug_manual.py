# parse_ug_manual.py

import fitz  # PyMuPDF
import os
import re

# Define the input and output file paths
PDF_PATH = "ug_manual.pdf"
OUTPUT_PATH = "ug_manual.txt"

def parse_ug_manual():
    """
    Parses the UG Manual PDF to extract and clean its text content.
    """
    if not os.path.exists(PDF_PATH):
        print(f"Error: '{PDF_PATH}' not found. Please make sure it's in the same folder.")
        return

    print(f"Opening and parsing '{PDF_PATH}'...")
    
    try:
        doc = fitz.open(PDF_PATH)
        full_text = ""
        for page_num, page in enumerate(doc):
            # Extract text from each page
            text = page.get_text()
            full_text += text + "\n"
        
        print(f"Successfully extracted text from all {doc.page_count} pages.")

    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return

    # --- Basic Cleaning ---
    # Replace multiple spaces with a single space
    cleaned_text = re.sub(r' +', ' ', full_text)
    # Replace multiple newlines with a single newline
    cleaned_text = re.sub(r'\n+', '\n', cleaned_text)

    # Save the cleaned text to a .txt file
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
        
    print(f"Cleaned text from the UG Manual has been saved to '{OUTPUT_PATH}'.")


if __name__ == "__main__":
    parse_ug_manual()