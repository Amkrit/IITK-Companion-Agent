# parse_ae_courses.py

import requests
import fitz  # PyMuPDF
import json
import re
import os

# URL for the Aerospace Engineering course PDF
URL = "https://www.iitk.ac.in/doaa/data/template/AE-template.pdf"
PDF_PATH = "AE-template.pdf"

def download_pdf():
    """Downloads the PDF file if it doesn't exist."""
    if os.path.exists(PDF_PATH):
        print(f"'{PDF_PATH}' already exists. Skipping download.")
        return True
    
    print(f"Downloading PDF from {URL}...")
    try:
        response = requests.get(URL)
        response.raise_for_status()
        with open(PDF_PATH, 'wb') as f:
            f.write(response.content)
        print("PDF downloaded successfully.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the PDF: {e}")
        return False

def parse_aerospace_pdf():
    """Parses the downloaded Aerospace Engineering PDF to extract course data."""
    if not download_pdf():
        return

    print(f"Opening and parsing '{PDF_PATH}'...")
    all_courses = []
    
    try:
        doc = fitz.open(PDF_PATH)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return

    # Regex to find course entries.
    pattern = re.compile(r'(?s)(AE\d{3}[A-Z]?):\s*(.*?)\(([\d\-]+)\)(.*?)(?=AE\d{3}[A-Z]?:\s|$)')

    for match in pattern.finditer(full_text):
        course_info = {}
        code = match.group(1).strip()
        title = match.group(2).strip().replace('\n', ' ')
        credits = match.group(3).strip()
        description = match.group(4).strip().replace('\n', ' ')
        description = re.sub(r'\s+', ' ', description)

        course_info['code'] = code
        course_info['title'] = title
        course_info['credits'] = credits
        course_info['description'] = description
        all_courses.append(course_info)

    print(f"Successfully parsed {len(all_courses)} Aerospace courses.")

    # Save the data to a JSON file
    output_filename = 'ae_courses.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_courses, f, indent=4, ensure_ascii=False)
        
    print(f"All Aerospace course data has been saved to '{output_filename}'.")


if __name__ == "__main__":
    parse_aerospace_pdf()