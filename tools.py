# tools.py

import json
from langchain.tools import tool

# Load the course data from the JSON file when the script starts
try:
    with open('ae_courses.json', 'r', encoding='utf-8') as f:
        COURSE_DATA = json.load(f)
except FileNotFoundError:
    print("Warning: 'ae_courses.json' not found. The prerequisite checker tool will not work.")
    COURSE_DATA = []

@tool
def course_prerequisite_checker(course_code: str) -> str:
    """
    Looks up the prerequisites for a given course code.
    Use this tool when a user asks for the prerequisites of a specific course.
    The input must be a valid course code, for example: 'AE201A'.
    """
    print(f"--- Calling course_prerequisite_checker tool with input: {course_code} ---")
    
    if not COURSE_DATA:
        return "Sorry, the course data is not available."

    # Search for the course in the loaded data
    for course in COURSE_DATA:
        if course.get('code') == course_code.strip().upper():
            # The description often contains prerequisite info
            description = course.get('description', 'No description available.')
            # A simple heuristic to find prerequisites in the description
            if 'prerequisite' in description.lower():
                return f"The prerequisites mentioned for {course_code} are: {description}"
            else:
                return f"No explicit prerequisites are listed in the description for {course_code}. It's best to check the UG Manual or with the instructor."
                
    return f"Sorry, I could not find a course with the code {course_code}."