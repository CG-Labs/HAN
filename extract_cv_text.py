import re

def parse_cv_sections(cv_text):
    # Define the section headers
    sections = {
        'PERSONAL_PROFILE': 'PERSONAL PROFILE',
        'KEY_SKILLS': 'KEY SKILLS',
        'PROFESSIONAL_EXPERIENCE': 'PROFESSIONAL EXPERIENCE',
        'EDUCATION': 'EDUCATION'
    }

    # Dictionary to hold the content of each section
    cv_data = {section: '' for section in sections}

    # Current section being processed
    current_section = None

    # Split the CV text into lines
    lines = cv_text.split('\n')

    # Iterate over each line in the CV text
    for line in lines:
        # Check if the line is a section header
        for section, header in sections.items():
            if header in line:
                current_section = section
                break

        # If the line is part of a section, add it to the section content
        if current_section:
            cv_data[current_section] += line + '\n'

    return cv_data

def extract_text_from_txt(file_path):
    # Read the plain text file
    with open(file_path, 'r') as file:
        cv_text = file.read()

    # Parse the CV sections
    cv_data = parse_cv_sections(cv_text)

    return cv_data

if __name__ == "__main__":
    # Path to the TXT file
    file_path = 'Alan_Woulfe_CV.txt'

    # Extract text from the TXT file
    cv_data = extract_text_from_txt(file_path)

    # For demonstration, print the extracted data
    for section, content in cv_data.items():
        print(f"Section: {section}")
        print(content)
        print("======================================")
