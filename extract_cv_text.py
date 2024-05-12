from docx import Document

def extract_text_from_doc(file_path):
    # Load the DOC file
    doc = Document(file_path)
    full_text = []

    # Extract text from each paragraph in the document
    for para in doc.paragraphs:
        full_text.append(para.text)

    # Join all text into a single string
    return '\n'.join(full_text)

if __name__ == "__main__":
    # Path to the DOC file
    file_path = 'Alan_Woulfe_CV.doc'

    # Extract text from the DOC file
    text_content = extract_text_from_doc(file_path)

    # Save the extracted text to a TXT file
    with open('cv_text.txt', 'w') as text_file:
        text_file.write(text_content)
