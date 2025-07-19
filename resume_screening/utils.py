import os
from PyPDF2 import PdfReader

folder_path = "D:\\resume_screening\\resumes"

def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text

def load_resumes(folder_path):
    resumes = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(os.path.join(folder_path, filename))
            resumes.append({"name": filename, "text": text})
    return resumes
