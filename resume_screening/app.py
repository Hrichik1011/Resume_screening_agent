import streamlit as st
import os
import google.generativeai as genai
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load API Key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Custom CSS for visual enhancements
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 0.3em;
        }
        .subtitle {
            text-align: center;
            font-size: 1.2em;
            color: #777;
        }
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px 24px;
            border-radius: 8px;
        }
        .result-box {
            background-color: #f0f2f6;
            border-left: 5px solid #4CAF50;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            font-family: 'Courier New', monospace;
        }
    </style>
""", unsafe_allow_html=True)

# UI Title
st.markdown('<div class="title">🤖 Gemini Resume Screener</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Match resumes with job descriptions using Google Gemini AI</div>', unsafe_allow_html=True)
st.markdown("---")

# Upload section
st.header("📥 Upload Files")

col1, col2 = st.columns(2)
with col1:
    jd_file = st.file_uploader("📄 Job Description (TXT format)", type=["txt"])
with col2:
    resume_files = st.file_uploader("📁 Resumes (PDF format)", type=["pdf"], accept_multiple_files=True)

# Helper: Extract PDF text
def extract_text_from_pdf(file):
    text = ""
    reader = PdfReader(file)
    for page in reader.pages:
        text += page.extract_text()
    return text

# Run the agent
st.markdown("---")
if st.button("🔍 Run Resume Screening"):
    if not jd_file or not resume_files:
        st.error("Please upload both a job description and at least one resume.")
    else:
        with st.spinner("🔎 Analyzing resumes and comparing with job description..."):
            # Read job description
            job_description = jd_file.read().decode()

            # Process resumes
            resumes = []
            for resume in resume_files:
                text = extract_text_from_pdf(resume)
                resumes.append({"name": resume.name, "text": text})

            # Split and embed
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = [splitter.create_documents([r["text"]])[0] for r in resumes]
            embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectordb = Chroma.from_documents(docs, embedding=embedding)

            # Vector search
            results = vectordb.similarity_search(job_description, k=3)
            top_resumes = [doc.page_content for doc in results]

            # LLM prompt to rank
            model = genai.GenerativeModel("gemini-2.5-flash")
            prompt = f"""
You are an AI hiring assistant.

Job Description:
{job_description}

Top 3 matched resumes:
1.
{top_resumes[0]}

2.
{top_resumes[1]}

3.
{top_resumes[2]}

Please rank them from most to least suitable, and explain why.
            """
            response = model.generate_content(prompt)
            ranking_output = response.text

        st.success("✅ Screening complete! Top candidates ranked below.")
        st.markdown('<div class="result-box">' + ranking_output.replace('\n', '<br>') + '</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("<center><small>Made with ❤️ using Streamlit + Gemini</small></center>", unsafe_allow_html=True)
