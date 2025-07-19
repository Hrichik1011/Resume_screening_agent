import os
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from utils import load_resumes

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load job description
with open("jd_input.txt", "r") as f:
    job_description = f.read()

# Load resumes
resumes = load_resumes("resumes")
texts = [res["text"] for res in resumes]
names = [res["name"] for res in resumes]

# Vector store setup
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = [splitter.create_documents([text])[0] for text in texts]

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectordb = Chroma.from_documents(docs, embedding=embeddings)

# Search relevant resumes
results = vectordb.similarity_search(job_description, k=3)
top_resumes_texts = [doc.page_content for doc in results]

# Ranking prompt for Gemini
model = genai.GenerativeModel("gemini-2.5-flash")

prompt = f"""
You are a hiring assistant AI.

Job Description:
{job_description}

Below are the top 3 matched resumes:

{top_resumes_texts[0]}

{top_resumes_texts[1]}

{top_resumes_texts[2]}

Rank these 3 resumes in terms of suitability for the job and explain your reasoning.
"""

response = model.generate_content(prompt)
ranking_output = response.text

with open("output_summary.txt", "w") as f:
    f.write(ranking_output)

print("✅ Resume ranking complete. See 'output_summary.txt'")
