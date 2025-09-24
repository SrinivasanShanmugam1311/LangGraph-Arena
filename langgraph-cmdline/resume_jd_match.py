"""
Resume vs Job Description Matching using LangGraph + LangChain embeddings.

Supports: .txt, .pdf, .doc/.docx resumes
Usage:
    export OPENAI_API_KEY="sk-..."
    python resume_jd_match_langgraph.py resume.pdf job_description.txt
"""

import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import docx
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langgraph.graph import StateGraph, END

# Load env
# Load env from the script directory (not the current working dir)
SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=SCRIPT_DIR / ".env")

# Validate and capture the API key once
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY or not OPENAI_API_KEY.startswith(("sk-", "sk-proj-")):
    raise RuntimeError(
        "OPENAI_API_KEY missing/invalid. Put it in a .env next to this script "
        "or export it in your shell before running."
    )

# -------------------------
# State for LangGraph
# -------------------------
class State(dict):
    resume_file: str
    jd_file: str
    resume_text: str
    jd_text: str
    resume_text_clean: str
    jd_text_clean: str
    similarity_score: float
    matching_keywords: set
    resume_only: set
    jd_only: set


# -------------------------
# Node 1: Read Resume
# -------------------------
def read_resume_node(state: State):
    file_path = Path(state["resume_file"])
    if not file_path.exists():
        raise FileNotFoundError(f"{file_path} not found")

    if file_path.suffix.lower() == ".txt":
        text = file_path.read_text()
    elif file_path.suffix.lower() == ".pdf":
        text = ""
        reader = PdfReader(str(file_path))
        for page in reader.pages:
            text += page.extract_text() + " "
    elif file_path.suffix.lower() in [".doc", ".docx"]:
        doc = docx.Document(str(file_path))
        text = " ".join([para.text for para in doc.paragraphs])
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")

    return {"resume_text": text, "jd_text": Path(state["jd_file"]).read_text()}


# -------------------------
# Node 2: Preprocess text
# -------------------------
def preprocess_node(state: State):
    def clean(text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return " ".join(text.split())

    return {
        "resume_text_clean": clean(state["resume_text"]),
        "jd_text_clean": clean(state["jd_text"])
    }


# -------------------------
# Node 3: Compute similarity
# -------------------------
def similarity_node(state: State):
    #emb = OpenAIEmbeddings()
    emb = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    resume_vec = emb.embed_query(state["resume_text_clean"])
    jd_vec = emb.embed_query(state["jd_text_clean"])

    score = cosine_similarity([resume_vec], [jd_vec])[0][0]

    # keyword overlaps
    resume_words = set(state["resume_text_clean"].split())
    jd_words = set(state["jd_text_clean"].split())
    matching_keywords = resume_words.intersection(jd_words)

    return {
        "similarity_score": score,
        "matching_keywords": matching_keywords,
        "resume_only": resume_words - jd_words,
        "jd_only": jd_words - resume_words,
    }


# -------------------------
# Build LangGraph Workflow
# -------------------------
workflow = StateGraph(State)

workflow.add_node("read_resume", read_resume_node)
workflow.add_node("preprocess", preprocess_node)
workflow.add_node("similarity", similarity_node)

workflow.set_entry_point("read_resume")
workflow.add_edge("read_resume", "preprocess")
workflow.add_edge("preprocess", "similarity")
workflow.add_edge("similarity", END)

app = workflow.compile()


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python resume_jd_match_langgraph.py resume_file jd_file")
        sys.exit(1)

    resume_file = sys.argv[1]
    jd_file = sys.argv[2]

    # Run graph
    final_state = app.invoke({"resume_file": resume_file, "jd_file": jd_file})

    print(f"\nSemantic Match Score (0-1): {final_state['similarity_score']:.2f}")
    print(f"Matching keywords ({len(final_state['matching_keywords'])}): {final_state['matching_keywords']}")
    print(f"Resume-only keywords ({len(final_state['resume_only'])}): {final_state['resume_only']}")
    print(f"JD-only keywords ({len(final_state['jd_only'])}): {final_state['jd_only']}")
