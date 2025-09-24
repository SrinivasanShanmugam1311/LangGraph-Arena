"""
Resume to JSON Extractor using LangGraph + GPT-4o-mini
Supports: PDF input or raw text input
Usage:
    export OPENAI_API_KEY="sk-..."
    python resume_to_json_langgraph.py --pdf resume.pdf --output resume_output.json
"""

import os
import json
import argparse
import traceback
from pathlib import Path
import PyPDF2
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")


# -------------------------
# Graph State
# -------------------------
class State(dict):
    pdf: str
    text: str
    llm_output: str
    json_output: dict


# -------------------------
# Node 1: Extract text from PDF
# -------------------------
def extract_pdf_text_node(state: State):
    if state.get("pdf"):
        pdf_path = Path(state["pdf"])
        if not pdf_path.exists():
            raise FileNotFoundError(f"{pdf_path} not found")

        text = ""
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return {"text": text}
    return {}


# -------------------------
# Node 2: Resume text → JSON (via LLM)
# -------------------------
def resume_to_json_node(state: State):
    if not state.get("text"):
        raise ValueError("No text found for processing")

    prompt_template = """
    Extract the following information from this resume text and return it strictly as valid JSON:
    {{
      "Name": "",
      "Email": "",
      "Phone": "",
      "Education": "",
      "Experience": [],
      "Skills": []
    }}

    Resume Text:
    {text}

    IMPORTANT:
    - Return ONLY valid JSON.
    - Do not include ```json or any extra text.
    """
    chat_prompt = ChatPromptTemplate.from_template(prompt_template)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

    pipeline = chat_prompt | llm
    output_message = pipeline.invoke({"text": state["text"]})

    return {"llm_output": output_message.content.strip()}


# -------------------------
# Node 3: Parse JSON safely
# -------------------------
def parse_json_node(state: State):
    raw = state["llm_output"]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        cleaned = raw.strip().replace("```json", "").replace("```", "")
        data = json.loads(cleaned)
    return {"json_output": data}


# -------------------------
# Build LangGraph workflow
# -------------------------
workflow = StateGraph(State)

workflow.add_node("extract_pdf_text", extract_pdf_text_node)
workflow.add_node("resume_to_json", resume_to_json_node)
workflow.add_node("parse_json", parse_json_node)

workflow.set_entry_point("extract_pdf_text")
workflow.add_edge("extract_pdf_text", "resume_to_json")
workflow.add_edge("resume_to_json", "parse_json")
workflow.add_edge("parse_json", END)

app = workflow.compile()


# -------------------------
# Main function with CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Extract resume info to JSON using GPT-4o-mini + LangGraph")
    parser.add_argument("--pdf", type=str, help="Path to the PDF resume")
    parser.add_argument("--text", type=str, help="Direct resume text")
    parser.add_argument("--output", type=str, default="resume_output.json", help="Output JSON file name")
    args = parser.parse_args()

    try:
        # Build initial state
        init_state = {}
        if args.pdf:
            init_state["pdf"] = args.pdf
        if args.text:
            init_state["text"] = args.text

        # Run graph
        final_state = app.invoke(init_state)

        # Pretty print JSON
        pretty_json = json.dumps(final_state["json_output"], indent=4)
        print(pretty_json)

        # Save JSON
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(pretty_json)
        print(f"\n✅ Saved JSON to {args.output}")

    except Exception as e:
        print("Error:", e)
        traceback.print_exc()


if __name__ == "__main__":
    main()
