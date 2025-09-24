"""
Takes a code snippet (file or pasted) and explains it in plain English using LangGraph.

Usage:
    export OPENAI_API_KEY="sk-..."
    python code_explainer.py path/to/code.py
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# ✅ LangGraph + LangChain imports
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, MessagesState, START, END

# Load environment variables
load_dotenv()

# ---- Prompt ----
PROMPT = """You are an expert senior developer and teacher.
Explain the following code in clear, simple terms for a beginner.
Highlight:
- what the code does
- main functions/classes and responsibilities
- potential gotchas
- a one-sentence summary

Code:
{code}
"""


# ---- Build Graph ----
def build_graph():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_template(PROMPT)

    # simple runnable: prompt → llm
    chain = prompt | llm

    graph = StateGraph(MessagesState)
    graph.add_node("explain_code", chain)
    graph.add_edge(START, "explain_code")
    graph.add_edge("explain_code", END)

    return graph.compile()


# ---- Runner ----
def explain_code(code: str) -> str:
    graph = build_graph()
    result = graph.invoke({"code": code})

    # result is a list of AI messages → get last content
    return result[-1].content if result else "No explanation generated."


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python code_explainer.py path/to/code.py")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print("File not found:", path)
        sys.exit(1)

    code_text = path.read_text()
    explanation = explain_code(code_text)

    print("\n--- Explanation ---\n")
    print(explanation)
