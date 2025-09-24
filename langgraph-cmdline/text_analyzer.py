"""
Text Analyzer using LangGraph and OpenAI Chat API.

This script reads text from a file, sends it to an LLM for analysis,
and returns the number of characters, words, paragraphs, and sentences
in JSON format.
"""

from pathlib import Path
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# Load text from file
txt_file = Path("sample.txt")  # replace with your file path
text = txt_file.read_text()

# Prompt template for counting
PROMPT = """
You are an expert text analyzer. Given the text below, return:
1) Number of characters include everything
2) Number of words
3) Number of paragraphs
4) Number of sentences

Text:
{text}

Provide the output in JSON format with keys: characters, words, paragraphs, sentences
"""

prompt = PromptTemplate(input_variables=["text"], template=PROMPT)

# Initialize LLM
llm = ChatOpenAI(temperature=0)

# --- Define Graph State ---
class State(dict):
    text: str
    result: str

# --- Define Nodes ---
def analyze_text(state: State):
    formatted_prompt = prompt.format(text=state["text"])
    response = llm.invoke(formatted_prompt)
    return {"result": response.content}

# --- Build Graph ---
workflow = StateGraph(State)

# Add node
workflow.add_node("analyze_text", analyze_text)

# Entry point
workflow.set_entry_point("analyze_text")

# End
workflow.add_edge("analyze_text", END)

# Compile graph
app = workflow.compile()

# Run graph
final_state = app.invoke({"text": text})
print(final_state["result"])
