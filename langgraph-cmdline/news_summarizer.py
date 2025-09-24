"""
News Summarizer using LangGraph + GPT
Fetch articles (via newspaper3k), summarize each, and produce a short digest.

Usage:
    export OPENAI_API_KEY="sk-..."
    python news_summarizer_langgraph.py https://example.com/article1 https://example.com/article2
"""

import sys
from dotenv import load_dotenv
from newspaper import Article
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# Load env
load_dotenv()

# -------------------------
# LangGraph State
# -------------------------
class State(dict):
    url: str
    article_text: str
    summary: str


# -------------------------
# Node 1: Fetch article
# -------------------------
def fetch_article_node(state: State):
    url = state["url"]
    art = Article(url)
    art.download()
    art.parse()
    text = art.title + "\n\n" + art.text
    return {"article_text": text}


# -------------------------
# Node 2: Summarize article
# -------------------------
def summarize_article_node(state: State):
    SUMMARY_PROMPT = """You are a concise news summarizer.
    Given the article text, produce:
    1) Headline (single line)
    2) 2-3 sentence summary
    3) 1-sentence "why it matters"

    Article:
    {article}
    """

    chat_prompt = ChatPromptTemplate.from_template(SUMMARY_PROMPT)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    pipeline = chat_prompt | llm
    result = pipeline.invoke({"article": state["article_text"]})

    return {"summary": result.content.strip()}


# -------------------------
# Build Workflow
# -------------------------
workflow = StateGraph(State)

workflow.add_node("fetch_article", fetch_article_node)
workflow.add_node("summarize_article", summarize_article_node)

workflow.set_entry_point("fetch_article")
workflow.add_edge("fetch_article", "summarize_article")
workflow.add_edge("summarize_article", END)

app = workflow.compile()


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python news_summarizer_langgraph.py <article_url> [article_url...]")
        sys.exit(1)

    urls = sys.argv[1:]
    for url in urls:
        print(f"\nFetching: {url}")
        try:
            final_state = app.invoke({"url": url})
            print("\n--- Summary ---\n", final_state["summary"])
        except Exception as e:
            print("❌ Failed for", url, "Error:", e)
