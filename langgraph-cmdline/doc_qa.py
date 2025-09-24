"""
Document Q&A using LangGraph + LangChain + FAISS + OpenAI
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langgraph.graph import StateGraph, END

# Load env
load_dotenv()


# -------------------------
# State
# -------------------------
class State(dict):
    pdf: str
    chunks: list
    db: FAISS
    query: str
    answer: str


# -------------------------
# Node 1: Load + split PDF
# -------------------------
def load_and_split_node(state: State):
    loader = PyPDFLoader(state["pdf"])
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    return {"chunks": chunks}


# -------------------------
# Node 2: Build FAISS vectorstore
# -------------------------
def build_vectorstore_node(state: State):
    emb = OpenAIEmbeddings()
    db = FAISS.from_documents(state["chunks"], emb)
    return {"db": db}


# -------------------------
# Node 3: Q&A on query
# -------------------------
def qa_node(state: State):
    retriever = state["db"].as_retriever(search_kwargs={"k": 4})
    llm = ChatOpenAI(temperature=0)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        verbose=False
    )

    ans = qa.invoke(state["query"])
    return {"answer": ans["result"]}


# -------------------------
# Build Graph
# -------------------------
workflow = StateGraph(State)

workflow.add_node("load_and_split", load_and_split_node)
workflow.add_node("build_vectorstore", build_vectorstore_node)
workflow.add_node("qa", qa_node)

workflow.set_entry_point("load_and_split")
workflow.add_edge("load_and_split", "build_vectorstore")
workflow.add_edge("build_vectorstore", "qa")
workflow.add_edge("qa", END)

app = workflow.compile()


# -------------------------
# CLI Main
# -------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python doc_qa_langgraph.py /path/to/doc.pdf")
        sys.exit(1)

    pdf = sys.argv[1]
    if not Path(pdf).exists():
        print("PDF not found:", pdf)
        sys.exit(1)

    print("Building vectorstore (this may take a minute)...")
    # Init graph up to build_vectorstore
    init_state = {"pdf": pdf}
    state = app.invoke({**init_state, "query": " "})  # dummy run to build DB

    print("Vectorstore ready. Document Q&A. Type 'exit' to quit.")

    while True:
        q = input("\nQuestion: ").strip()
        if q.lower() in ("exit", "quit"):
            break

        final_state = app.invoke({**init_state, "query": q})
        print("\nAnswer:\n", final_state["answer"])


if __name__ == "__main__":
    main()
