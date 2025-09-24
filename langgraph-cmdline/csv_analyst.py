"""
CSV Analyst using LangGraph.
Load a CSV into Pandas, then run natural language queries using OpenAI + Pandas agent.
"""

import sys
import pandas as pd
from dotenv import load_dotenv

# ✅ LangChain + LangGraph imports
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langgraph.prebuilt import create_agent_executor

# Load environment variables
load_dotenv()


def run_agent(csv_path: str):
    # ✅ Load CSV
    df = pd.read_csv(csv_path)
    print(f"CSV loaded successfully with shape {df.shape} and columns {list(df.columns)}")

    # ✅ Create LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # ✅ Create Pandas agent
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=False,
        allow_dangerous_code=True,   # allow execution of df operations
    )

    # ✅ Wrap agent with LangGraph executor
    graph_agent = create_agent_executor(agent, checkpointer=None)

    print("Ask questions about the data (type 'exit' to quit).")

    while True:
        q = input("\nQuestion: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        try:
            # LangGraph requires structured input
            res = graph_agent.invoke({"input": q})
            print("\nAnswer:\n", res["output"])
        except Exception as e:
            print("Agent error:", e)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python csv_analyst.py data.csv")
        sys.exit(1)

    run_agent(sys.argv[1])
