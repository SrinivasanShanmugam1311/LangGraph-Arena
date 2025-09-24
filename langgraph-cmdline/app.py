from typing import TypedDict
from langgraph.graph import StateGraph, END
import matplotlib.pyplot as plt

# Define state
class State(TypedDict):
    operation: str
    a: float
    b: float
    result: float

# Node functions
def add_node(state: State):
    return {"result": state["a"] + state["b"]}

def sub_node(state: State):
    return {"result": state["a"] - state["b"]}

def mul_node(state: State):
    return {"result": state["a"] * state["b"]}

def div_node(state: State):
    if state["b"] == 0:
        return {"result": float("inf")}  # handle divide by zero
    return {"result": state["a"] / state["b"]}

# Router function
def router(state: State):
    op = state["operation"].lower()
    if op == "add":
        return "add_label"
    elif op == "sub":
        return "sub_label"
    elif op == "mul":
        return "mul_label"
    elif op == "div":
        return "div_label"
    else:
        raise ValueError(f"Unknown operation: {op}")

# Build graph
workflow = StateGraph(State)

workflow.add_node("router", lambda state: state)  
# Add nodes
workflow.add_node("add_node_fun", add_node)
workflow.add_node("sub_node_fun", sub_node)
workflow.add_node("mul_node_fun", mul_node)
workflow.add_node("div_node_fun", div_node)

# Set entry point (router)
workflow.set_entry_point("router")

# Add conditional edges
workflow.add_conditional_edges(
    "router",
    router,
    {
        "add_label": "add_node_fun",
        "sub_label": "sub_node_fun",
        "mul_label": "mul_node_fun",
        "div_label": "div_node_fun",
    }
)

# Connect math nodes to END
workflow.add_edge("add_node_fun", END)
workflow.add_edge("sub_node_fun", END)
workflow.add_edge("mul_node_fun", END)
workflow.add_edge("div_node_fun", END)

# Compile app
app = workflow.compile()

# ✅ Draw graph
print(app.invoke({"operation": "add", "a": 10, "b": 5})["result"])   # 15
print(app.invoke({"operation": "sub", "a": 10, "b": 5})["result"])   # 5
print(app.invoke({"operation": "mul", "a": 10, "b": 5})["result"])   # 50
print(app.invoke({"operation": "div", "a": 10, "b": 5})["result"])   # 2.0
