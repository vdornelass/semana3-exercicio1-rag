from langgraph.graph import END, START, StateGraph

from src.graph.nodes.format_output import format_output
from src.graph.nodes.generate import generate
from src.graph.nodes.retrieve import retrieve
from src.graph.state import State


def build_graph():
    graph = StateGraph(State)

    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("format_output", format_output)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "format_output")
    graph.add_edge("format_output", END)
    
    return graph.compile()