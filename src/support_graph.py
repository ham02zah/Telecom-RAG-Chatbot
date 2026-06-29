from typing import TypedDict

from langgraph.graph import END, StateGraph

from src.llm_assistant import generate_answer
from src.rag_pipeline import RAGPipeline, RetrievedFAQ
from src.text_normalizer import normalize_text


class SupportState(TypedDict):
    query: str
    normalized_query: str
    matches: list[RetrievedFAQ]
    answer: str


_pipeline = RAGPipeline()


def normalize_node(state: SupportState) -> SupportState:
    return {**state, "normalized_query": normalize_text(state["query"])}


def retrieve_node(state: SupportState) -> SupportState:
    matches = _pipeline.retrieve(state["normalized_query"])
    return {**state, "matches": matches}


def generate_node(state: SupportState) -> SupportState:
    answer = generate_answer(state["query"], state["matches"])
    return {**state, "answer": answer}


def build_support_graph():
    graph = StateGraph(SupportState)
    graph.add_node("normalize", normalize_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.set_entry_point("normalize")
    graph.add_edge("normalize", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


_support_graph = build_support_graph()


def run_support_workflow(query: str) -> dict:
    return _support_graph.invoke(
        {
            "query": query,
            "normalized_query": "",
            "matches": [],
            "answer": "",
        }
    )