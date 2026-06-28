from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from src.rag_pipeline import retrieve_documents
from src.llm_assistant import generate_support_answer
from src.text_normalizer import normalize_text


class SupportState(TypedDict):
    user_query: str
    normalized_query: str
    retrieved_docs: List[Dict[str, Any]]
    final_answer: str
    answer_source: str


def normalize_query_node(state: SupportState):
    state["normalized_query"] = normalize_text(state["user_query"])
    return state


def retrieve_node(state: SupportState):
    state["retrieved_docs"] = retrieve_documents(state["normalized_query"])
    return state


def generate_answer_node(state: SupportState):
    response = generate_support_answer(
        query=state["user_query"],
        results=state["retrieved_docs"],
    )

    state["final_answer"] = response["answer"]
    state["answer_source"] = response["source"]

    return state


def build_support_graph():
    workflow = StateGraph(SupportState)

    workflow.add_node("normalize_query", normalize_query_node)
    workflow.add_node("retrieve_docs", retrieve_node)
    workflow.add_node("generate_answer", generate_answer_node)

    workflow.set_entry_point("normalize_query")

    workflow.add_edge("normalize_query", "retrieve_docs")
    workflow.add_edge("retrieve_docs", "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()


def run_support_workflow(user_query):
    graph = build_support_graph()

    initial_state = {
        "user_query": user_query,
        "normalized_query": "",
        "retrieved_docs": [],
        "final_answer": "",
        "answer_source": "",
    }

    return graph.invoke(initial_state)