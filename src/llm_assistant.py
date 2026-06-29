import os

from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL, USE_OLLAMA
from src.rag_pipeline import RetrievedFAQ


def _format_context(matches: list[RetrievedFAQ]) -> str:
    chunks = []
    for item in matches:
        chunks.append(
            f"Category: {item.category}\n"
            f"Question: {item.question}\n"
            f"Answer: {item.answer}"
        )
    return "\n\n".join(chunks)


def _fallback_answer(matches: list[RetrievedFAQ]) -> str:
    if not matches:
        return "Sorry, I could not find a matching FAQ. Please contact Jazz support at 111."
    best = matches[0]
    return (
        f"**{best.question}**\n\n{best.answer}\n\n"
        f"_Category: {best.category} | Confidence: {best.score:.2f}_"
    )


def generate_answer(query: str, matches: list[RetrievedFAQ]) -> str:
    if not USE_OLLAMA:
        return _fallback_answer(matches)

    try:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError:
        return _fallback_answer(matches)

    context = _format_context(matches)
    system_prompt = (
        "You are a helpful Jazz Telecom customer support assistant. "
        "Answer using only the provided FAQ context. "
        "Be concise, friendly, and accurate. "
        "If the context does not contain the answer, say you are unsure and suggest contacting Jazz support."
    )
    user_prompt = f"Customer question:\n{query}\n\nFAQ context:\n{context}"

    try:
        llm = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            temperature=0.2,
        )
        response = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        return response.content.strip()
    except Exception:
        return _fallback_answer(matches)