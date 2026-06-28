import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src.config import OLLAMA_MODEL, OLLAMA_BASE_URL, USE_OLLAMA


def build_context_from_results(results):
    """Convert retrieved FAQ chunks into LLM context."""
    context_parts = []

    for index, result in enumerate(results, start=1):
        context_parts.append(
            f"Source {index}\n"
            f"Category: {result['category']}\n"
            f"Question: {result['question']}\n"
            f"Answer: {result['answer']}\n"
            f"Similarity Score: {result['score']:.4f}"
        )

    return "\n\n".join(context_parts)


def fallback_answer(query, results):
    """Use best FAQ answer directly when Ollama is unavailable."""
    if not results:
        return {
            "answer": "Sorry, I could not find relevant support information for your question. Please contact customer support.",
            "source": "Fallback - no match",
        }

    best = results[0]

    return {
        "answer": best["answer"],
        "source": f"{best['category']} | {best['question']}",
    }


def generate_support_answer(query, results):
    """
    Generate grounded support answer.

    Local:
    - uses Ollama through LangChain

    Streamlit Cloud / no Ollama:
    - returns best retrieved FAQ answer
    """

    if not USE_OLLAMA:
        return fallback_answer(query, results)

    context = build_context_from_results(results)

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful telecom and JazzCash customer support assistant.

Rules:
- Answer only using the provided FAQ context.
- If the answer is not in the context, say you do not know and ask the user to contact support.
- Keep the answer short, clear, and customer-friendly.
- If the user asked in Roman Urdu style, you may reply in simple Roman Urdu or English.
- Do not invent package prices, codes, or policies.

User Question:
{query}

FAQ Context:
{context}

Write a helpful support answer.
"""
    )

    try:
        llm = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.2,
        )

        chain = prompt | llm

        response = chain.invoke(
            {
                "query": query,
                "context": context,
            }
        )

        return {
            "answer": response.content,
            "source": f"Ollama model: {OLLAMA_MODEL}",
        }

    except Exception:
        return fallback_answer(query, results)