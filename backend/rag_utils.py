"""AuroraStay RAG functions shared by the notebook's Flask deployment."""

from __future__ import annotations

from langchain_openai import ChatOpenAI


POLICY_ANSWER_RULES = """Use only the supplied AuroraStay policy excerpts, never external assumptions.
Answer every part of the question, including relevant property, rate-plan, time, fee,
exception, and approval conditions. Distinguish service animals from pets and incidental
holds from final charges. If the excerpts lack a required fact, say exactly what cannot
be established; do not invent a property-specific price or promise a refund. If sources
conflict, explain the conflict and recommend staff verification. Keep the answer concise.
End with the source document names used, not unrelated retrieved documents."""


def retrieve(query: str, k: int = 5, vectorstore=None) -> list[str]:
    """Return ranked source-labelled hotel-policy excerpts."""
    if vectorstore is None:
        raise ValueError("A prebuilt vectorstore is required")
    if not query.strip() or not 1 <= k <= 10:
        raise ValueError("Provide a nonblank query and k between 1 and 10")
    results = vectorstore.similarity_search(query, k=k)
    return [
        f"Rank {rank} | Source: {doc.metadata.get('source')} | "
        f"Chunk: {doc.metadata.get('chunk_id')}\n{doc.page_content}"
        for rank, doc in enumerate(results, 1)
    ]


def generate(query: str, retrieved_chunks: list[str], model: ChatOpenAI) -> str:
    """Generate a staff-facing draft from retrieved policy excerpts only."""
    context = "\n\n".join(retrieved_chunks)
    prompt = (
        "You answer AuroraStay hotel-policy questions for staff.\n"
        f"Question: {query}\n\nRetrieved policy excerpts:\n{context}\n\n"
        f"Rules:\n{POLICY_ANSWER_RULES}\n\nAnswer:"
    )
    return str(model.invoke(prompt).content)


def rag(
    query: str, k: int = 5, model_name: str = "gpt-4o-mini",
    temperature: float = 0.0, top_p: float = 1.0,
    max_tokens: int = 512, vectorstore=None,
) -> tuple[str, list[str]]:
    """Reuse one FAISS store for retrieval and generate an evidence-backed draft."""
    model = ChatOpenAI(model=model_name, temperature=temperature,
                       top_p=top_p, max_tokens=max_tokens)
    chunks = retrieve(query=query, k=k, vectorstore=vectorstore)
    return generate(query, chunks, model), chunks
