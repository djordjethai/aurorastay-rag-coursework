"""AuroraStay RAG functions shared by the notebook's Flask deployment."""

from __future__ import annotations

import math
import re
from collections import Counter

from langchain_openai import ChatOpenAI


POLICY_ANSWER_RULES = """Use only the supplied AuroraStay policy excerpts, never external assumptions.
Answer every part of the question, including relevant property, rate-plan, time, fee,
exception, and approval conditions. Distinguish service animals from pets and incidental
holds from final charges. If the excerpts lack a required fact, say exactly what cannot
be established; do not invent a property-specific price or promise a refund. If sources
conflict, explain the conflict and recommend staff verification. Keep the answer concise.
End with the source document names used, not unrelated retrieved documents."""


def retrieve(query: str, k: int = 5, vectorstore=None) -> list[str]:
    """Combine FAISS rank with lexical evidence and cover each named property."""
    if vectorstore is None:
        raise ValueError("A prebuilt vectorstore is required")
    if not query.strip() or not 1 <= k <= 10:
        raise ValueError("Provide a nonblank query and k between 1 and 10")
    ranked = vectorstore.similarity_search_with_score(query, k=vectorstore.index.ntotal)
    docs = [doc for doc, _ in ranked]
    # Small policy-vocabulary normalization handles common guest paraphrases.
    aliases = {'dog': 'animal', 'dogs': 'animal', 'smoke': 'smoking', 'smokes': 'smoking', 'vape': 'vaping',
               'vapes': 'vaping', 'loud': 'noise', 'noisy': 'noise',
               'disturbance': 'noise', 'disturbances': 'noise'}
    words = lambda text: {aliases.get(term, term) for term in re.findall(r'[a-z0-9]+', text.lower())}
    doc_words = [words(doc.page_content) for doc in docs]
    city_codes = {}
    for doc in docs:
        match = re.search(r'\b(ASH-[A-Z]{3})[^|\n]*?,\s*([A-Za-z]+)\s+[A-Z]{2}\b', doc.page_content)
        if match:
            city_codes[match.group(2).lower()] = match.group(1)
    wanted = list(dict.fromkeys(re.findall(r'\bASH-[A-Z]{3}\b', query.upper())))
    for city, code in city_codes.items():
        if re.search(rf'\b{re.escape(city)}\b', query, re.IGNORECASE) and code not in wanted:
            wanted.append(code)
    stop = {'a', 'an', 'and', 'are', 'at', 'aurorastay', 'be', 'been', 'booked', 'can', 'desk', 'do', 'for', 'from', 'front', 'guest', 'guests', 'has', 'have', 'hotel', 'i', 'in', 'is', 'it', 'me', 'my', 'next', 'of', 'on', 'or', 'property', 'room', 'since', 'stay', 'supposed', 'the', 'there', 'to', 'want', 'what', 'when', 'where', 'who', 'will', 'with'}
    terms = (words(query) - stop) | {code.split('-')[1].lower() for code in wanted}
    if 'noise' in terms:
        terms.update({'quiet', 'complaint'})
    counts = Counter(term for entry in doc_words for term in entry)
    idf = {term: math.log((len(docs) + 1) / (counts[term] + 1)) + 1 for term in terms}
    scores = [sum(idf[term] for term in terms & entry) + 1 / (rank + 1)
              for rank, entry in enumerate(doc_words)]
    order = sorted(range(len(docs)), key=lambda i: (-scores[i], i))
    selected = []
    for code in wanted[:k]:
        match = next((i for i in order if code in docs[i].page_content and
                      docs[i].metadata.get('content_type') == 'table_row'), None)
        if match is not None and match not in selected:
            selected.append(match)
    # Treat k as a maximum; do not pad a focused answer with weak passages.
    floor = scores[order[0]] * (0.45 if wanted and ' and ' in query.lower() else 0.65)
    for i in order:
        if scores[i] < floor:
            break
        row_codes = set(re.findall(r'\bASH-[A-Z]{3}\b', docs[i].page_content))
        if wanted and row_codes and not row_codes.intersection(wanted):
            continue  # Never substitute another property's charge row.
        if row_codes and any(code in docs[j].page_content for j in selected for code in row_codes):
            continue  # One property row per named hotel leaves room for conditions.
        if i not in selected:
            selected.append(i)
        if len(selected) == k:
            break
    return [
        f"Rank {rank} | Source: {docs[i].metadata.get('source')} | "
        f"Chunk: {docs[i].metadata.get('chunk_id')}\n{docs[i].page_content}"
        for rank, i in enumerate(selected[:k], 1)
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
