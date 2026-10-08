"""Local Flask API for the AuroraStay staff-facing RAG proof of concept."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, request
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from rag_utils import rag, retrieve

EMBED_MODEL = "HarishMaths/Hotel-Policy-Embedding"
INDEX_DIR = Path(__file__).resolve().parent / "faiss_index"
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL, model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
# Only load the index created by this notebook; FAISS pickle must not be untrusted.
vectorstore = FAISS.load_local(str(INDEX_DIR), embeddings,
                               allow_dangerous_deserialization=True)
rag_api = Flask("AuroraStay staff policy API")


def parameters() -> tuple[str, int, str, float, float, int]:
    data = request.get_json(silent=True) or {}
    query = str(data.get("query", "")).strip()
    k = int(data.get("k", 5))
    model_name = str(data.get("model_name", "gpt-4o-mini"))
    temperature = float(data.get("temperature", 0.0))
    top_p = float(data.get("top_p", 1.0))
    max_tokens = int(data.get("max_tokens", 512))
    if not query or not 1 <= k <= 10 or not 0 <= temperature <= 1 or not 0 < top_p <= 1 or not 64 <= max_tokens <= 2048:
        raise ValueError("Invalid query or generation parameters")
    return query, k, model_name, temperature, top_p, max_tokens


@rag_api.get("/")
def home():
    return jsonify({"service": "AuroraStay RAG API", "vectors": vectorstore.index.ntotal})


@rag_api.post("/v1/relevant_chunks")
def relevant_chunks():
    try:
        query, k, *_ = parameters()
        return jsonify({"query": query, "k": k,
                        "relevant_chunks": retrieve(query, k, vectorstore)})
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@rag_api.post("/v1/answer_with_relevant_chunks")
def answer_with_relevant_chunks():
    try:
        query, k, model_name, temperature, top_p, max_tokens = parameters()
        answer, chunks = rag(query, k, model_name, temperature, top_p,
                             max_tokens, vectorstore)
        return jsonify({"query": query, "k": k, "answer": answer,
                        "relevant_chunks": chunks, "model_name": model_name,
                        "temperature": temperature, "top_p": top_p,
                        "max_tokens": max_tokens})
    except (TypeError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400


@rag_api.post("/v1/answer")
def answer_only():
    response = answer_with_relevant_chunks()
    if isinstance(response, tuple):
        return response
    return jsonify({"answer": response.get_json()["answer"]})


if __name__ == "__main__":
    rag_api.run(host="127.0.0.1", port=7860)
