# AuroraStay policy Q&A proof of concept

Private coursework deployment bundle generated from the executed full-code notebook.
It contains only the Flask API, Streamlit UI, pinned dependencies, Dockerfiles, and
the prebuilt FAISS index. The source policy PDFs and course materials are not here.
The index contains excerpts from proprietary policy documents: keep this repository
private and do not redistribute its contents.

The API loads the embedding model and FAISS index once at startup and exposes
`POST /v1/relevant_chunks`, `POST /v1/answer_with_relevant_chunks`, and
`POST /v1/answer`. The Streamlit UI displays an answer draft and expandable
source passages for staff verification. It does **not** compute live DeepEval
scores or automatically approve guest-facing answers.

V4 contains a 118-vector index
that excludes navigation/administrative fragments and duplicate policy rows.
Retrieval reranks FAISS candidates by policy terms, covers each named property,
and treats `k` as the maximum number of passages rather than padding with weak
matches. The notebook's unchanged four offline metrics meet their mean gates
on development, reused supplied-test, and newly locked questions. That small
test does not establish production reliability or remove staff review.

## Codespaces run

Add `OPENAI_API_KEY` as a GitHub Codespaces secret before starting the Codespace.
`OPENAI_API_BASE` is optional for a compatible custom endpoint. Never commit keys.
In the Codespace terminal, run the two containers with host networking. In this
Codespace, Docker's user-defined bridge DNS could not resolve `api.openai.com`,
while host networking could; no application code change was needed.

```bash
docker build -t aurorastay-api ./backend
docker build -t aurorastay-ui ./frontend
docker run -d --name aurorastay-api --network host \
  -e OPENAI_API_KEY -e OPENAI_API_BASE aurorastay-api
docker run -d --name aurorastay-ui --network host \
  -e BACKEND_URL=http://127.0.0.1:7860 aurorastay-ui
```

Keep forwarded ports private. Use the authenticated Codespaces browser or
port-forwarding to inspect the UI and API. Example API payload:

```json
{"query":"What is the baseline smoking or vaping remediation charge?","k":5,"model_name":"gpt-4o-mini","temperature":0,"top_p":1,"max_tokens":512}
```

V4 was rebuilt in the private Codespace at commit `8f5df00`. The Flask health
endpoint reported 118 vectors and the Streamlit page returned HTTP 200. Through
private forwarded ports, the smoking-charge retrieval returned two focused
passages and the full RAG endpoint returned a source-backed answer. A separate
Atlanta/Miami query returned both named property rows. V4 also passed local
Flask tests. These checks do not establish general answer reliability or prove
an end-to-end browser interaction; staff should review the retrieved evidence
before using a draft with a guest.
