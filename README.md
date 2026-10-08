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

## Codespaces run

Add `OPENAI_API_KEY` as a GitHub Codespaces secret before starting the Codespace.
`OPENAI_API_BASE` is optional for a compatible custom endpoint. Never commit keys.
In the Codespace terminal, run:

```bash
docker network create aurorastay-net
docker build -t aurorastay-api ./backend
docker build -t aurorastay-ui ./frontend
docker run -d --name aurorastay-api --network aurorastay-net \
  -p 7860:7860 -e OPENAI_API_KEY -e OPENAI_API_BASE aurorastay-api
docker run -d --name aurorastay-ui --network aurorastay-net \
  -p 8501:8501 -e BACKEND_URL=http://aurorastay-api:7860 aurorastay-ui
```

Keep forwarded ports private. Use the authenticated Codespaces browser or
port-forwarding to inspect the UI and API. Example API payload:

```json
{"query":"What is the pet charge at ASH-CHI Loop?","k":3,"model_name":"gpt-4o-mini","temperature":0,"top_p":1,"max_tokens":512}
```

The Docker containers have not been built or tested merely by being present in
this repository; record their actual Codespaces test results separately.
