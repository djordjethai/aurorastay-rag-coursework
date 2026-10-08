"""Internal AuroraStay staff interface for policy Q&A."""
import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:7860").rstrip("/")
st.set_page_config(page_title="AuroraStay policy assistant", page_icon="🏨", layout="wide")
st.title("🏨 AuroraStay policy assistant")
st.caption("Internal staff proof of concept. Verify source passages before advising a guest.")

with st.sidebar:
    st.header("Answer settings")
    k = st.slider("Retrieved passages", 1, 10, 5)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    max_tokens = st.slider("Maximum answer tokens", 64, 2048, 512, 64)

response_type = st.radio("View", ["Answer + sources", "Answer only"], horizontal=True)
query = st.text_input("Hotel-policy question", placeholder="What is the pet charge at ASH-CHI Loop?")
if st.button("Ask", type="primary"):
    if not query.strip():
        st.warning("Enter a question.")
    else:
        endpoint = "/v1/answer_with_relevant_chunks" if response_type == "Answer + sources" else "/v1/answer"
        payload = {"query": query, "k": k, "temperature": temperature,
                   "top_p": 1.0, "max_tokens": max_tokens}
        try:
            with st.spinner("Checking hotel policies..."):
                response = requests.post(f"{BACKEND_URL}{endpoint}", json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            st.subheader("Draft answer")
            st.write(result["answer"])
            for index, chunk in enumerate(result.get("relevant_chunks", []), 1):
                with st.expander(f"Source passage {index}"):
                    st.text(chunk)
        except requests.RequestException as exc:
            st.error(f"The local policy API did not respond: {exc}")
