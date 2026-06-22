import os
import streamlit as st
from dotenv import load_dotenv

from rag import ask

load_dotenv()

st.set_page_config(page_title="Technical Support Bot", page_icon="🛠️")

st.title("🛠️ Technical Support Bot")
st.caption(
    "Ask a question about the Upwork API."
)

if not os.getenv("LLM_API_KEY"):
    st.warning(
        "LLM_API_KEY is not set. Copy .env.example to .env and fill in your "
        "key before asking questions."
    )

query = st.text_input(
    "Your question",
    placeholder="e.g. How long is an OAuth access token valid for?",
)

ask_clicked = st.button("Ask", type="primary")

if ask_clicked and query.strip():
    with st.spinner("Retrieving relevant docs and asking the model..."):
        try:
            result = ask(query)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            result = None

    if result:
        st.subheader("Answer")
        st.write(result["answer"])

        st.subheader("Latency")
        st.write(f"{result['latency_seconds']:.2f} seconds")

        st.subheader("Sources")
        if result["sources"]:
            for i,s in enumerate(result["sources"]):
                with st.expander(f"Source {i+1} (distance: {s['distance']:.4f})"):
                    st.code(s["text"],language="text")
        else:
            st.info("No supporting documentation was retrieved.")

elif ask_clicked:
    st.info("Please type a question first.")