# tabs/ca_interpreter.py
# This file builds the UI for Tab 1: CA Interpreter
# It collects user input, calls the RAG engine, and displays the result

import streamlit as st
from utils.rag_utils import get_ca_answer, CA_FILES

def render():
    """
    This function renders the entire Tab 1 UI.
    It gets called from app.py whenever the user is on this tab.
    """

    # --- Header ---
    st.header("📄 Collective Agreement Interpreter")
    st.write(
        "Select a union and ask a plain-language question. "
        "The answer will be grounded in the actual collective agreement text."
    )

    st.divider()

    # --- Union selector ---
    # CA_FILES is the dictionary we defined in rag_utils — its keys are the union names
    union_name = st.selectbox(
        label="Which collective agreement are you asking about?",
        options=list(CA_FILES.keys()),  # ["OSSTF (Secondary Teachers)", "ETFO...", "ETFOOT..."]
        index=0
    )

    # --- Question input ---
    question = st.text_area(
        label="Your question",
        placeholder="e.g. How many sick days is an OSSTF teacher entitled to per year?",
        height=100
    )

    # --- Submit button ---
    submitted = st.button("Get Answer", type="primary")

    # --- On submit ---
    if submitted:

        # Basic validation — don't proceed if question is empty
        if not question.strip():
            st.warning("Please enter a question before submitting.")
            return

        # Get API key from Streamlit secrets
        # This reads from .streamlit/secrets.toml — never hardcode the key here
        api_key = st.secrets["ANTHROPIC_API_KEY"]

        # Show a spinner while we wait for the API call
        with st.spinner("Searching the collective agreement and generating answer..."):
            result = get_ca_answer(
                union_name=union_name,
                question=question,
                api_key=api_key
            )

        # --- Display the answer ---
        st.success("Answer generated")
        st.markdown("### Answer")
        st.markdown(result["answer"])  # Claude's plain-language response

        st.divider()

        # --- Source citations ---
        # Show the raw CA chunks that Claude used to generate the answer
        # Collapsed by default so they don't overwhelm the UI
        with st.expander("📎 Source excerpts from the Collective Agreement"):
            for i, doc in enumerate(result["sources"]):
                page_num = doc.metadata.get("page", "?")
                st.markdown(f"**Excerpt {i+1} — Page {int(page_num) + 1}**")
                st.caption(doc.page_content)
                st.divider()
