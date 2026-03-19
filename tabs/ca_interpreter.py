# tabs/ca_interpreter.py
# This file builds the UI for Tab 1: CA Interpreter
# It collects user input, calls the RAG engine, and displays the result

import streamlit as st
import re
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
        options=list(CA_FILES.keys()), # ["OSSTF (Secondary Teachers)", "ETFO...", "ETFOOT..."]
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
       #api_key = st.secrets["ANTHROPIC_API_KEY"]
        api_key = st.secrets["GEMINI_API_KEY"]

        # Show a spinner while we wait for the API call
        with st.spinner("Searching the collective agreement and generating answer..."):
            result = get_ca_answer(
                union_name=union_name,
                question=question,
                api_key=api_key
            )

        # --- Display the answer ---
        # Get current timestamp to show when answer was generated
        from datetime import datetime
        import pytz
        est =pytz.timezone("America/Toronto")
        timestamp = datetime.now().strftime("%I:%M %p %Z")

        st.success("**Answer**") 

        # Styled container makes it visually clear this is the answer area
        # and that it has been refreshed with  a new response
        with st.container(border=True):
           #st.success("Answer generated")
            st.markdown(f"*Generated at {timestamp}*")
            st.divider()
            st.markdown(result["answer"]) # Claude's plain-language response


        # --- Source citations ---
        # Show the raw CA chunks that Claude used to generate the answer
        # Collapsed by default so they don't overwhelm the UI
        with st.expander("📎 Source excerpts from the Collective Agreement"):
            # Extract cited section references from Gemini's answer
            # We look for patterns matching all CA numbering formats
            cited_refs = re.findall(
                r'\b(C\d+\.\d+|L\d+\.\d+|L-[A-Z]\.\d+\.\d+|\d+\.\d+\.\d+|LETTER OF AGREEMENT #\d+|LETTER OF UNDERSTANDING)',
                result["answer"],
                re.IGNORECASE
            )
            cited_refs = [ref.upper() for ref in cited_refs]

            # Filter excerpts to only those containing cited references
            filtered_docs = []
            for doc in result["sources"]:
                content_upper = doc.page_content.upper()
                if any(ref in content_upper for ref in cited_refs):
                    if len(doc.page_content.strip()) >= 150:
                        filtered_docs.append(doc)

            # Fall back to all sources if filter removes everything
            display_docs = filtered_docs if filtered_docs else result["sources"]

            st.caption(f"Showing {len(display_docs)} excerpts directly referenced in the answer.")

            for i, doc in enumerate(display_docs):
                page_num = doc.metadata.get("page", "?")
                section = doc.metadata.get("section", "")
                st.markdown(f"**Excerpt {i+1}** — {section} — Page {int(page_num) + 1}")
                st.caption(doc.page_content)
                st.divider()