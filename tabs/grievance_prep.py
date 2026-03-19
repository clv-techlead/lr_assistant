#
# tabs/grievance_prep.py
# This file builds the UI for Tab 2: Grievance Prep Assistant
# It takes grievance facts as input and returns a structured legal risk analysis
# grounded in the relevant collective agreement

import streamlit as st
from datetime import datetime
from utils.rag_utils import load_or_build_index, build_hybrid_retriever, CA_FILES, GEMINI_MODEL
import google.genai as genai


def get_grievance_analysis(union_name: str, facts: str,
                            stage: str, remedy: str, api_key: str) -> str:
    """
    Takes grievance details, retrieves relevant CA clauses,
    and returns a structured analysis from Gemini.
    """

    # Reuse the same FAISS index we built in Tab 1 — no need to rebuild <-comment associated with old rag_utils
    #retriever = load_or_build_index(union_name)  <-code associated with old rag_utils
    vectorstore, chunks = load_or_build_index(union_name)
    retriever = build_hybrid_retriever(vectorstore, chunks)

    # This finds the most relevant clauses to ground the analysis
    search_query = facts    
    relevant_docs = retriever(search_query)

    # Build context from retrieved clauses
    context_text = "\n\n---\n\n".join([
        f"[Page {doc.metadata.get('page', '?') + 1}]\n{doc.page_content}"
        for doc in relevant_docs
    ])

    # Structured prompt — we tell Gemini exactly what sections to produce
    # This ensures consistent, organized output every time
    prompt = f"""You are a senior Labour Relations advisor at the Toronto District School Board (TDSB).
You are helping an LR professional prepare for a grievance meeting.
Your analysis must be grounded in the collective agreement excerpts provided.
Always take the perspective of the Board (employer), not the union.

--- GRIEVANCE DETAILS ---
Union: {union_name}
Stage: {stage}
Facts: {facts}
Remedy Sought: {remedy}

--- RELEVANT COLLECTIVE AGREEMENT EXCERPTS ---
{context_text}

--- YOUR ANALYSIS ---
Provide your analysis in exactly these five sections:

1. RISK ASSESSMENT
Rate the Board's risk as High, Moderate, or Low and explain why in 2-3 sentences.

2. RECOMMENDED BOARD POSITION
State the recommended position the Board should take in this grievance in 2-3 sentences.

3. KEY ARGUMENTS FOR THE BOARD
List 3-5 specific arguments the Board can make, grounded in the CA language provided.

4. RELEVANT CA CLAUSES
List the specific article numbers and clause references most relevant to this grievance.

5. SUGGESTED INVESTIGATION QUESTIONS
List 4-6 questions the LR advisor should investigate before the grievance meeting."""

    # Call Gemini
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    return response.text


def render():
    """
    Renders the Tab 2 UI — Grievance Prep Assistant
    """

    # --- Header ---
    st.header("📋 Grievance Prep Assistant")
    st.write(
        "Enter the grievance details below to generate a structured risk assessment "
        "and preparation guide grounded in the collective agreement."
    )

    st.divider()

    # --- Input form ---
    # Two columns for union and article number side by side
  
    union_name = st.selectbox(
        label="Union",
        options=list(CA_FILES.keys()),
        key="grievance_union"
    )


    # Stage of grievance dropdown
    stage = st.selectbox(
        label="Stage of Grievance",
        options=["Informal", "Step 1", "Step 2", "Arbitration"]
    )

    # Facts input — larger text area since this needs detail
    facts = st.text_area(
        label="Brief Description of Facts",
        placeholder="e.g. Teacher was denied a requested leave of absence for a family event. Principal cited operational needs but no documentation was provided.",
        height=150
    )

    # Remedy sought
    remedy = st.text_input(
        label="Remedy Being Sought by Union",
        placeholder="e.g. Approval of leave, compensation for lost wages"
    )

    # --- Submit button ---
    submitted = st.button("Analyze Grievance", type="primary")

    if submitted:

        # Validation — make sure key fields are filled
        if not facts.strip():
            st.warning("Please describe the facts of the grievance.")
            return

        # Get API key from Streamlit secrets
        api_key = st.secrets["GEMINI_API_KEY"]

        with st.spinner("Retrieving relevant clauses and analyzing grievance..."):
            analysis = get_grievance_analysis(
                union_name=union_name,
                facts=facts,
                stage=stage,
                remedy=remedy,
                api_key=api_key
            )

        # --- Display result ---
        import pytz
        est =pytz.timezone("America/Toronto")
        timestamp = datetime.now().strftime("%I:%M %p %Z")

        st.success("Analysis complete")

        with st.container(border=True):
            st.markdown(f"**Grievance Analysis** &nbsp;·&nbsp; *Generated at {timestamp}*")
            st.divider()
            st.markdown(analysis)