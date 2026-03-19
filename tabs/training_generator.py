# tabs/training_generator.py
# Tab 3: Training Material Generator
# Generates a plain-language reference guide for school administrators
# on key Labour Relations topics, grounded in the relevant CA
#
# Features:
# - 6 topic areas grounded in CA via hybrid retrieval
# - Preview / Edit Content tabs for reviewing and tweaking before download
# - Refine with additional instructions
# - Download as a polished PDF

import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime
import pytz
import json
import re
import google.genai as genai
from utils.rag_utils import load_or_build_index, build_hybrid_retriever, CA_FILES, GEMINI_MODEL

# --- Topic definitions ---
# Each topic maps to a search query for CA retrieval
TRAINING_TOPICS = {
    "Attendance Management and Sick Leave":
        "sick leave attendance management documentation medical confirmation short term disability",
    "Return to Work and Modified Duties":
        "return to work modified duties medical restrictions accommodation gradual return",
    "Workplace Accommodation Obligations":
        "accommodation duty to accommodate undue hardship Human Rights Code disability",
    "Managing Sick Leave Without Triggering Accommodation":
        "sick leave attendance culpable innocent absenteeism accommodation trigger medical",
    "Leave of Absence — Approval and Denial":
        "leave of absence approval denial application documentation discretionary",
    "Progressive Discipline":
        "discipline reprimand suspension dismissal just cause progressive discipline",
}

# Section definitions — labels and keys used throughout
SECTIONS = [
    ("overview",       "1. Overview"),
    ("provisions",     "2. Key CA Provisions"),
    ("definitions",    "3. Key Definitions"),
    ("dos_donts",      "4. Dos and Don'ts"),
    ("mistakes",       "5. Common Mistakes"),
    ("faq",            "6. Frequently Asked Questions"),
    ("legislation",    "7. Relevant Legislation"),
    ("call_lr",        "8. When to Call LR"),
]


def generate_guide_json(
    union_name: str,
    topic: str,
    context: str,
    api_key: str
) -> dict:
    """
    Calls Gemini to generate the reference guide content.
    Returns a Python dict with one key per section.
    Using JSON ensures reliable parsing into the edit form.
    """
    # Retrieve relevant CA provisions
    search_query = TRAINING_TOPICS[topic]
    if context.strip():
        search_query = f"{search_query} {context}"

    vectorstore, chunks = load_or_build_index(union_name)
    retriever = build_hybrid_retriever(vectorstore, chunks)
    relevant_docs = retriever(search_query)

    context_text = "\n\n---\n\n".join([
        f"[{doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
        for doc in relevant_docs
        if len(doc.page_content.strip()) >= 150
    ])

    prompt = f"""You are a senior Labour Relations advisor at the Toronto District School Board (TDSB).
You are creating a plain-language reference guide for school principals and administrators.

Topic: {topic}
Union: {union_name} Collective Agreement (2022-2026)
{f'Additional context: {context}' if context.strip() else ''}

Use the collective agreement excerpts provided to ground the guide in actual CA language.
Also draw on your knowledge of relevant Ontario legislation including the Ontario Human Rights Code,
Employment Standards Act, and Occupational Health and Safety Act where applicable.

Write for a principal audience — intelligent but not LR experts. Be practical and specific.

Return ONLY a valid JSON object with exactly these 8 keys. No markdown, no code fences, just raw JSON:

{{
  "overview": "3-4 sentence overview of the topic and why it matters for principals.",
  "provisions": "Key CA provisions that apply. For each cite the section reference (e.g. L38.0, C9.1) and explain in plain language what it means. Use clear paragraph breaks between provisions.",
  "definitions": "4-6 key terms defined in plain language. Format as TERM: definition, one per line.",
  "dos_donts": "Two clearly labelled lists. Start with DO: followed by at least 5 specific actionable items. Then DONT: followed by at least 5 specific items principals must avoid.",
  "mistakes": "4-5 specific common mistakes principals make on this topic and why each is a problem. Number each one.",
  "faq": "5-6 realistic questions a principal would ask with clear direct answers. Format as Q: followed by A: for each.",
  "legislation": "Plain language summary of which Ontario laws apply and what each requires. Cover OHRC, ESA, and any other relevant legislation.",
  "call_lr": "At least 5 specific trigger situations where the principal must contact Labour Relations before acting. Number each one."
}}

--- COLLECTIVE AGREEMENT EXCERPTS ---
{context_text}"""

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    # Clean response and parse JSON
    raw = response.text.strip()
    # Strip markdown code fences if Gemini adds them despite instructions
    raw = re.sub(r'^```json\s*', '', raw)
    raw = re.sub(r'^```\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # If JSON parsing fails return a graceful error dict
        return {key: f"Content generation error for this section. Please try again."
                for key, _ in SECTIONS}


def build_html(guide_data: dict, topic: str, union_name: str) -> str:
    """
    Renders the guide data dict into the polished HTML template.
    White background, professional grey/black palette.
    Gemini content is post-processed to add formatting before rendering.
    """
    est = pytz.timezone("America/Toronto")
    generated_date = datetime.now(est).strftime("%B %d, %Y %I:%M %p %Z")

    def format_content(key: str, content: str) -> str:
        """
        Post-processes Gemini's plain text output into formatted HTML.
        Handles bold, italics, lists, removes asterisks, formats Q&A etc.
        """

        # Handle dos_donts dict format before any conversion
        if key == "dos_donts" and isinstance(content, dict):
            result = '<p class="section-label do-label">✓ What To Do</p><ul class="do-list">'
            for item in content.get('DO', content.get('do', [])):
                result += f'<li>{item}</li>'
            result += '</ul>'
            result += '<p class="section-label dont-label">✗ What Not To Do</p><ul class="dont-list">'
            for item in content.get('DONT', content.get('dont', content.get("DON'T", []))):
                result += f'<li>{item}</li>'
            result += '</ul>'
            return result

        if isinstance(content, dict):
            content = "\n".join(f"{k}: {v}" for k, v in content.items())
        elif isinstance(content, list):
            content = "\n".join(str(item) for item in content)
        else:
            content = str(content)

        # Remove asterisks used as bullet points or emphasis markers
        content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
        content = re.sub(r'\*(.+?)\*', r'<em>\1</em>', content)
        content = re.sub(r'^\*\s*', '', content, flags=re.MULTILINE)

        lines = content.split('\n')
        html_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Section: Dos and Don'ts — style DO and DON'T lines distinctively
            if key == "dos_donts":
                if line.upper().startswith("DO:") or line.upper().startswith("DOS:"):
                    html_lines.append(f'<p class="section-label do-label">✓ What To Do</p><ul class="do-list">')
                    continue
                elif line.upper().startswith("DONT:") or line.upper().startswith("DON'T:") or line.upper().startswith("DON'TS:"):
                    html_lines.append(f'</ul><p class="section-label dont-label">✗ What Not To Do</p><ul class="dont-list">')
                    continue
                else:
                    # Remove leading numbers or dashes
                    clean = re.sub(r'^[\d\-\.\)]+\s*', '', line)
                    html_lines.append(f'<li>{clean}</li>')
                    continue

            # Section: FAQ — format Q: and A: pairs
            if key == "faq":
                if line.upper().startswith("Q:"):
                    question = line[2:].strip()
                    html_lines.append(f'<p class="faq-question">Q: {question}</p>')
                    continue
                elif line.upper().startswith("A:"):
                    answer = line[2:].strip()
                    html_lines.append(f'<p class="faq-answer">A: {answer}</p>')
                    continue

            # Section: Definitions — format TERM: definition
            if key == "definitions":
                if ':' in line:
                    parts = line.split(':', 1)
                    term = parts[0].strip()
                    definition = parts[1].strip() if len(parts) > 1 else ''
                    html_lines.append(
                        f'<p class="definition"><span class="def-term">{term}:</span> {definition}</p>'
                    )
                    continue

            # Numbered items — mistakes and when to call LR
            if key in ("mistakes", "call_lr"):
                clean = re.sub(r'^[\d]+[\.\)]\s*', '', line)
                html_lines.append(f'<li>{clean}</li>')
                continue

            # CA reference numbers — highlight inline
            line = re.sub(
                r'\b(C\d+\.\d+|L\d+\.\d+|L-[A-Z]\.\d+\.\d+|\d+\.\d+\.\d+)\b',
                r'<span class="ref">\1</span>',
                line
            )

            html_lines.append(f'<p>{line}</p>')

        result = '\n'.join(html_lines)

        # Wrap numbered list sections in <ul> tags
        if key in ("mistakes", "call_lr"):
            result = f'<ul class="numbered-list">{result}</ul>'

        # Close any unclosed lists in dos_donts
        if key == "dos_donts" and not result.endswith('</ul>'):
            result += '</ul>'

        return result

    # Build all sections
    sections_html = ""
    for key, label in SECTIONS:
        content = guide_data.get(key, "")
        formatted = format_content(key, content)
        sections_html += f'<h2 id="{key.replace("_", "-")}">{label}</h2>\n{formatted}\n'

    return f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Source+Sans+3:wght@300;400;600&display=swap" rel="stylesheet">
        <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}

            body {{
                font-family: 'Source Sans 3', sans-serif;
                font-weight: 300;
                background: #ffffff;
                color: #1a1a1a;
                display: flex;
                min-height: 100vh;
            }}

            /* --- Sidebar --- */
            .sidebar {{
                width: 210px;
                min-width: 210px;
                background: #1a1a1a;
                color: #e0e0e0;
                padding: 28px 18px;
                position: sticky;
                top: 0;
                height: 100vh;
                overflow-y: auto;
            }}

            .sidebar-title {{
                font-family: 'Playfair Display', serif;
                font-size: 11px;
                letter-spacing: 2.5px;
                text-transform: uppercase;
                color: #888;
                margin-bottom: 20px;
                padding-bottom: 12px;
                border-bottom: 1px solid #333;
            }}

            .sidebar nav a {{
                display: block;
                color: #aaa;
                text-decoration: none;
                font-size: 11.5px;
                font-weight: 400;
                padding: 7px 0;
                border-bottom: 1px solid #2a2a2a;
                transition: color 0.2s;
                line-height: 1.4;
            }}

            .sidebar nav a:hover {{ color: #ffffff; }}

            /* --- Main Content --- */
            .main {{
                flex: 1;
                padding: 48px 56px;
                max-width: 820px;
                background: #ffffff;
            }}

            /* --- Document Header --- */
            .doc-header {{
                border-bottom: 2px solid #1a1a1a;
                padding-bottom: 20px;
                margin-bottom: 32px;
            }}

            .doc-label {{
                font-size: 10px;
                letter-spacing: 3px;
                text-transform: uppercase;
                color: #888;
                font-weight: 600;
                margin-bottom: 10px;
            }}

            .doc-title {{
                font-family: 'Playfair Display', serif;
                font-size: 26px;
                color: #1a1a1a;
                line-height: 1.3;
                margin-bottom: 10px;
            }}

            .doc-meta {{
                font-size: 11.5px;
                color: #999;
                font-weight: 400;
            }}

            /* --- Disclaimer --- */
            .disclaimer {{
                background: #f5f5f5;
                border-left: 3px solid #555;
                padding: 12px 16px;
                font-size: 12px;
                color: #555;
                margin-bottom: 40px;
                line-height: 1.7;
                border-radius: 0 4px 4px 0;
            }}

            .disclaimer strong {{
                color: #1a1a1a;
                font-weight: 600;
            }}

            /* --- Section Headings --- */
            h2 {{
                font-family: 'Playfair Display', serif;
                font-size: 18px;
                color: #1a1a1a;
                margin-top: 44px;
                margin-bottom: 14px;
                padding-bottom: 8px;
                border-bottom: 1px solid #e0e0e0;
                scroll-margin-top: 20px;
            }}

            /* --- Body Text --- */
            p {{
                font-size: 13.5px;
                line-height: 1.85;
                margin-bottom: 10px;
                font-weight: 300;
                color: #333;
            }}

            strong {{
                font-weight: 600;
                color: #1a1a1a;
            }}

            em {{
                font-style: italic;
                color: #444;
            }}

            /* --- Lists --- */
            ul {{
                margin: 8px 0 14px 0;
                padding-left: 0;
                list-style: none;
            }}

            li {{
                font-size: 13.5px;
                line-height: 1.75;
                margin-bottom: 8px;
                font-weight: 300;
                color: #333;
                padding-left: 16px;
                position: relative;
            }}

            li::before {{
                content: '–';
                position: absolute;
                left: 0;
                color: #999;
            }}

            /* --- Dos and Don'ts --- */
            .section-label {{
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 2px;
                text-transform: uppercase;
                margin: 16px 0 8px 0;
            }}

            .do-label {{ color: #2d6a4f; }}
            .dont-label {{ color: #c1121f; margin-top: 24px; }}

            .do-list li::before {{ color: #2d6a4f; content: '✓'; }}
            .dont-list li::before {{ color: #c1121f; content: '✗'; }}

            /* --- FAQ --- */
            .faq-question {{
                font-weight: 600;
                color: #1a1a1a;
                margin-top: 16px;
                margin-bottom: 4px;
                font-size: 13.5px;
            }}

            .faq-answer {{
                color: #444;
                font-size: 13.5px;
                padding-left: 16px;
                border-left: 2px solid #e0e0e0;
                margin-bottom: 12px;
            }}

            /* --- Definitions --- */
            .definition {{
                margin-bottom: 10px;
                padding: 8px 12px;
                background: #fafafa;
                border-radius: 4px;
            }}

            .def-term {{
                font-weight: 600;
                color: #1a1a1a;
                font-size: 13.5px;
            }}

            /* --- Numbered lists --- */
            .numbered-list {{
                counter-reset: item;
                list-style: none;
                padding-left: 0;
            }}

            .numbered-list li {{
                counter-increment: item;
                padding-left: 28px;
            }}

            .numbered-list li::before {{
                content: counter(item) '.';
                position: absolute;
                left: 0;
                color: #999;
                font-weight: 600;
                font-size: 12px;
            }}

            /* --- CA Reference tags --- */
            .ref {{
                background: #f0f0f0;
                color: #1a1a1a;
                font-size: 11px;
                font-weight: 600;
                padding: 1px 5px;
                border-radius: 3px;
                font-family: monospace;
                border: 1px solid #ddd;
            }}

            /* --- Footer --- */
            .doc-footer {{
                margin-top: 60px;
                padding-top: 16px;
                border-top: 1px solid #e0e0e0;
                font-size: 11px;
                color: #bbb;
                text-align: center;
            }}

            @media print {{
                .sidebar {{ display: none; }}
                .main {{ padding: 20px; max-width: 100%; }}
            }}
        </style>
    </head>
    <body>
        <div class="sidebar">
            <div class="sidebar-title">Contents</div>
            <nav>
                <a href="#overview">1. Overview</a>
                <a href="#provisions">2. Key CA Provisions</a>
                <a href="#definitions">3. Key Definitions</a>
                <a href="#dos-donts">4. Dos and Don'ts</a>
                <a href="#mistakes">5. Common Mistakes</a>
                <a href="#faq">6. Frequently Asked Questions</a>
                <a href="#legislation">7. Relevant Legislation</a>
                <a href="#call-lr">8. When to Call LR</a>
            </nav>
        </div>

        <div class="main">
            <div class="doc-header">
                <div class="doc-label">TDSB Labour Relations — Administrator Reference Guide</div>
                <div class="doc-title">{topic}</div>
                <div class="doc-meta">
                    {union_name} &nbsp;·&nbsp; {generated_date}
                </div>
            </div>

            <div class="disclaimer">
                <strong>Important:</strong> This reference guide is intended as a practical job aid
                for school administrators. It does not constitute legal advice and should not replace
                consultation with your Labour Relations advisor on complex or sensitive matters.
                Always verify provisions against the current collective agreement.
            </div>

            <div id="guide-content">
                {sections_html}
            </div>

            <div class="doc-footer">
                LR Operations Assistant &nbsp;·&nbsp; TDSB &nbsp;·&nbsp; {generated_date}
            </div>
    </div>

        <div style="position:fixed; bottom:20px; right:20px; z-index:999;">
            <button onclick="window.print()" style="
                background:#1a1a1a;
                color:white;
                border:none;
                padding:10px 20px;
                font-family:'Source Sans 3',sans-serif;
                font-size:13px;
                cursor:pointer;
                border-radius:4px;
                box-shadow:0 2px 8px rgba(0,0,0,0.3);
            ">🖨️ Print / Save as PDF</button>
        </div>

    </body>
    </html>
    """

def render():
    """
    Renders Tab 3 — Training Material Generator UI
    """
    st.header("🎓 Training Material Generator")
    st.write(
        "Generate a plain-language reference guide for school administrators "
        "on key Labour Relations topics, grounded in the collective agreement."
    )

    st.divider()

    # --- Input form ---
    col1, col2 = st.columns(2)

    with col1:
        union_name = st.selectbox(
            label="Union",
            options=list(CA_FILES.keys()),
            key="training_union"
        )

    with col2:
        topic = st.selectbox(
            label="Topic",
            options=list(TRAINING_TOPICS.keys()),
            key="training_topic"
        )

    context = st.text_area(
        label="Additional context (optional)",
        placeholder="e.g. Focus on occasional teachers, or this is for new principals, or we've had recent grievances on this topic...",
        height=80
    )

    submitted = st.button("Generate Reference Guide", type="primary")

    if submitted:
        api_key = st.secrets["GEMINI_API_KEY"]

        with st.spinner("Retrieving CA provisions and generating reference guide..."):
            guide_data = generate_guide_json(
                union_name=union_name,
                topic=topic,
                context=context,
                api_key=api_key
            )

        # Store in session state so tabs and buttons don't trigger full rerun
        st.session_state['guide_data'] = guide_data
        st.session_state['guide_topic'] = topic
        st.session_state['guide_union'] = union_name
        st.success("Reference guide generated")

    # --- Display area — only shown after generation ---
    if 'guide_data' not in st.session_state:
        return

    guide_data = st.session_state['guide_data']
    topic = st.session_state['guide_topic']
    union_name = st.session_state['guide_union']

    st.divider()

    # --- Preview / Edit tabs ---
    preview_tab, edit_tab = st.tabs(["📄 Preview", "✏️ Edit Content"])

    with preview_tab:
        # Build and display the HTML preview
        preview_html = build_html(guide_data, topic, union_name)

        # Download button above the preview
        st.markdown(
            '<div style="background:#f5f5f5; border-left:3px solid #555; padding:10px 14px; '
            'font-size:12px; color:#555; border-radius:0 4px 4px 0;">'
            '💡 To save as PDF: click the guide below to focus it, then press '
            'Ctrl+P (or Cmd+P on Mac) and select "Save as PDF".</div>',
            unsafe_allow_html=True
        )

        if st.button(" Refresh Preview"):
            st.rerun()
        components.html(preview_html, height=900, scrolling=True)

    with edit_tab:
        st.write("Edit any section below. Changes will be reflected when you download the PDF.")

        # Create a text area for each section
        edited_data = {}
        for key, label in SECTIONS:
            edited_data[key] = st.text_area(
                label=label,
                value=guide_data.get(key, ""),
                height=200,
                key=f"edit_{key}"
            )

        # Download from edited content
        col1, col2 = st.columns(2)

        with col1:
            if st.button("💾 Save Edits", type="primary"):
                st.session_state['guide_data'] = edited_data
                st.success("Edits saved - switch to Preview tab to see changes.")

        with col2:
            st.markdown(
                '<div style="background:#f5f5f5; border-left:3px solid #555; padding:10px 14px; '
                'font-size:12px; color:#555; border-radius:0 4px 4px 0;">'
                '💡 To save edited version as PDF: go to Preview tab, click Refresh Preview, '
                'then press Ctrl+P (or Cmd+P on Mac).</div>',
                unsafe_allow_html=True
            )

    # --- Refine section ---
    st.divider()
    st.markdown("**🔄 Refine this guide**")
    st.caption("Add instructions to generate a new version with different focus or additional detail.")

    refinement = st.text_area(
        label="Refinement instructions",
        placeholder="e.g. Focus more on occasional teachers, add more detail on documentation requirements, include specific examples of common scenarios...",
        height=80,
        key="refinement_input"
    )

    if st.button("Generate Refined Version", type="secondary"):
        if refinement.strip():
            api_key = st.secrets["GEMINI_API_KEY"]
            with st.spinner("Generating refined version..."):
                refined_data = generate_guide_json(
                    union_name=union_name,
                    topic=topic,
                    context=refinement,
                    api_key=api_key
                )
                st.session_state['guide_data'] = refined_data
                st.rerun()
        else:
            st.warning("Please enter refinement instructions.")
