# tabs/roadmap.py
# Tab 4: Product Roadmap
# Presents the strategic vision for future development
# Pure visual presentation — no API calls needed

import streamlit as st


def render():
    """
    Renders Tab 4 — Product Roadmap
    """

    st.header("📊 Product Roadmap")
    st.write(
        "The LR Operations Assistant is designed as a complete Labour Relations "
        "intelligence system — from day-to-day agreement interpretation through to "
        "strategic bargaining preparation."
    )

    st.divider()

    # --- System vision statement ---
    st.markdown(
        """
        <div style="
            background:#fdf3dc;
            color:#e8e4d9;
            padding:24px 28px;
            border-radius:6px;
            margin-bottom:32px;
        ">
            <div style="
                font-size:10px;
                letter-spacing:3px;
                text-transform:uppercase;
                color:#8a6a1f;
                font-weight:600;
                margin-bottom:10px;
            ">Strategic Vision</div>
            <div style="
                font-size:16px;
                line-height:1.7;
                font-weight:300;
            ">
                The tools built here address the operational core of the LR Lead role — 
                interpreting agreements, preparing for grievances, and building administrator capacity. 
                The next phase extends that foundation into <strong style="color:#8a6a1f;">
                systematic intelligence</strong> — turning the day-to-day work of labour relations 
                into organizational knowledge that informs strategy, reduces risk, and strengthens 
                the Board's position at the bargaining table.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Current state ---
    st.markdown("#### ✅ Currently Available")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div style="
                border:1px solid #e0e0e0;
                border-top:3px solid #fdf3dc;
                border-radius:4px;
                padding:16px;
                height:160px;
            ">
                <div style="font-weight:600; font-size:14px; margin-bottom:8px;">
                    📄 CA Interpreter
                </div>
                <div style="font-size:12px; color:#555; line-height:1.6;">
                    Plain-language interpretation of collective agreement provisions, 
                    grounded in actual clause text with source citations.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div style="
                border:1px solid #e0e0e0;
                border-top:3px solid #fdf3dc;
                border-radius:4px;
                padding:16px;
                height:160px;
            ">
                <div style="font-weight:600; font-size:14px; margin-bottom:8px;">
                    📋 Grievance Prep
                </div>
                <div style="font-size:12px; color:#555; line-height:1.6;">
                    Structured risk assessment, recommended Board position, 
                    key arguments and investigation questions for grievance preparation.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div style="
                border:1px solid #e0e0e0;
                border-top:3px solid #fdf3dc;
                border-radius:4px;
                padding:16px;
                height:160px;
            ">
                <div style="font-weight:600; font-size:14px; margin-bottom:8px;">
                     🧭 Field Guide Generator
                </div>
                <div style="font-size:12px; color:#555; line-height:1.6;">
                    Plain-language reference guides for school administrators 
                    on key LR topics, grounded in the collective agreement.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔜 Year 1 Roadmap")

    # --- Roadmap items ---
    roadmap_items = [
        {
            "icon": "❶",
            "title": "Grievance Intelligence Dashboard",
            "problem": "Grievance data held across multiple files and systems represents an untapped source of operational and strategic intelligence.",
            "capabilities": [
                "Deadline tracker — calculates Step 1, Step 2 and arbitration timelines, flags files at risk",
                "Trend analytics — grievance volume by union, article, school and stage over time",
                "Arbitration outcomes — win/loss tracking with root cause categorization",
                "Cost exposure reporting — models financial impact of active grievance files",
            ],
            "value": "Operational Risk Management"
        },
        {
            "icon": "❷",
            "title": "Regulatory Radar",
            "problem": "Legislative changes to the ESA, OHRC and Labour Relations Act can create CA compliance gaps that go undetected until a grievance is filed.",
            "capabilities": [
                "Monitors Ontario legislative amendments in real time",
                "Flags collective agreement articles potentially affected by new legislation",
                "Generates plain-language briefing notes on regulatory changes",
                "Alerts LR team before changes take effect — not after",
            ],
            "value": "Proactive Risk Management"
        },
        {
            "icon": "❸",
            "title": "Bargaining Intelligence Module",
            "problem": "Bargaining preparation relies heavily on institutional memory and manual research — both of which are time-intensive and inconsistent.",
            "capabilities": [
                "Analyzes grievance trend data to identify high-dispute clauses",
                "Benchmarks TDSB collective agreement language against comparator boards",
                "Surfaces emerging arbitration trends relevant to TDSB",
                "Generates pre-bargaining briefs for each union relationship",
            ],
            "value": "Strategic Bargaining Advantage"
        },
    ]

    for item in roadmap_items:
        st.markdown(
            f"""
            <div style="
                border:1px solid #e0e0e0;
                border-left:4px solid #fdf3dc;
                border-radius:0 6px 6px 0;
                padding:20px 24px;
                margin-bottom:16px;
                background:#fafafa;
            ">
                <div style="display:flex; align-items:flex-start; gap:16px;">
                    <div style="font-size:28px; line-height:1;">{item['icon']}</div>
                    <div style="flex:1;">
                        <div style="
                            display:flex;
                            justify-content:space-between;
                            align-items:flex-start;
                            margin-bottom:8px;
                        ">
                            <div style="font-weight:600; font-size:15px; color:#fdf3dc;">
                                {item['title']}
                            </div>
                            <div style="
                                font-size:10px;
                                font-weight:600;
                                letter-spacing:1.5px;
                                text-transform:uppercase;
                                color:#8a6a1f;
                                background:#fdf3dc;
                                padding:3px 8px;
                                border-radius:3px;
                                white-space:nowrap;
                                margin-left:12px;
                            ">{item['value']}</div>
                        </div>
                        <div style="
                            font-size:12px;
                            color:#666;
                            font-style:italic;
                            margin-bottom:12px;
                            line-height:1.5;
                        ">
                            <strong style="color:#555;">The gap:</strong> {item['problem']}
                        </div>
                        <div style="display:flex; flex-wrap:wrap; gap:6px;">
                            {" ".join([
                                f'<div style="'
                                f'font-size:11px; color:#444; background:#fff; '
                                f'border:1px solid #ddd; border-radius:3px; '
                                f'padding:3px 8px; line-height:1.5;">'
                                f'→ {cap}</div>'
                                for cap in item['capabilities']
                            ])}
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    # --- Closing note ---
    st.markdown(
        """
        <div style="
            font-size:12px;
            color:#999;
            text-align:center;
            padding:8px;
            font-style:italic;
        ">
            Roadmap priorities reflect the operational realities of large, multi-unionized school board LR practice and the strategic needs of a Senior Manager in this environment.
        </div>
        """,
        unsafe_allow_html=True
    )
