import os
import json

import pandas as pd
import streamlit as st

from src.config import INDEX_PATH, DATA_PATH
from src.support_graph import run_support_workflow
from src.rag_pipeline import retrieve_documents


REPORTS_PATH = "reports/rag_stats.json"

PAGES = [
    "AI Support Chat",
    "FAQ Explorer",
    "Knowledge Insights",
    "Visualization Gallery",
]


st.set_page_config(
    page_title="Jazz Telecom Support AI",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed",
)


CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(220, 20, 60, 0.35), transparent 28%),
        radial-gradient(circle at top right, rgba(255, 255, 255, 0.10), transparent 30%),
        radial-gradient(circle at bottom left, rgba(139, 0, 0, 0.30), transparent 32%),
        linear-gradient(135deg, #0b0b0b 0%, #1a1a1a 45%, #2b0000 100%);
    color: white;
}

header[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stDecoration"] {
    display: none;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

label,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
.stTextInput label,
.stTextArea label,
.stSelectbox label,
.stRadio label,
.stSlider label {
    color: #ffffff !important;
    font-weight: 700 !important;
}

.stTextInput input,
.stTextArea textarea,
.stNumberInput input {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-radius: 14px !important;
    border: 2px solid #dc143c !important;
}

div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-radius: 14px !important;
    border: 2px solid #dc143c !important;
}

div[data-baseweb="select"] span {
    color: #000000 !important;
}

ul[role="listbox"] {
    background-color: #ffffff !important;
}

li[role="option"],
li[role="option"] span,
li[role="option"] div {
    color: #000000 !important;
    background-color: #ffffff !important;
}

.main-title {
    font-size: 3rem;
    line-height: 1.05;
    font-weight: 800;
    margin-bottom: 0.6rem;
    background: linear-gradient(90deg, #ffffff, #ff4d6d, #dc143c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.main-subtitle {
    font-size: 1.05rem;
    color: #f5f5f5;
    max-width: 980px;
    margin-bottom: 1.5rem;
}

.glass-card {
    padding: 1.5rem;
    border-radius: 24px;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(220, 20, 60, 0.35);
    box-shadow: 0 22px 45px rgba(0, 0, 0, 0.30);
    backdrop-filter: blur(16px);
    margin-bottom: 1.2rem;
}

.section-title {
    font-size: 1.55rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    color: #ffffff;
}

.section-caption {
    color: #f1f1f1;
    font-size: 0.95rem;
}

.nav-title {
    font-size: 1.7rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 0.3rem;
}

.nav-subtitle {
    color: #ffd6dd;
    font-size: 0.95rem;
    font-weight: 600;
    margin-bottom: 1rem;
}

.nav-section {
    color: #ffffff;
    font-size: 1.05rem;
    font-weight: 800;
    margin-top: 1rem;
    margin-bottom: 0.6rem;
}

.feature-box {
    margin-top: 1rem;
    padding: 1rem;
    border-radius: 18px;
    color: #f5f5f5;
    font-size: 0.95rem;
    line-height: 1.9;
    background: rgba(220, 20, 60, 0.12);
    border: 1px solid rgba(220, 20, 60, 0.30);
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 28px !important;
    background:
        linear-gradient(180deg, rgba(43, 0, 0, 0.95), rgba(26, 26, 26, 0.92)),
        rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(220, 20, 60, 0.35) !important;
    box-shadow: 0 22px 50px rgba(0, 0, 0, 0.34) !important;
}

div[role="radiogroup"] label {
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(220, 20, 60, 0.30) !important;
    border-radius: 18px !important;
    padding: 0.9rem 1rem !important;
    margin-bottom: 0.75rem !important;
    width: 100% !important;
}

div[role="radiogroup"] label:hover {
    background: linear-gradient(90deg, #8b0000, #dc143c, #ff4d6d) !important;
}

div[role="radiogroup"] p {
    color: #ffffff !important;
    font-weight: 800 !important;
}

.answer-card {
    padding: 1.5rem;
    border-radius: 24px;
    background: linear-gradient(135deg, rgba(220, 20, 60, 0.25), rgba(139, 0, 0, 0.18));
    border: 1px solid rgba(255, 77, 109, 0.45);
    color: #ffffff;
}

.source-card {
    padding: 1rem;
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(220, 20, 60, 0.25);
    margin-bottom: 0.8rem;
}

.metric-card {
    padding: 1.2rem;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(220, 20, 60, 0.20), rgba(255, 255, 255, 0.06));
    border: 1px solid rgba(220, 20, 60, 0.30);
    text-align: center;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #ff4d6d;
}

.metric-label {
    font-size: 0.85rem;
    color: #f5f5f5;
}

div.stButton > button:first-child {
    width: 100%;
    border-radius: 18px;
    height: 3.1rem;
    border: none;
    font-size: 0.95rem;
    font-weight: 800;
    color: white;
    background: linear-gradient(90deg, #8b0000, #dc143c, #ff4d6d);
}

.stTabs [data-baseweb="tab"] {
    border-radius: 999px;
    padding: 10px 18px;
    background: rgba(255, 255, 255, 0.08);
    color: white;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #8b0000, #dc143c);
}
</style>
"""


st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


if "menu_open" not in st.session_state:
    st.session_state.menu_open = True

if "page" not in st.session_state:
    st.session_state.page = "AI Support Chat"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def toggle_menu():
    st.session_state.menu_open = not st.session_state.menu_open


@st.cache_resource
def check_index():
    return os.path.exists(INDEX_PATH)


def load_stats():
    if not os.path.exists(REPORTS_PATH):
        return None
    with open(REPORTS_PATH, "r") as file:
        return json.load(file)


def display_metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


index_exists = check_index()
stats = load_stats()


top_col1, top_col2 = st.columns([0.16, 0.84])

with top_col1:
    button_label = "☰ Open Menu" if not st.session_state.menu_open else "✕ Close Menu"
    st.button(button_label, on_click=toggle_menu)

with top_col2:
    st.markdown(
        """
        <div class="main-title">Jazz Telecom Support RAG Chatbot</div>
        <div class="main-subtitle">
            A telecom and JazzCash-style AI support assistant using RAG, LangGraph,
            multilingual FAQ retrieval, and grounded customer support answers.
        </div>
        """,
        unsafe_allow_html=True,
    )


if not index_exists:
    st.error(
        "RAG index not found. Build it first by running:\n\n"
        "`python3.11 -m src.build_index`"
    )
    st.stop()


if st.session_state.menu_open:
    menu_col, content_col = st.columns([0.25, 0.75], gap="large")
else:
    content_col = st.container()


if st.session_state.menu_open:
    with menu_col:
        with st.container(border=True):
            st.markdown(
                """
                <div class="nav-title">📱 Jazz Support AI</div>
                <div class="nav-subtitle">Telecom + JazzCash RAG Dashboard</div>
                <div class="nav-section">Navigation</div>
                """,
                unsafe_allow_html=True,
            )

            selected_page = st.radio(
                label="Select dashboard section",
                options=PAGES,
                index=PAGES.index(st.session_state.page),
                label_visibility="collapsed",
            )

            st.session_state.page = selected_page

            st.markdown(
                """
                <div class="nav-section">Project Features</div>
                <div class="feature-box">
                    • English + Roman Urdu support<br>
                    • TF-IDF RAG retrieval<br>
                    • LangGraph workflow<br>
                    • Ollama answer generation<br>
                    • FAQ source citation<br>
                    • JazzCash-inspired UI
                </div>
                """,
                unsafe_allow_html=True,
            )


with content_col:
    page = st.session_state.page

    if page == "AI Support Chat":
        left_col, right_col = st.columns([1.0, 1.0])

        with left_col:
            st.markdown(
                """
                <div class="glass-card">
                    <div class="section-title">Customer Support Chat</div>
                    <div class="section-caption">
                        Ask balance, package, recharge, JazzCash, SIM, or internet questions.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            user_query = st.text_area(
                "Ask your support question",
                height=160,
                placeholder="Example: balance kaise check karun? / How do I send money using JazzCash?",
            )

            example_questions = [
                "How can I check my balance?",
                "internet package activate kaise karein?",
                "How do I send money using JazzCash?",
                "My recharge failed but money was deducted",
                "SIM par network nahi aa raha",
            ]

            st.write("Quick examples:")
            example_col1, example_col2 = st.columns(2)

            for index, example in enumerate(example_questions):
                target_col = example_col1 if index % 2 == 0 else example_col2
                with target_col:
                    if st.button(example, key=f"example_{index}"):
                        user_query = example

            ask_button = st.button("Get Support Answer")

        with right_col:
            st.markdown(
                """
                <div class="glass-card">
                    <div class="section-title">AI Answer</div>
                    <div class="section-caption">
                        Grounded answer generated from retrieved telecom FAQs.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if ask_button:
                if not user_query.strip():
                    st.warning("Please enter a question.")
                else:
                    with st.spinner("Searching knowledge base and generating answer..."):
                        workflow_result = run_support_workflow(user_query)

                    st.session_state.chat_history.append(
                        {
                            "question": user_query,
                            "answer": workflow_result["final_answer"],
                            "sources": workflow_result["retrieved_docs"],
                            "answer_source": workflow_result["answer_source"],
                        }
                    )

            if st.session_state.chat_history:
                latest = st.session_state.chat_history[-1]

                st.markdown(
                    f"""
                    <div class="answer-card">
                        <div class="section-title">Question</div>
                        <div class="section-caption">{latest["question"]}</div>
                        <br>
                        <div class="section-title">Answer</div>
                        <div class="section-caption">{latest["answer"]}</div>
                        <br>
                        <div class="section-caption">Source: {latest["answer_source"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.write("### Retrieved FAQ Sources")

                for source in latest["sources"]:
                    st.markdown(
                        f"""
                        <div class="source-card">
                            <b>{source["category"]}</b> | {source["language"]}<br>
                            <b>Q:</b> {source["question"]}<br>
                            <b>A:</b> {source["answer"]}<br>
                            <b>Score:</b> {source["score"]:.4f}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("Ask a support question to get started.")

    elif page == "FAQ Explorer":
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">FAQ Knowledge Base Explorer</div>
                <div class="section-caption">
                    Browse telecom and JazzCash support FAQs by category and language.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        df = pd.read_csv(DATA_PATH)

        category_filter = st.selectbox(
            "Filter by Category",
            ["All"] + sorted(df["category"].unique().tolist()),
        )

        language_filter = st.selectbox(
            "Filter by Language",
            ["All"] + sorted(df["language"].unique().tolist()),
        )

        search_text = st.text_input("Search FAQ", placeholder="balance, JazzCash, recharge...")

        filtered_df = df.copy()

        if category_filter != "All":
            filtered_df = filtered_df[filtered_df["category"] == category_filter]

        if language_filter != "All":
            filtered_df = filtered_df[filtered_df["language"] == language_filter]

        if search_text.strip():
            filtered_df = filtered_df[
                filtered_df["question"].str.contains(search_text, case=False, na=False)
                | filtered_df["answer"].str.contains(search_text, case=False, na=False)
            ]

        st.dataframe(filtered_df, use_container_width=True)

    elif page == "Knowledge Insights":
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">Knowledge Base Insights</div>
                <div class="section-caption">
                    Statistics about the telecom support FAQ dataset and RAG index.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if stats:
            c1, c2, c3 = st.columns(3)

            with c1:
                display_metric_card("Total FAQs", str(stats["total_faqs"]))
            with c2:
                display_metric_card("Categories", str(stats["categories"]))
            with c3:
                display_metric_card("Languages", str(len(stats["languages"])))

            st.write("Supported languages:", ", ".join(stats["languages"]))
        else:
            st.warning("RAG stats not found. Run build_index first.")

    elif page == "Visualization Gallery":
        st.markdown(
            """
            <div class="glass-card">
                <div class="section-title">Visualization Gallery</div>
                <div class="section-caption">
                    Explore FAQ category and language distribution.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        graph_paths = {
            "Category Distribution": "visualizations/category_distribution.png",
            "Language Distribution": "visualizations/language_distribution.png",
            "Top Categories": "visualizations/top_categories.png",
        }

        tabs = st.tabs(list(graph_paths.keys()))

        for tab, (title, path) in zip(tabs, graph_paths.items()):
            with tab:
                if os.path.exists(path):
                    st.image(path, use_container_width=True)
                else:
                    st.warning(f"{title} not found. Run build_index first.")