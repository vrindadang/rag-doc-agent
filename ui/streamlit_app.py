import sys, subprocess
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ingestor import ingest_pdf, list_ingested_files
from app.retriever import retrieve_similar_chunks
from app.generator import stream_answer, get_sources_display
from agents.supabase_store import get_all_versions, get_version_doc, get_agent_logs

st.set_page_config(page_title="RAG Doc Agent", page_icon="🤖", layout="wide")

st.markdown("""
<style>
[data-testid="stChatInput"] {
    position: fixed;
    bottom: 1rem;
    left: 22rem;   /* adjust for sidebar width */
    right: 2rem;
    z-index: 999;
    background: white;
    padding-top: 0.5rem;
}

.main .block-container {
    padding-bottom: 6rem;
}
</style>
""", unsafe_allow_html=True)

def load_ingested_files() -> list[dict]:
    try:
        return list_ingested_files()
    except RuntimeError as exc:
        st.warning(str(exc))
        return []


def load_doc_versions() -> list[dict]:
    try:
        return get_all_versions()
    except RuntimeError as exc:
        st.warning(str(exc))
        return []


def auto_ingest_samples():
    sample_dir = Path("sample_docs")
    if not sample_dir.exists():
        return
    already_ingested = {f["filename"] for f in load_ingested_files()}
    for pdf_path in sample_dir.glob("*.pdf"):
        if pdf_path.name not in already_ingested:
            with st.spinner(f"Loading sample document: {pdf_path.name}..."):
                try:
                    ingest_pdf(pdf_path.read_bytes(), filename=pdf_path.name)
                except RuntimeError as exc:
                    st.warning(str(exc))
                    return

auto_ingest_samples()

tab_rag, tab_docs = st.tabs(["RAG Chat", "Documentation"])


# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — RAG CHAT
# ═══════════════════════════════════════════════════════════════════════
with tab_rag:
    st.header("RAG Chat")
    st.caption("Answers come only from your uploaded documents.")

    # Sidebar
    with st.sidebar:
        st.header("Documents")

        ingested = load_ingested_files()

        if ingested:
            st.success(f"{len(ingested)} document(s) ready")
        else:
            st.info("No documents loaded. Upload a PDF below.")

        st.divider()

        uploaded_file = st.file_uploader(
            "Upload PDF",
            type=["pdf"]
        )

        if uploaded_file:
            ingested_names = {f["filename"] for f in ingested}

            if uploaded_file.name not in ingested_names:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        result = ingest_pdf(
                            uploaded_file.read(),
                            filename=uploaded_file.name
                        )

                        st.success(
                            f"✓ Ingested {result['chunks_indexed']} chunks"
                        )

                        st.rerun()

                    except RuntimeError as exc:
                        st.error(str(exc))
            else:
                st.info(
                    f"{uploaded_file.name} is already loaded."
                )

    # Initialize chat history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": "Upload a PDF and ask me questions about it."
            }
        ]

    # Display chat history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if (
                message["role"] == "assistant"
                and message.get("sources")
            ):
                with st.expander("Sources used"):
                    for source in message["sources"]:
                        st.caption(source)

    # Chat input
    if prompt := st.chat_input(
        "Ask a question about your documents..."
    ):

        available_files = load_ingested_files()

        if not available_files:
            st.warning(
                "Please upload a PDF using the sidebar first."
            )
            st.stop()

        # User message
        st.session_state.chat_messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant response
        with st.chat_message("assistant"):

            message_placeholder = st.empty()
            full_answer = ""

            with st.spinner("Searching documents..."):
                try:
                    chunks = retrieve_similar_chunks(
                        prompt,
                        top_k=8
                    )
                except RuntimeError as exc:
                    st.error(str(exc))
                    st.stop()

            if not chunks:

                full_answer = (
                    "No relevant content found. "
                    "Try rephrasing your question."
                )

                message_placeholder.markdown(
                    full_answer
                )

                sources = []

            else:

                for token in stream_answer(
                    prompt,
                    chunks
                ):
                    full_answer += token
                    message_placeholder.markdown(
                        full_answer + "▌"
                    )

                message_placeholder.markdown(
                    full_answer
                )

                sources = get_sources_display(
                    chunks
                )

                with st.expander(
                    "Sources used"
                ):
                    for source in sources:
                        st.caption(source)

        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": full_answer,
                "sources": sources
            }
        )


# ═══════════════════════════════════════════════════════════════════════
# TAB 2 — DOCUMENTATION
# ═══════════════════════════════════════════════════════════════════════
with tab_docs:
    st.header("Auto-Updated Documentation")
    st.caption("Maintained by two LangGraph agents. Every push to app/ triggers an update.")

    # ─────────────────────────────────────────────
    # Trigger section FIRST
    # ─────────────────────────────────────────────
    st.subheader("Trigger Documentation Update")
    st.caption("Runs the Modifier and Reviewer agents right now without needing a git push.")

    if st.button("Run Agents Now", type="primary"):
        with st.status("Running agents...", expanded=True) as run_status:
            st.write("Starting Modifier agent...")

            result = subprocess.run(
                [
                    sys.executable, "-m", "agents.doc_updater",
                    "Manual trigger from Streamlit UI.",
                    "manual",
                    "manual"
                ],
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).parent.parent)
            )

            if result.returncode == 0:
                st.write("Agents completed successfully.")
                run_status.update(
                    label="Documentation updated!",
                    state="complete"
                )
                st.success(
                    "New version created. Select it from the dropdown below."
                )
                st.caption("Full agent output")
                st.code(result.stdout)
                st.rerun()
            else:
                run_status.update(
                    label="Something went wrong.",
                    state="error"
                )
                st.error(
                    "Agent run failed. Check the error below."
                )
                st.caption("Error details")
                st.code(result.stderr)

    st.divider()

    # ─────────────────────────────────────────────
    # Version dropdown SECOND
    # ─────────────────────────────────────────────
    versions = load_doc_versions()

    if not versions:
        st.info(
            "No documentation yet. Click 'Run Agents Now' above to generate the first version."
        )

    else:
        version_labels = {
            f"v{v['version']}  ·  {v['created_at'][:10]}  ·  {v['trigger_source']}": v
            for v in versions
        }

        selected_label = st.selectbox(
            "Select version to view",
            list(version_labels.keys())
        )

        selected = version_labels[selected_label]

        col_docs, col_logs = st.columns([3, 2])

        with col_docs:
            st.subheader(
                f"Documentation — v{selected['version']}"
            )

            sha = (
                selected.get("commit_sha") or "manual"
            )[:8]

            st.caption(
                f"Commit: `{sha}` | Triggered by: {selected['trigger_source']}"
            )

            with st.spinner("Loading documentation..."):
                try:
                    doc_content = get_version_doc(
                        selected["version"]
                    )
                except RuntimeError as exc:
                    st.error(str(exc))
                    doc_content = ""

            st.markdown(doc_content)

        with col_logs:
            st.subheader("Agent Thinking")

            try:
                logs = get_agent_logs(
                    selected["id"]
                )
            except RuntimeError as exc:
                st.error(str(exc))
                logs = []

            if not logs:
                st.info(
                    "No agent logs for this version."
                )
            else:
                for log in logs:
                    is_modifier = (
                        log["agent_name"] == "modifier"
                    )

                    color = (
                        "blue"
                        if is_modifier
                        else "green"
                    )

                    with st.expander(
                        f"{log['agent_name'].capitalize()} — Iteration {log['iteration']}",
                        expanded=True
                    ):
                        st.markdown(
                            f":{color}[{log['message']}]"
                        )

                        st.caption(
                            log["created_at"][:19]
                            .replace("T", " ")
                        )