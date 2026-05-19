import streamlit as st
import os
import subprocess # Allows us to trigger ingest.py automatically behind the scenes
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

# Page setup
st.set_page_config(page_title="SourceCode-Sensei", layout="wide")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")
st.markdown('<div class="main-title">SOURCECODE-SENSEI_ //</div>', unsafe_allow_html=True)

# INITIALIZE STATES IMMEDIATELY AT THE TOP TO PREVENT ATTRIBUTEERROR
if "all_chats" not in st.session_state:
    st.session_state.all_chats = {}  # Format: {"Session Archive #1": [messages]}
if "current_chat_title" not in st.session_state:
    st.session_state.current_chat_title = "Active Session"
if "messages" not in st.session_state:
    st.session_state.messages = []

# 1. Verification Check
if not os.path.exists("faiss_code_index"):
    st.error("Error: 'faiss_code_index' not found. Please run 'python ingest.py' first!")
    st.stop()

# 2. Load the System
@st.cache_resource
def load_system():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_code_index", embeddings, allow_dangerous_deserialization=True)
    llm = OllamaLLM(model="qwen2.5-coder:1.5b")

    # Initialize BM25 (Keyword Matcher) — k bumped to 4 for better coverage
    all_docs = list(db.docstore._dict.values())
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = 4

    # Relevance threshold — FAISS L2 distance, lower = more similar; drop anything above this
    RELEVANCE_THRESHOLD = 1.2

    # Custom Hybrid Retriever Function
    # Returns list of dicts: {doc, score, start_line, end_line, source}
    def custom_hybrid_retriever(query):
        # BM25: keyword-precise, include all k=4 results
        keyword_docs = bm25_retriever.invoke(query)

        # FAISS: scored search so we can filter weak matches
        semantic_results = db.similarity_search_with_score(query, k=4)

        combined = []
        seen_contents = set()

        # Process BM25 results
        for doc in keyword_docs:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                start_line = doc.metadata.get("start_line", "?")
                lines = doc.page_content.splitlines()
                end_line = doc.metadata.get("end_line", len(lines) if start_line == "?" else start_line + len(lines) - 1)
                combined.append({
                    "doc": doc,
                    "score": None,
                    "start_line": start_line,
                    "end_line": end_line,
                    "source": "bm25"
                })

        # Process FAISS results — filter weak matches by threshold
        for doc, score in semantic_results:
            if score > RELEVANCE_THRESHOLD:
                continue
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                start_line = doc.metadata.get("start_line", "?")
                lines = doc.page_content.splitlines()
                end_line = doc.metadata.get("end_line", len(lines) if start_line == "?" else start_line + len(lines) - 1)
                combined.append({
                    "doc": doc,
                    "score": round(float(score), 4),
                    "start_line": start_line,
                    "end_line": end_line,
                    "source": "faiss"
                })

        return combined

    return db, llm, custom_hybrid_retriever

db, llm, retriever = load_system()

# 3. Sidebar Codebase Management Layout Configuration
with st.sidebar:

    # ====================================================
    # 📍 ORDER POSITION 1: Codebase Management Title (pinned to top, no gap)
    # ====================================================
    st.markdown(
        "<h2 style='margin-top: 0px; margin-bottom: 0px; padding-top: 0px; font-size: 1.6rem; font-weight: bold;'>Codebase Management</h2>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    # ====================================================
    # 📍 ORDER POSITION 2: Session Controls & Recent Logs
    # ====================================================
    st.markdown("<h3 style='margin-top: 0px; margin-bottom: 10px; font-size: 1.2rem; font-weight: bold;'>Session Controls</h3>", unsafe_allow_html=True)

    if st.button("+ Start New Chat", key="sidebar_start_new_chat_btn"):
        if st.session_state.messages:
            # BACKGROUND SMART AUTO-NAMING
            first_user_prompt = "New Session"
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    first_user_prompt = msg["content"]
                    break

            summary_prompt = (
                f"You are a utility sub-routine. Summarize the following user programming inquiry into a crisp, "
                f"2 to 3 word title for a chat history log. Do not include quotes, punctuation, or explanations.\n"
                f"Inquiry: {first_user_prompt}\n"
                f"Summary:"
            )

            try:
                raw_summary = llm.invoke(summary_prompt).strip()
                clean_summary = raw_summary.replace('"', '').replace("'", "").replace("`", "")
                archive_title = f"📁 {clean_summary} ({len(st.session_state.messages)} msgs)"
            except Exception:
                session_num = len(st.session_state.all_chats) + 1
                archive_title = f"📁 Session Archive #{session_num} ({len(st.session_state.messages)} msgs)"

            st.session_state.all_chats[archive_title] = st.session_state.messages

        st.session_state.messages = []
        st.rerun()

    if st.session_state.all_chats:
        st.write("")
        st.markdown("<span style='color:#8b949e; font-size:0.85rem; font-weight: bold;'>RECENT CHAT LOGS:</span>", unsafe_allow_html=True)
        for title, archived_msgs in st.session_state.all_chats.items():
            if st.button(title, key=f"load_{title}"):
                st.session_state.messages = archived_msgs
                st.rerun()

    st.markdown("---")

    # ====================================================
    # 📍 ORDER POSITION 3: Upload Code Files & Indexed Elements
    # ====================================================
    st.markdown("<h3 style='margin-top: 0px; margin-bottom: 10px; font-size: 1.2rem; font-weight: bold;'>Upload Code Files</h3>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Add new files to Sensei's memory:",
        type=["py", "c", "h", "cpp", "js", "ts", "java", "go"],
        accept_multiple_files=True,
        key="forced_sidebar_uploader"
    )

    if uploaded_files:
        if st.button("🚀 Save & Re-index Codebase", key="forced_reindex_btn"):
            with st.spinner("Saving files and updating memory index..."):
                os.makedirs("my_code", exist_ok=True)

                for uploaded_file in uploaded_files:
                    file_path = os.path.join("my_code", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"Saved: `{uploaded_file.name}`")

                venv_python = os.path.join("venv", "Scripts", "python.exe")
                result = subprocess.run([venv_python, "ingest.py"], capture_output=True, text=True)

                if result.returncode == 0:
                    st.success("🎉 Database successfully re-indexed!")
                    st.cache_resource.clear()
                    st.rerun()
                else:
                    st.error(f"Ingestion failed: {result.stderr}")

    # Telemetry Engine and Delete File UI list
    try:
        total_chunks = len(db.docstore._dict)
        unique_files = set()
        for doc_id, doc in db.docstore._dict.items():
            file_path = doc.metadata.get('source') or doc.metadata.get('Source') or 'Unknown'
            unique_files.add(os.path.basename(file_path))

        st.write("")
        st.metric(label="Total Tracked Files", value=len(unique_files))
        st.metric(label="Total Smart Code Chunks", value=total_chunks)

        # Context Window Progress Gauge
        MAX_CONTEXT_TOKENS = 32768  # qwen2.5-coder:1.5b max context
        total_chars = sum(len(doc.page_content) for doc in db.docstore._dict.values())
        estimated_tokens = total_chars // 4
        usage_pct = min(estimated_tokens / MAX_CONTEXT_TOKENS, 1.0)

        if usage_pct < 0.5:
            gauge_color = "normal"
        elif usage_pct < 0.8:
            gauge_color = "off"
        else:
            gauge_color = "inverse"

        st.write("")
        st.markdown("<span style='color:#8b949e; font-size:0.85rem; font-weight: bold;'>CONTEXT WINDOW USAGE:</span>", unsafe_allow_html=True)
        st.progress(usage_pct)
        st.caption(f"~{estimated_tokens:,} / {MAX_CONTEXT_TOKENS:,} tokens used ({usage_pct*100:.1f}%) \n\n {'⚠️ Getting full' if usage_pct > 0.8 else '✅ Healthy'}")

        if unique_files:
            st.markdown("<h4 style='margin-top: 10px; font-size: 1.0rem; font-weight: bold;'>Indexed Files:</h4>", unsafe_allow_html=True)
            for f in sorted(unique_files):
                col_file, col_del = st.columns([0.75, 0.25])

                with col_file:
                    st.markdown(f"🧬 `{f}`")

                with col_del:
                    if st.button("🗑️", key=f"del_forced_{f}"):
                        file_to_delete = os.path.join("my_code", f)
                        if os.path.exists(file_to_delete):
                            os.remove(file_to_delete)
                            st.sidebar.warning(f"Removed: {f}")

                            with st.spinner("Purging data streams..."):
                                venv_python = os.path.join("venv", "Scripts", "python.exe")
                                result = subprocess.run([venv_python, "ingest.py"], capture_output=True, text=True)

                                if result.returncode == 0:
                                    st.cache_resource.clear()
                                    st.rerun()
                                else:
                                    st.error("Failed to re-index after file deletion.")

            st.write("")
            if "confirm_clear_all" not in st.session_state:
                st.session_state.confirm_clear_all = False

            if not st.session_state.confirm_clear_all:
                if st.button("🗑️ Clear All Files", key="clear_all_btn", use_container_width=True):
                    st.session_state.confirm_clear_all = True
                    st.rerun()
            else:
                st.warning("⚠️ This will delete all indexed files. Are you sure?")
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("✅ Yes", key="confirm_yes_btn", use_container_width=True):
                        deleted_count = 0
                        for f in list(unique_files):
                            file_to_delete = os.path.join("my_code", f)
                            if os.path.exists(file_to_delete):
                                os.remove(file_to_delete)
                                deleted_count += 1
                        with st.spinner("Purging all data streams..."):
                            venv_python = os.path.join("venv", "Scripts", "python.exe")
                            result = subprocess.run([venv_python, "ingest.py"], capture_output=True, text=True)
                            if result.returncode == 0:
                                st.session_state.confirm_clear_all = False
                                st.cache_resource.clear()
                                st.rerun()
                            else:
                                st.error("Failed to re-index after clearing files.")
                with col_no:
                    if st.button("❌ No", key="confirm_no_btn", use_container_width=True):
                        st.session_state.confirm_clear_all = False
                        st.rerun()

    except Exception as e:
        st.warning("Could not read index statistics.")

    st.markdown("---")

    # ====================================================
    # 📍 ORDER POSITION 4: System Settings Pinned to Bottom
    # ====================================================
    st.markdown("<h3 style='margin-top: 0px; margin-bottom: 10px; font-size: 1.2rem; font-weight: bold;'>System Settings</h3>", unsafe_allow_html=True)
    st.info("Model: `qwen2.5-coder:1.5b` \n\n Embeddings: `all-MiniLM-L6-v2` \n\n Search: `Custom Hybrid (FAISS + BM25)` \n\n Supported: `py · c · h · cpp · js · ts · java · go`")

st.info("Ask me anything about the code files listed in the sidebar.")

# 4. Chat Logic & History Processing Workspace
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# User Input
if prompt := st.chat_input("Ex: How does the login logic work?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your code..."):
            results = retriever(prompt)

            source_chunks = []
            context_blocks = []

            for r in results:
                doc = r["doc"]
                start_line = r["start_line"]
                end_line = r["end_line"]
                score = r["score"]
                src_tag = r["source"]

                file_path = doc.metadata.get('source') or doc.metadata.get('Source') or 'Unknown File'
                clean_name = os.path.basename(file_path)

                source_chunks.append((clean_name, start_line, end_line, score, src_tag))
                context_blocks.append(
                    f"--- From File: {clean_name} (Lines {start_line}–{end_line}) ---\n{doc.page_content}"
                )

            context = "\n\n".join(context_blocks)

            if source_chunks:
                with st.expander("📍 Referenced Code Chunks", expanded=False):
                    for clean_name, start_line, end_line, score, src_tag in source_chunks:
                        score_str = f" · score: {score}" if score is not None else ""
                        st.markdown(f"- `{clean_name}` — Lines **{start_line}–{end_line}** `[{src_tag}{score_str}]`")

            chat_history_str = ""
            for msg in st.session_state.messages[:-1]:
                role_label = "User" if msg["role"] == "user" else "Assistant"
                chat_history_str += f"{role_label}: {msg['content']}\n"

            full_prompt = (
                f"You are SourceCode-Sensei, an expert programming assistant.\n"
                f"Your goal is to provide clean, production-grade code explanations and solutions.\n\n"
                f"CRITICAL FORMATTING & TRUTH RULES:\n"
                f"1. Always wrap any code snippets in standard Markdown code blocks.\n"
                f"2. Explicitly specify the language identifier after the initial backticks (e.g., ```python or ```c).\n"
                f"3. Never leave code blocks unlabelled.\n"
                f"4. Keep explanations concise, scannable, and developer-focused.\n"
                f"5. TRUTH CONSTRAINT: Carefully verify if the provided Codebase Context contains the actual logic requested. If the context does not contain any code related to the user's question, state clearly that you cannot find that file or feature in the current codebase instead of forcing an explanation or inventing connections.\n\n"
                f"--- Codebase Context ---\n{context}\n\n"
                f"--- Conversation History ---\n{chat_history_str}\n"
                f"Current User Question: {prompt}\n"
                f"Assistant:"
            )

            response = llm.invoke(full_prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})