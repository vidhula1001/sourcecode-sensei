import streamlit as st
import os
import subprocess # Allows us to trigger ingest.py automatically behind the scenes
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever

# Page setup
st.set_page_config(page_title="SourceCode-Sensei", layout="wide")
#st.title("SourceCode-Sensei 🧠")

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
    
    # Initialize BM25 (Keyword Matcher)
    all_docs = list(db.docstore._dict.values())
    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_retriever.k = 2  
    
    # Custom Hybrid Retriever Function
    def custom_hybrid_retriever(query):
        keyword_docs = bm25_retriever.invoke(query)
        semantic_docs = db.similarity_search(query, k=2)
        
        combined_docs = []
        seen_contents = set()
        for doc in (keyword_docs + semantic_docs):
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                combined_docs.append(doc)
        return combined_docs

    return db, llm, custom_hybrid_retriever

db, llm, retriever = load_system()

# 3. Sidebar Codebase Management & File Uploads (Ultra-Compact Reverted Layout)
with st.sidebar:
    # POSITION 1: The Title (Custom HTML to kill the top padding)
    st.markdown("<h2 style='margin-top: -20px; margin-bottom: 10px; font-size: 1.6rem; font-weight: bold;'>Codebase Management</h2>", unsafe_allow_html=True)
    
    # POSITION 2: The Upload Section (Directly underneath title, no panel wrappers)
    st.markdown("<h4 style='margin-top: 5px; margin-bottom: 5px; font-size: 1.1rem; font-weight: 600;'>Upload Code Files</h4>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Add new files to Sensei's memory:", 
        type=["py", "c", "h", "cpp"], 
        accept_multiple_files=True,
        label_visibility="collapsed" # Hides duplicate small label text to save vertical space
    )
    
    if uploaded_files:
        st.write("")
        if st.button("🚀 Save & Re-index Codebase"):
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
                    
    st.markdown("<hr style='margin-top: 10px; margin-bottom: 10px;' />", unsafe_allow_html=True)
    
    # POSITION 3: Session Controls
    st.markdown("<h4 style='margin-top: 0px; margin-bottom: 10px; font-size: 1.1rem; font-weight: 600;'>Session Controls</h4>", unsafe_allow_html=True)
    
    if st.button("+ Start New Chat"):
        if st.session_state.messages:
            # 🤖 BACKGROUND AUTO-NAMING RETRIEVAL ENGINE
            # Find the very first message the user typed in this chat session
            first_user_prompt = "New Session"
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    first_user_prompt = msg["content"]
                    break
            
            # Construct a small background utility prompt for Qwen
            summary_prompt = (
                f"You are a utility sub-routine. Summarize the following user programming inquiry into a crisp, "
                f"2 to 3 word title for a chat history log. Do not include quotes, punctuation, or explanations.\n"
                f"Inquiry: {first_user_prompt}\n"
                f"Summary:"
            )
            
            try:
                # Call Qwen quickly in the background to name the chat semantically
                raw_summary = llm.invoke(summary_prompt).strip()
                # Clean up any trailing quotes or backticks the model might add
                clean_summary = raw_summary.replace('"', '').replace("'", "").replace("`", "")
                archive_title = f"📁 {clean_summary} ({len(st.session_state.messages)} msgs)"
            except Exception:
                # Fallback if the local model lags or hits a timeout
                session_num = len(st.session_state.all_chats) + 1
                archive_title = f"📁 Session Archive #{session_num} ({len(st.session_state.messages)} msgs)"
                
            # Archive the session under its brand-new semantic title
            st.session_state.all_chats[archive_title] = st.session_state.messages
            
        # Flush active screen workspace clear
        st.session_state.messages = []
        st.rerun()
            
        # Flush the active screen workspace clear
        st.session_state.messages = []
        st.rerun()

    # Automatically map out and print your archived chat history log buttons
    if st.session_state.all_chats:
        st.write("")
        st.markdown("<span style='color:#8b949e; font-size:0.85rem;'>RECENT CHAT LOGS:</span>", unsafe_allow_html=True)
        for title, archived_msgs in st.session_state.all_chats.items():
            if st.button(title, key=f"load_{title}"):
                st.session_state.messages = archived_msgs
                st.rerun()
                
    st.markdown("<hr style='margin-top: 10px; margin-bottom: 10px;' />", unsafe_allow_html=True)

    # POSITION 4: Telemetry Stats & File List
    try:
        total_chunks = len(db.docstore._dict)
        unique_files = set()
        for doc_id, doc in db.docstore._dict.items():
            file_path = doc.metadata.get('source', 'Unknown')
            unique_files.add(os.path.basename(file_path))
            
        st.metric(label="Total Tracked Files", value=len(unique_files))
        st.metric(label="Total Smart Code Chunks", value=total_chunks)
        
        st.markdown("<h4 style='margin-top: 10px; margin-bottom: 5px; font-size: 1.1rem; font-weight: 600;'>Indexed Files:</h4>", unsafe_allow_html=True)
        for f in sorted(unique_files):
            col_file, col_del = st.columns([0.75, 0.25])
            
            with col_file:
                st.markdown(f"🧬 `{f}`")
                
            with col_del:
                if st.button("🗑️", key=f"del_{f}"):
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
    except Exception as e:
        st.warning("Could not read index statistics.")
        
    # POSITION 5: System Settings at the bottom
    st.markdown("<hr style='margin-top: 10px; margin-bottom: 10px;' />", unsafe_allow_html=True)
    st.markdown("<h4 style='margin-top: 0px; margin-bottom: 5px; font-size: 1.1rem; font-weight: 600;'>System Settings</h4>", unsafe_allow_html=True)
    st.info("Model: `qwen2.5-coder:1.5b` \n\n Embeddings: `all-MiniLM-L6-v2` \n\n Search: `Custom Hybrid (FAISS + BM25)`")

st.info("Ask me anything about the code files listed in the sidebar.")

# 4. Chat Logic & History Processing
# Show history
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
            docs = retriever(prompt)
            
            sources = set()
            context_blocks = []
            for d in docs:
                file_path = d.metadata.get('source', 'Unknown File')
                clean_name = os.path.basename(file_path)
                sources.add(clean_name)
                context_blocks.append(f"--- From File: {clean_name} ---\n{d.page_content}")
            
            context = "\n\n".join(context_blocks)
            
            if sources:
                with st.expander("📍 Referenced Code Files", expanded=False):
                    for src in sources:
                        st.markdown(f"- `{src}`")
            
            chat_history_str = ""
            for msg in st.session_state.messages[:-1]:
                role_label = "User" if msg["role"] == "user" else "Assistant"
                chat_history_str += f"{role_label}: {msg['content']}\n"
            
            full_prompt = (
                f"You are SourceCode-Sensei, an expert programming assistant.\n"
                f"Your goal is to provide clean, production-grade code explanations and solutions.\n\n"
                f"CRITICAL FORMATTING RULES:\n"
                f"1. Always wrap any code snippets in standard Markdown code blocks.\n"
                f"2. Explicitly specify the language identifier after the initial backticks (e.g., ```python or ```c).\n"
                f"3. Never leave code blocks unlabelled.\n"
                f"4. Keep explanations concise, scannable, and developer-focused.\n\n"
                f"--- Codebase Context ---\n{context}\n\n"
                f"--- Conversation History ---\n{chat_history_str}\n"
                f"Current User Question: {prompt}\n"
                f"Assistant:"
            )
            
            response = llm.invoke(full_prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})