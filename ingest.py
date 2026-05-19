import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

def get_line_number(text, char_offset):
    """Return the 1-based line number at a given character offset in text."""
    return text[:char_offset].count("\n") + 1

def split_with_line_numbers(splitter, raw_text, metadata):
    """
    Split raw_text using the given splitter, then find each chunk's position
    in the original source to attach start_line and end_line metadata.
    Uses the stripped chunk for searching so leading whitespace never breaks find().
    """
    chunks = splitter.split_text(raw_text)
    documents = []
    search_start = 0  # Cursor so we don't re-match earlier occurrences

    for chunk in chunks:
        # Strip leading whitespace for the search so indented chunks are found correctly
        stripped = chunk.lstrip()
        idx = raw_text.find(stripped, search_start)

        if idx == -1:
            # Last resort: search from the beginning in case cursor overshot
            idx = raw_text.find(stripped)

        if idx == -1:
            start_line = "?"
            end_line = "?"
        else:
            start_line = get_line_number(raw_text, idx)
            end_line = get_line_number(raw_text, idx + len(stripped) - 1)
            search_start = idx + len(stripped)

        doc_metadata = {**metadata, "start_line": start_line, "end_line": end_line}
        documents.append(Document(page_content=chunk, metadata=doc_metadata))

    return documents

def create_vector_db():
    code_path = "./my_code"

    # Define language-specific splitters using LangChain's built-in tokens
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=1000, chunk_overlap=100
    )
    cpp_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.CPP, chunk_size=1000, chunk_overlap=100
    )

    final_chunks = []
    file_count = 0

    # 1. Walk through the code directory
    for root, _, files in os.walk(code_path):
        for file in files:
            file_path = os.path.join(root, file)

            if file.endswith(('.py', '.c', '.h', '.cpp')):
                try:
                    # Load the raw text so we can track character positions
                    loader = TextLoader(file_path, encoding='utf-8')
                    loaded_docs = loader.load()  # List of 1 Document
                    raw_text = loaded_docs[0].page_content
                    base_metadata = loaded_docs[0].metadata  # Has 'source' key

                    # 2. Split with line number tracking
                    if file.endswith('.py'):
                        chunks = split_with_line_numbers(python_splitter, raw_text, base_metadata)
                    else:  # .c, .h, or .cpp
                        chunks = split_with_line_numbers(cpp_splitter, raw_text, base_metadata)

                    final_chunks.extend(chunks)
                    file_count += 1
                    print(f"Indexed {file}: split into {len(chunks)} smart chunks.")

                except Exception as e:
                    print(f"Skipping {file} due to error: {e}")

    if not final_chunks:
        print("No valid code files found in /my_code! Add some files and try again.")
        return

    # 3. Create Embeddings
    print("\nInitializing local embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Save to FAISS
    print(f"Processing a total of {len(final_chunks)} chunks from {file_count} files...")
    vector_db = FAISS.from_documents(final_chunks, embeddings)
    vector_db.save_local("faiss_code_index")
    print("Success! 'faiss_code_index' updated with line-aware chunking. You're ready to test.")

if __name__ == "__main__":
    create_vector_db()