import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_db():
    code_path = "./my_code"
    
    # Define our language-specific splitters using LangChain's built-in tokens
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
                    # Load the individual file
                    loader = TextLoader(file_path, encoding='utf-8')
                    single_doc = loader.load() # This creates a list containing exactly 1 Document
                    
                    # 2. Apply the language-aware splitter based on the file extension
                    if file.endswith('.py'):
                        chunks = python_splitter.split_documents(single_doc)
                    else: # .c, .h, or .cpp
                        chunks = cpp_splitter.split_documents(single_doc)
                    
                    final_chunks.extend(chunks)
                    file_count += 1
                    print(f"Indexed {file}: split into {len(chunks)} smart chunks.")
                    
                except Exception as e:
                    print(f"Skipping {file} due to error: {e}")

    if not final_chunks:
        print("No valid code files found in /my_code! Add some files and try again.")
        return

    # 3. Create Embeddings (Local Math)
    print("\nInitializing local embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Save everything to FAISS (The Local Database)
    print(f"Processing a total of {len(final_chunks)} chunks from {file_count} files...")
    vector_db = FAISS.from_documents(final_chunks, embeddings)
    vector_db.save_local("faiss_code_index")
    print("Success! 'faiss_code_index' updated with multi-file awareness. You're ready to test.")

if __name__ == "__main__":
    create_vector_db()