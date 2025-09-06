# build_rag_index.py

import os
from langchain.document_loaders import TextLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS

# Define the paths to our data and where to save the index
DATA_DIR = os.getcwd() # Assumes data files are in the same folder
MANUAL_PATH = os.path.join(DATA_DIR, "ug_manual.txt")
COURSES_PATH = os.path.join(DATA_DIR, "ae_courses.json")
DB_FAISS_PATH = "vectorstore/db_faiss"

def create_vector_db():
    """
    Creates a FAISS vector database from the UG Manual and course data.
    """
    # --- 1. Load Documents ---
    print("Loading documents...")
    manual_loader = TextLoader(MANUAL_PATH, encoding='utf-8')
    
    # JSONLoader needs a jq schema to know what text to extract.
    # We're telling it to combine the code, title, and description for each course.
    courses_loader = JSONLoader(
        file_path=COURSES_PATH,
        jq_schema='.[].code + ": " + .[].title + " - " + .[].description',
        text_content=False # Indicates jq_schema is used
    )
    
    documents = manual_loader.load() + courses_loader.load()
    print(f"Loaded {len(documents)} document sections.")

    # --- 2. Split Documents into Chunks ---
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")

    # --- 3. Create Embeddings ---
    print("Creating embeddings... (This may take a few minutes)")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # --- 4. Create and Save the Vector Store ---
    print("Creating and saving the FAISS vector store...")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(DB_FAISS_PATH)
    print(f"Vector store created and saved at '{DB_FAISS_PATH}'.")

if __name__ == "__main__":
    create_vector_db()