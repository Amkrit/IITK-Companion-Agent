# rag_engine.py (Final Optimized Version)

from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Imports for running a model locally
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
import torch
import os

DB_FAISS_PATH = "vectorstore/db_faiss"
# The ID for your fine-tuned model on the Hub
FINETUNED_MODEL_ID = "IITKProject/iitk-companion-flan-t5-base"

def get_rag_chain():
    """
    Loads the fine-tuned model and sets up the robust RAG chain.
    """
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local(
        DB_FAISS_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    print(f"✅ Loading your fine-tuned model: {FINETUNED_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(FINETUNED_MODEL_ID)
    print("✅ Fine-tuned model and tokenizer loaded successfully.")

    # FIX: Pass generation parameters directly to the pipeline
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        temperature=0.1
    )
    
    llm = HuggingFacePipeline(pipeline=pipe)

    # FIX: Use "map_reduce" to handle long documents and avoid warnings
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce", 
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    return qa_chain

# ... (rest of the file remains the same) ...