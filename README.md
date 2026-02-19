# IITK Companion Agent 🎓🤖

An AI-powered academic assistant built for IIT Kanpur students using Retrieval-Augmented Generation (RAG).

---

## 📌 Overview

IITK Companion Agent helps students:

- Answer academic queries from UG manuals  
- Retrieve course information  
- Provide context-aware responses using RAG  
- Act as an AI academic assistant  

---

## 🏗️ How It Works

1. Parse UG manual and course data  
2. Build embeddings using RAG  
3. Retrieve relevant context  
4. Generate grounded AI responses  

---

## 📂 Project Structure

- `agent_app.py` – Main assistant app  
- `build_rag_index.py` – Builds embedding index  
- `rag_engine.py` – RAG logic  
- `parse_ug_manual.py` – Manual parsing  
- `tools.py` – Utility functions  
- `ae_courses.json` – Course data  

---

## ⚙️ Installation

```bash
git clone https://github.com/Amkrit/IITK-Companion-Agent.git
cd IITK-Companion-Agent
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
