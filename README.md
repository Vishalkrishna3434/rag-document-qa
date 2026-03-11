# 🤖 Document Q&A System (RAG)

A retrieval-augmented generation system that answers 
questions from uploaded PDF documents using semantic search and LLM inference.

![Status](https://img.shields.io/badge/Status-In%20Development-yellow)
![Stack](https://img.shields.io/badge/Stack-Python%20%7C%20LangChain-blue)

## ✨ Features
- Upload PDF documents
- Automatic text chunking and embedding generation
- Semantic search using FAISS vector indexing
- Context-aware answers using LLM inference
- Simple CLI interface

## 🛠 Tech Stack
- Python
- LangChain
- FAISS
- OpenAI API / HuggingFace

## ⚙️ Setup
```bash
pip install langchain faiss-cpu pypdf openai
python app.py
```

## 🔍 How It Works
1. Upload PDF → text extracted and split into chunks
2. Chunks → converted to embeddings
3. Embeddings → stored in FAISS index
4. Query → semantically searched → top chunks retrieved
5. LLM → generates answer from retrieved context
