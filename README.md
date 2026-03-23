# Chat with PDF using Gemini 💁

A RAG (Retrieval-Augmented Generation) app that lets you upload PDF files and ask natural language questions about their content using Google Gemini.

---

## How It Works

PDF → Extract Text → Split into Chunks → Generate Embeddings → Store in FAISS → User asks Question → Retrieve Relevant Chunks → Gemini generates Answer

---

## Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- Google Gemini API
- PyPDF2

---

## Project Structure
```
rag-document-qa/
├── app.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/Vishalkrishna3434/rag-document-qa.git
cd rag-document-qa
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your API key

Create a `.env` file in the project folder:
```
GOOGLE_API_KEY=your_api_key_here
```

Get your free API key from: https://aistudio.google.com/app/apikey

### 4. Run the app
```bash
streamlit run app.py
```

---

## Usage

1. Upload one or more PDF files from the sidebar
2. Click **Submit & Process** and wait for "Done"
3. Type your question in the text box
4. Get your answer

---

## Models Used

| Purpose | Model |
|---|---|
| Embeddings | `models/gemini-embedding-001` |
| Answer Generation | `gemini-2.5-flash` |

---

## Requirements

- Python 3.9 – 3.12
- Google Gemini API key (free at AI Studio)

---

## Example Use Cases

- Research paper Q&A
- Notes summarization
- Document search
- Study assistant
- Legal / technical document analysis

---

## Security

- Never commit your `.env` file
- `faiss_index/` is generated locally — do not push to GitHub
- If your API key is accidentally pushed, regenerate it immediately at AI Studio

---

## Known Limitations

- Requires internet connection
- Large PDFs may take a few seconds to process
- FAISS index resets if the folder is deleted

---

## Future Improvements

- Chat history across questions
- Support for DOCX and TXT files
- Cloud deployment
- Persistent vector database