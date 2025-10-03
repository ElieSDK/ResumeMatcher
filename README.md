# ResumeMatcher (RAG-Enabled)

**Repository structure**  

```
.
+-- data
|   +-- jds/                # Job description files (PDF, DOCX, TXT)
|   +-- resumes/            # Candidate resumes (PDF, DOCX, TXT)
+-- index                   # FAISS index files
|   +-- index.faiss
|   +-- index.pkl
|   +-- index_store.faiss
|   +-- index_store.pkl
+-- output
|   +-- console.txt         # Example output from model
+-- src
|   +-- main.py             # Main script
+-- LICENSE
+-- README.md
+-- requirements.txt
```

---

## **Project Overview**

**Job Matcher (RAG-Enabled)** is an AI-powered tool for **matching resumes to job descriptions**. It uses a **Retrieval-Augmented Generation (RAG) pipeline**:

1. Text from resumes and job descriptions is **extracted and cleaned** from PDF, DOCX, or TXT files.
2. Text is **converted into embeddings** using OpenAI’s embedding model.
3. Embeddings are stored in a **FAISS index** for fast similarity search.
4. When comparing a resume to a job description:
   - The system retrieves relevant context from the FAISS index.
   - Sends the resume, job description, and retrieved context to an **LLM (GPT-3.5)**.
   - Produces a **match score (0-100)** and **explanation bullets** for the match.
5. Users can interact with the system via **CLI** or **Streamlit web interface**.

**URL:** [https://resumematcher-rag.streamlit.app/](https://resumematcher-rag.streamlit.app/)   

---

## **Pipeline Diagram**

```mermaid
flowchart TD
    A[Resume + Job Description] --> B[Text Extraction & Cleaning]
    B --> C[Convert to Embeddings]
    C --> D[FAISS Index / Retriever]
    D --> E[Prompt Template + Context]
    E --> F[LLM (GPT-3.5)]
    F --> G[Output: Score + Explanation]
```

---

## **Features**

- Supports multiple file formats: PDF, DOCX, TXT.
- Generates structured JSON output with:
  - `score` → numeric match score (0–100)
  - `explanation` → key points justifying the score
- RAG-based retrieval from FAISS index for context-aware matching.
- CLI and web interface support.
- Clean temporary file handling for uploads.
- Ready for deployment on Streamlit.

---

## **Installation**

1. Clone the repository:

```bash
git clone https://github.com/yourusername/job-matcher.git
cd job-matcher
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

> **Requirements include:** `langchain`, `pdfplumber`, `python-docx`, `faiss`, `openai`, `streamlit`, `dotenv`

4. Set up your OpenAI API key in a `.env` file:

```
OPENAI_API_KEY=your_openai_api_key
```

---

## **Usage**

### **1. Index documents**

```bash
python src/main.py --action index --data-dir ./data --index-dir ./index
```

- Scans `data/resumes` and `data/jds`
- Extracts text and stores embeddings in the FAISS index.
- Saves the index to `index/`.

---

### **2. Match a resume to a job description (CLI)**

```bash
python src/main.py --action match --resume ./data/resumes/resume.pdf --jd ./data/jds/job.pdf --index-dir ./index
```

- Produces JSON output with `score` and `explanation`.  
- Example output is stored in `output/console.txt`.

---

### **3. Run the Streamlit web interface**

```bash
streamlit run src/main.py -- --action serve --index-dir ./index
```

- **URL:** [https://resumematcher-rag.streamlit.app/](https://resumematcher-rag.streamlit.app/)  
- Upload a resume and job description through the UI.  
- Click **Get Match Score** to see the match score and explanations.  

---

## **Output Example**

`output/console.txt` contains an example JSON:

```json
{
  "score": 85,
  "explanation": [
    "Candidate has strong Python skills matching job requirements",
    "Relevant experience in data analysis and machine learning",
    "No experience listed in cloud deployment required by job"
  ]
}
```

---

## **Folder Explanation**

- `data/` → contains input resumes and job descriptions.  
- `index/` → contains the FAISS index (`.faiss`, `.pkl`, `index_store`).  
- `output/` → stores example output from the CLI.  
- `src/` → contains the main script (`main.py`).  
- `LICENSE` → project license.  

---

## **License**

This project is licensed under the MIT License – see the `LICENSE` file for details.

---

## **Future Improvements**

- Support for more file formats (ODT, RTF).  
- Advanced scoring with weighted skills and experience.  
