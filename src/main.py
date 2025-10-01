import warnings
import os
import sys
import argparse
import pathlib
import tempfile
import json
from typing import List, Dict, Any, Optional

# Third-party libraries
import pdfplumber
import docx
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from dotenv import load_dotenv

# LangChain components
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import Runnable

import logging 
# Set the log level for pdfminer to ERROR, effectively silencing WARNING and INFO messages.
logging.getLogger("pdfminer").setLevel(logging.ERROR)

# Suppress PDFMiner warnings (used by pdfplumber)
warnings.filterwarnings("ignore", category=UserWarning, module="pdfminer")

# Streamlit setup
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# --- Configuration & Initialization ---
# Load environment variables
load_dotenv("/home/es/Desktop/code/.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize spaCy for NLP
# Use Optional[spacy.Language] for correct typing
NLP: Optional[spacy.Language] = spacy.load("en_core_web_sm")

# Define the LLM Prompt
MATCHING_PROMPT_TEMPLATE = """
You are an expert recruiter. Your task is to compare a candidate's resume against a job description.
The goal is to determine the suitability and quantify the match.
Return your complete response as a single JSON object.

JSON Schema:
{{
    "score": <int, 0-100, where 100 is a perfect match>,
    "explanation": <list of strings, key bullet points justifying the score, highlighting matches and gaps>
}}

---
Resume:
{resume}

---
Job Description:
{job}

---
Retrieved Context (from similar documents):
{context}
"""

# --- File Handling and Text Preprocessing ---

def extract_text(path: str) -> str:
    """Extracts text content from PDF, DOCX, or plain text files."""
    p = pathlib.Path(path)
    suffix = p.suffix.lower()

    try:
        if suffix == ".pdf":
            with pdfplumber.open(path) as pdf:
                # Use a more robust check for None pages
                return "\n".join([page.extract_text() or "" for page in pdf.pages])
        elif suffix in (".docx", ".doc"):
            doc = docx.Document(path)
            # Use doc.paragraphs if all text is needed, or a more specific structure
            return "\n".join(paragraph.text for paragraph in doc.paragraphs)
        else:
            # Assumes other files are plain text
            return p.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"Error extracting text from {path}: {e}")
        return ""

def clean_text(text: str) -> str:
    """Removes empty lines and strips whitespace from text lines."""
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())

def tokenize(text: str) -> List[str]:
    """Tokenizes and lemmatizes text, removing stopwords, using spaCy if available."""
    if not NLP:
        # Fallback to simple split if spaCy is unavailable
        return [t.lower() for t in text.split() if t.lower() not in STOP_WORDS and t.isalpha()]
    
    doc = NLP(text)
    # Use is_alpha to filter out punctuation/numbers
    return [t.lemma_.lower() for t in doc if not t.is_stop and t.is_alpha]

# --- LangChain Core Components ---

def get_embeddings() -> OpenAIEmbeddings:
    """Initializes and returns the OpenAI Embeddings model."""
    print("[Info] Initializing OpenAI embeddings...")
    return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

def index_documents(data_dir: str, index_dir: str):
    """
    Loads documents from the data directory, embeds them, and saves the FAISS index.
    """
    resumes_dir = pathlib.Path(data_dir) / "resumes"
    jds_dir = pathlib.Path(data_dir) / "jds"
    docs: List[Document] = []
    
    # List of valid file extensions
    valid_extensions = {".pdf", ".txt", ".docx", ".doc"}

    for folder, source in [(resumes_dir, "resume"), (jds_dir, "job")]:
        for f in folder.glob("**/*"):
            if f.suffix.lower() in valid_extensions:
                text = clean_text(extract_text(str(f)))
                if text.strip():
                    docs.append(Document(
                        page_content=text, 
                        metadata={"source": source, "filename": f.name}
                    ))

    if not docs:
        print(f"[Warning] No documents found in {data_dir}. Index not created.")
        return

    embeddings = get_embeddings()
    vs = FAISS.from_documents(docs, embeddings)
    os.makedirs(index_dir, exist_ok=True)
    vs.save_local(index_dir)
    print(f"[Success] Indexed {len(docs)} documents. Saved index to {index_dir}")

def retrieve(query: str, index_dir: str, k: int = 5) -> List[Document]:
    """
    Loads the FAISS index and retrieves top-k documents relevant to the query.
    """
    if not os.path.exists(index_dir):
        print(f"[Error] Index directory not found: {index_dir}")
        return []
        
    embeddings = get_embeddings()
    # Note: allow_dangerous_deserialization=True is necessary for loading FAISS locally
    vs = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    
    return vs.similarity_search(query, k=k)

def match(resume_path: str, jd_path: str, index_dir: str, k: int = 5) -> Dict[str, Any]:
    """
    Performs the RAG-based matching between a resume and a job description.
    """
    resume_text = clean_text(extract_text(resume_path))
    jd_text = clean_text(extract_text(jd_path))

    if not resume_text or not jd_text:
        return {"error": "Could not extract text from one or both files."}

    # Retrieve context from the vector store
    # Combining both texts for a more holistic retrieval query
    context_docs = retrieve(resume_text + "\n" + jd_text, index_dir, k)
    context_str = "\n---\n".join([d.page_content for d in context_docs])

    # Setup LLM and Chain
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0, model_name="gpt-3.5-turbo-0125")
    prompt = PromptTemplate(
        input_variables=["resume", "job", "context"], 
        template=MATCHING_PROMPT_TEMPLATE
    )

    # Use the pipe operator (|) for a sequential chain (LCEL standard)
    chain: Runnable = prompt | llm
    
    # Invoke the chain; the response is an AIMessage object
    response = chain.invoke({
        "resume": resume_text, 
        "job": jd_text, 
        "context": context_str
    })
    
    # Extract the string content from the AIMessage
    resp_str = response.content

    # Parse the expected JSON output
    try:
        # Attempt to find and load the JSON object within the raw response string
        start = resp_str.find("{")
        end = resp_str.rfind("}")
        
        if start != -1 and end != -1:
            json_str = resp_str[start:end+1]
            result = json.loads(json_str)
        else:
            raise ValueError("JSON structure not found in LLM response.")
            
    except Exception as e:
        print(f"[Warning] Failed to parse JSON. Error: {e}")
        # Fallback to returning the raw response string
        result = {"raw_llm_output": resp_str}

    print(json.dumps(result, indent=2))
    return result

# --- Streamlit Application ---

def run_ui(index_dir: str):
    """Initializes and runs the Streamlit web interface."""
    if not STREAMLIT_AVAILABLE:
        st.error("Streamlit is not installed. Please run: pip install streamlit")
        return

    st.title("📄 Job Matcher (RAG-Enabled)")
    st.markdown("Upload a Resume and a Job Description to get an AI-powered match score.")

    # File Uploaders
    r_file = st.file_uploader("Upload Resume", type=["pdf", "txt", "docx", "doc"])
    j_file = st.file_uploader("Upload Job Description", type=["pdf", "txt", "docx", "doc"])

    if r_file and j_file:
        # Use tempfile to save uploaded files for disk access
        with tempfile.NamedTemporaryFile(delete=False, suffix=pathlib.Path(r_file.name).suffix) as rtmp:
            rtmp.write(r_file.read())
            r_path = rtmp.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=pathlib.Path(j_file.name).suffix) as jtmp:
            jtmp.write(j_file.read())
            j_path = jtmp.name

        if st.button("Get Match Score", use_container_width=True):
            with st.spinner("Analyzing documents and generating match score..."):
                try:
                    result = match(r_path, j_path, index_dir)
                    
                    if "error" in result:
                        st.error(result["error"])
                    elif "score" in result:
                        score = result["score"]
                        st.success(f"### Match Score: {score}/100")
                        
                        st.write("### Key Explanations:")
                        if isinstance(result.get("explanation"), list):
                            st.markdown("\n".join(f"- {point}" for point in result["explanation"]))
                        else:
                            st.json(result) # Show full JSON if structure is unexpected
                    else:
                        st.warning("The LLM returned a non-standard response.")
                        st.json(result) # Show raw output

                except Exception as e:
                    st.error(f"An unexpected error occurred during matching: {e}")
                finally:
                    # Clean up temporary files
                    os.unlink(r_path)
                    os.unlink(j_path)
                    
    else:
        st.info("Please upload both a Resume and a Job Description to proceed.")

# --- Main Execution ---

def main():
    """Main function to handle command-line arguments."""
    parser = argparse.ArgumentParser(description="A RAG-based tool for matching resumes to job descriptions.")
    parser.add_argument(
        "--action", 
        choices=["index", "match", "serve"], 
        required=True,
        help="Action to perform: 'index' documents, 'match' two files, or 'serve' the Streamlit UI."
    )
    parser.add_argument("--data-dir", default="./data", help="Directory containing 'resumes' and 'jds' folders.")
    parser.add_argument("--index-dir", default="./index", help="Directory to save/load the FAISS vector store index.")
    parser.add_argument("--resume", help="Path to the resume file (for 'match' action).")
    parser.add_argument("--jd", help="Path to the job description file (for 'match' action).")
    args = parser.parse_args()

    if args.action == "index":
        index_documents(args.data_dir, args.index_dir)
        
    elif args.action == "match":
        if not args.resume or not args.jd:
            print("[Error] --resume and --jd arguments are required for the 'match' action.")
            sys.exit(1)
        # The result is printed within the match function
        match(args.resume, args.jd, args.index_dir)
        
    elif args.action == "serve":
        # Streamlit must be run via the CLI, but this function prepares the environment
        if not STREAMLIT_AVAILABLE:
            print("Install streamlit to use the UI. Then run: streamlit run your_script_name.py -- --action serve")
            sys.exit(1)
            
        # NOTE: For Streamlit to work correctly via the CLI, you usually run:
        # streamlit run your_script_name.py 
        # The internal run_ui is called, but we handle the index_dir here.
        # This setup is mostly for internal testing if running the python script directly.
        run_ui(args.index_dir)

if __name__ == "__main__":
    main()