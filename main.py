import os
import fitz  # PyMuPDF
import tempfile
import re
import requests
import email
from bs4 import BeautifulSoup
import extract_msg
from docx import Document
from typing import List
from fastapi import FastAPI, Request, Body, HTTPException, Depends
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# LangChain & LLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeStore
from pinecone import Pinecone, ServerlessSpec

from fastapi.security import HTTPBearer
from fastapi.openapi.models import APIKey, APIKeyIn, SecuritySchemeType
from fastapi.openapi.utils import get_openapi
from fastapi import Security

load_dotenv()
load_dotenv(dotenv_path=".env")

security_scheme = HTTPBearer()

app = FastAPI(
    title="HackRx AI Assistant",
    description="Extract answers from uploaded documents using Gemini LLM",
    version="1.0.0"
)


# Inject Bearer Auth into OpenAPI docs
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        }
    }
    for path in openapi_schema["paths"].values():
        for method in path.values():
            method.setdefault("security", [{"BearerAuth": []}])
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

# Token required by HackRx
API_KEY = os.getenv("API_KEY")


# ---- Auth Dependency ----
def verify_token(request: Request):
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    token = auth.split("Bearer ")[1]
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid token")


# ---- Embedding Model ----
embedding_model = HuggingFaceEmbeddings(model_name="./models/all-MiniLM-L6-v2")

faiss_store = None

# ---- Pinecone Initialization ----
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = os.getenv("PINECONE_INDEX_NAME")
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # For MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=os.getenv("PINECONE_ENV")  # usually "us-east-1"
        )
    )
# ---- Utility Functions ----
def clean_text(text):
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()


def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=200,
        chunk_overlap=50,
        length_function=len,
    )
    return splitter.split_text(text)

def embed_to_pinecone(chunks):
    # The PineconeStore class will automatically use the API key
    # from the environment variables you loaded with load_dotenv().
    return PineconeStore.from_texts(
        texts=chunks,
        embedding=embedding_model,
        index_name=index_name,
        namespace=None
    )

def embed_to_memory(chunks):
    return FAISS.from_texts(chunks, embedding_model)


def extract_text_from_pdf_bytes(data: bytes):
    with fitz.open(stream=data, filetype="pdf") as doc:
        return "".join([page.get_text() for page in doc])


def extract_text_from_docx_bytes(data: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as temp:
        temp.write(data)
        temp.flush()
        doc = Document(temp.name)
        return "\n".join([para.text for para in doc.paragraphs])


def extract_text_from_eml_bytes(data: bytes):
    msg = email.message_from_bytes(data)
    for part in msg.walk():
        content_type = part.get_content_type()
        payload = part.get_payload(decode=True)
        if payload:
            try:
                if content_type == "text/plain":
                    return payload.decode(errors="ignore")
                elif content_type == "text/html":
                    return BeautifulSoup(payload.decode(errors="ignore"), "html.parser").get_text()
            except Exception:
                continue
    return ""


def extract_text_from_msg_bytes(data: bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".msg") as temp:
        temp.write(data)
        temp.flush()
        msg = extract_msg.Message(temp.name)
        return msg.body or ""


def get_llm_chain():
    prompt_template = """
    You are an expert AI assistant trained to extract and summarize detailed information from insurance policy documents.

    Your goal is to answer the user's QUESTION using only the provided CONTEXT. Follow these rules:

    📝 Guidelines:

    Base your answer strictly on the CONTEXT.

    Aim for 30-40 words per answer

    Use formal policy language where applicable (e.g., “shall indemnify”, “subject to”, “provided that”).

    Each answer should be a complete, self-contained clause: detailed enough to capture eligibility, limits, waiting periods, conditions, and exceptions — but without becoming overly verbose or repetitive.

    If multiple distinct points are found, return them as separate items in the list.

    If the CONTEXT does not provide relevant information say no data found.
    
    Stick to what's asked by user.

    Don't add any additional comments of your own stick to the answer itself.

    —

    📄 CONTEXT:
    {context}

    ❓ QUESTION:
    {question}

    —
    """
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    llm = ChatGoogleGenerativeAI(model="gemma-3n-e2b-it", temperature=0.3)
    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)


# ---- Main HackRx Endpoint ----
@app.post("/api/v1/hackrx/run")
async def hackrx_run(request: Request, _: None = Depends(verify_token), payload: dict = Body(...)):
    global faiss_store
    url = payload.get("documents")
    questions = payload.get("questions", [])

    if not url or not questions:
        return {"error": "Missing 'documents' or 'questions'."}

    try:
        # Step 1: Download file
        response = requests.get(url)
        if response.status_code != 200:
            return {"error": "Failed to download document from URL."}
        file_data = response.content
        ext = url.split(".")[-1].split("?")[0].lower()

        # Step 2: Extract text based on file type
        if ext == "pdf":
            full_text = extract_text_from_pdf_bytes(file_data)
        elif ext == "docx":
            full_text = extract_text_from_docx_bytes(file_data)
        elif ext == "eml":
            full_text = extract_text_from_eml_bytes(file_data)
        elif ext == "msg":
            full_text = extract_text_from_msg_bytes(file_data)
        else:
            return {"error": f"Unsupported file format: {ext}"}

        # Step 3: Clean, chunk, embed
        cleaned = clean_text(full_text)
        chunks = get_chunks(cleaned)
        vector_store = embed_to_pinecone(chunks)

        # Step 4: Run Gemini on questions
        chain = get_llm_chain()
        answers = []
        for q in questions:
            docs = vector_store.similarity_search(q, k=7)
            result = chain({"input_documents": docs, "question": q}, return_only_outputs=True)
            answers.append(result["output_text"])

        return {"answers": answers}

    except Exception as e:
        return {"error": str(e)}



