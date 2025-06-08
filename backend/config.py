import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY=os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME=os.environ.get("PINECONE_INDEX_NAME")
PINECONE_ENVIRONMENT=os.environ.get("PINECONE_ENVIRONMENT")
UPLOAD_DIR = "uploaded_documents" 
os.makedirs(UPLOAD_DIR, exist_ok=True)
