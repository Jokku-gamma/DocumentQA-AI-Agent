import openai
from openai import AsyncOpenAI
from config import OPENAI_API_KEY, UPLOAD_DIR
from models import ProcessedDocument
import uuid
import os
import aiofiles
import filetype
import json
import base64
import fitz
from typing import List, Dict, Any, Optional
import io 
import httpx
from PIL import Image 
import pytesseract 
import arxiv
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
custom_client = httpx.AsyncClient(trust_env=False)
client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    http_client=custom_client
)
chroma_client = chromadb.Client() 
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)
ALL_DOCUMENTS_COLLECTION_NAME = "all_documents_knowledge_base"
all_documents_collection = chromadb.Client().get_or_create_collection( # Corrected: Use chroma_client instance
    name=ALL_DOCUMENTS_COLLECTION_NAME,
    embedding_function=openai_ef
)
documents_store: dict[str, ProcessedDocument] = {}
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200, 
    length_function=len,
    is_separator_regex=False,
)

async def embed_and_store_chunks(document_id: str, filename: str, full_text: str):
    """Chunks the text, embeds them, and stores in the shared ChromaDB collection."""
    
    chunks = text_splitter.split_text(full_text)
    chunk_ids = [f"{document_id}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [
        {"document_id": document_id, "filename": filename, "chunk_index": i} 
        for i in range(len(chunks))
    ]

    all_documents_collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=chunk_ids
    )
    print(f"Document {filename} (ID: {document_id}) chunked and embedded into the shared ChromaDB collection.")
async def process_pdf_document(file_content: bytes, filename: str) -> ProcessedDocument:
    kind = filetype.guess(file_content)
    if kind is None:
        raise ValueError("Could not determine file type from content.")
    if kind.mime != "application/pdf":
        raise ValueError(f"Unsupported file type: {kind.mime}. Only PDFs are allowed.")

    doc_id = str(uuid.uuid4())
    
    if doc_id in documents_store:
        print(f"Document {doc_id} already processed in this session. Returning existing.")
        return documents_store[doc_id]
    extracted_metadata = {}
    full_document_text_for_rag = "" 
    combined_direct_text = "" 
    combined_tesseract_text = "" 
    try:
        pdf_document = fitz.open(stream=file_content, filetype="pdf")
        num_pages = len(pdf_document)
        direct_text_chunks = []
        for page_num in range(num_pages):
            page = pdf_document.load_page(page_num)
            text = page.get_text("text")
            direct_text_chunks.append(text)
        combined_direct_text = "\n\n".join(direct_text_chunks).strip()
        if len(combined_direct_text) > 200:
            print(f"Direct text extraction yielded sufficient text for {filename}.")
            pass
        else:
            print(f"Direct text extraction yielded little text for {filename}. Trying Tesseract OCR.")
            tesseract_text_chunks = []
            max_pages_for_ocr = min(num_pages, 5)

            for page_num in range(max_pages_for_ocr):
                page = pdf_document.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(3, 3)) 
                img_bytes = pix.pil_tobytes(format="PNG")
                
                try:
                    img = Image.open(io.BytesIO(img_bytes))
                    ocr_text = pytesseract.image_to_string(img)
                    tesseract_text_chunks.append(ocr_text)
                except Exception as ocr_e:
                    print(f"Tesseract OCR failed on page {page_num} for {filename}: {ocr_e}")
                    tesseract_text_chunks.append("") 
            
            combined_tesseract_text = "\n\n".join(tesseract_text_chunks).strip()

            if len(combined_tesseract_text) > 200: 
                print(f"Tesseract OCR successful for {filename}.")
                pass 
            else:
                print(f"Tesseract OCR yielded little text. Relying solely on OpenAI Vision API for extraction.")
                pass 
                image_parts = []
        zoom_x = 3
        zoom_y = 3
        mat = fitz.Matrix(zoom_x, zoom_y)
        
        max_pages_to_process_vision = min(num_pages, 10) # Limit for Vision API to manage cost/speed
        
        for page_num in range(max_pages_to_process_vision):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap(matrix=mat) 
            img_bytes = pix.pil_tobytes(format="PNG")
            base64_image = base64.b64encode(img_bytes).decode('utf-8')
            image_parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}",
                    "detail": "high"
                },
            })
        
        if num_pages > max_pages_to_process_vision:
            print(f"Warning: Only processed the first {max_pages_to_process_vision} pages of the PDF for Vision API structured extraction.")

        prompt_text = f"""
        Analyze the following PDF document.
        Extract the following information:
        - **Title:** The main title of the paper.
        - **Abstract:** The abstract of the paper.
        - **Sections:** A list of sections, where each section has:
            - `title`: The section title.
            - `content`: A summary of the section's text.
            - (Optional) `tables`: If tables are present in the section, describe them or extract key data.
            - (Optional) `figures`: If figures are present, describe them.
        - **References:** A list of references.

        Output the extracted information as a JSON object with the following schema:
        {{
            "title": "...",
            "abstract": "...",
            "sections": [
                {{"title": "...", "content": "...", "tables": [...], "figures": [...]}},
                ...
            ],
            "references": [...]
        }}

        If any element is not found, use an empty string or empty list as appropriate.
        Ensure high accuracy, especially for tables and key results.
        """
        messages_content = [{"type": "text", "text": prompt_text}] + image_parts

        vision_response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": messages_content}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        extracted_metadata = json.loads(vision_response.choices[0].message.content)
        full_document_text_for_rag = extracted_metadata.get("abstract", "") + "\n\n"
        for section in extracted_metadata.get("sections", []):
            full_document_text_for_rag += f"{section.get('title', '')}\n{section.get('content', '')}\n\n"
        pdf_document.close() 
    except Exception as e:
        print(f"Error during PDF content processing pipeline: {e}")
        if not full_document_text_for_rag.strip() and combined_direct_text.strip():
            print("Falling back to direct text extraction for RAG due to Vision API failure.")
            full_document_text_for_rag = combined_direct_text
        elif not full_document_text_for_rag.strip() and combined_tesseract_text.strip():
            print("Falling back to Tesseract text extraction for RAG due to Vision API failure.")
            full_document_text_for_rag = combined_tesseract_text
        else:
            raise ValueError(f"Failed to process PDF document (no text extracted): {e}")
    if not full_document_text_for_rag.strip():
        raise ValueError("No meaningful text could be extracted from the PDF using any method.")
    await embed_and_store_chunks(doc_id, filename, full_document_text_for_rag)
    processed_doc = ProcessedDocument(
        id=doc_id,
        filename=filename,
        extracted_text=full_document_text_for_rag, 
        metadata=extracted_metadata 
    )
    documents_store[doc_id] = processed_doc
    return processed_doc

async def query_document(question: str, filename: Optional[str] = None) -> str:
    query_params = {
        "query_texts": [question],
        "n_results": 5, 
    }

    if filename:
        query_params["where"] = {"filename": {"$eq": filename}}
    results = all_documents_collection.query(**query_params)
    retrieved_chunks = results['documents'][0] if results and results['documents'] else []
    retrieved_metadatas = results['metadatas'][0] if results and results['metadatas'] else []
    
    if not retrieved_chunks:
        if filename:
            return f"No relevant information found in '{filename}' for your question, or the file was not found."
        return "No relevant information found in the documents for your question."
    context_with_sources = []
    for i, chunk in enumerate(retrieved_chunks):
        metadata = retrieved_metadatas[i]
        chunk_filename = metadata.get("filename", "Unknown Document")
        chunk_index = metadata.get("chunk_index", "N/A")
        context_with_sources.append(f"--- Document: {chunk_filename} (Chunk {chunk_index}) ---\n{chunk}")
    
    context = "\n\n".join(context_with_sources)
    prompt = f"""
    You are an AI assistant specializing in answering questions based on provided document content.
    Use ONLY the following context to answer the user's question.
    Cite the source document filename (and optionally chunk index if helpful) when answering.
    If the answer is not explicitly stated in the context, state that you cannot find the information.

    Document Context:
    {context}

    User's Question: {question}

    Answer:
    """
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

async def summarize_document_section(document_id: str, section_title: Optional[str] = None, granularity: str = "overview") -> str:
    if document_id not in documents_store:
        raise ValueError(f"Document with ID {document_id} not found in this session's memory.")
    doc = documents_store[document_id]
    if not doc.metadata:
        return "Cannot summarize: structured metadata was not extracted for this document."
    content_to_summarize = ""
    if section_title:
        found_section = False
        for section in doc.metadata.get("sections", []):
            if section["title"].lower() == section_title.lower():
                content_to_summarize = section["content"]
                found_section = True
                break
        if not found_section:
            raise ValueError(f"Section '{section_title}' not found in document's extracted metadata.")
    else:
        content_to_summarize += doc.metadata.get("abstract", "") + "\n\n"
        for section in doc.metadata.get("sections", []):
            content_to_summarize += f"{section['title']}:\n{section['content']}\n\n"

    if not content_to_summarize.strip():
        return "No content found to summarize in the extracted metadata."

    prompt = f"""
    Summarize the following content.
    Granularity: {granularity}.

    Content:
    {content_to_summarize}

    Summary:
    """
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

async def extract_evaluation_results(document_id: str, query: str) -> Dict[str, Any]:
    if document_id not in documents_store:
        raise ValueError(f"Document with ID {document_id} not found in this session's memory.")
    doc = documents_store[document_id]
    if not doc.metadata:
        return {"error": "Cannot extract results: structured metadata was not extracted for this document."}
    document_content = json.dumps(doc.metadata, indent=2)
    prompt = f"""
    From the following document content, extract specific evaluation results or key data points related to the query: "{query}".
    Focus on numerical values, metrics, and their descriptions.
    Output the extracted information as a JSON object with relevant key-value pairs.
    For example, if the query is "accuracy and F1-score", the output could be:
    {{"accuracy": "95.2%", "f1_score": "0.89"}}
    If the query is "key findings", it could be:
    {{"finding_1": "...", "finding_2": "..."}}
    If no relevant information is found, return an empty JSON object: {{}}.

    Document Content (structured JSON):
    {document_content}

    Extraction Query: {query}

    Extracted Results (JSON):
    """
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        print(f"Warning: OpenAI did not return valid JSON for extraction. Raw response: {response.choices[0].message.content}")
        return {"error": "Could not parse extracted results as JSON."}

async def arxiv_lookup_tool(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    print(f"Simulating Arxiv lookup for: '{query}' with {max_results} results.")
    try:
        search = arxiv.Search(
            query = query,
            max_results = max_results,
            sort_by = arxiv.SortCriterion.Relevance,
            sort_order = arxiv.SortOrder.Descending
        )
        papers_data = []
        for result in search.results():
            papers_data.append({
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "summary": result.summary,
                "url": result.entry_id,
                "published": result.published.isoformat()
            })
        return papers_data
    except Exception as e:
        print(f"Error calling Arxiv API: {e}")
        return []