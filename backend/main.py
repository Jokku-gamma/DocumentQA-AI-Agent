from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import aiofiles
import json
from typing import Optional
from gemini import (
    process_pdf_document, 
    query_document, 
    summarize_document_section, 
    extract_evaluation_results,
    arxiv_lookup_tool, 
    documents_store 
)
from openai import AsyncOpenAI 
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

class QueryRequest(BaseModel):
    question: str
    filename: Optional[str] = None 
tools = [
    {
        "type": "function",
        "function": {
            "name": "arxiv_lookup_tool",
            "description": "Looks up academic papers on Arxiv based on a search query. Use this for general research paper queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query for Arxiv (e.g., 'explainable AI', 'quantum computing security', 'neural networks').",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of search results to return (default is 5, max 10).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_document",
            "description": "Answers questions based on the content of uploaded PDF documents using Retrieval Augmented Generation (RAG). Use this if the user asks about an *uploaded* document or wants information *from* a document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to answer from the document context.",
                    },
                    "filename": {
                        "type": "string",
                        "description": "Optional: The specific filename (e.g., 'my_report.pdf') to query. If not provided, searches all uploaded documents.",
                    },
                },
                "required": ["question"],
            },
        },
    },
]


@app.get("/")
async def read_root():
    return {"message": "Gemini PDF Chatbot Backend is running!"}

@app.post("/upload-document/")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_content = await file.read()
    try:
        processed_doc = await process_pdf_document(file_content, file.filename)
        return JSONResponse(content={
            "message": "File processed successfully",
            "filename": processed_doc.filename,
            "document_id": processed_doc.id,
            "extracted_text_preview": processed_doc.extracted_text[:200] + "..."
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")

@app.get("/documents/")
async def list_documents():
    """Lists all currently processed documents in memory."""
    return [doc.dict() for doc in documents_store.values()]


@app.post("/query-document/")
async def query_document_endpoint(request: QueryRequest):
    user_question = request.question
    target_filename = request.filename

    messages = [{"role": "user", "content": user_question}]
    try:
        response = await client.chat.completions.create(
            model="gpt-4o", 
            messages=messages,
            tools=tools,       
            tool_choice="auto", 
            temperature=0.7    
        )
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        if tool_calls:
            available_functions = {
                "arxiv_lookup_tool": arxiv_lookup_tool,
                "query_document": query_document, 
            }
            messages.append(response_message) 

            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = available_functions.get(function_name)

                if function_to_call:
                    function_args = json.loads(tool_call.function.arguments)
                    
                    print(f"LLM decided to call tool: {function_name} with args: {function_args}")

                    if function_name == "arxiv_lookup_tool":
                        tool_output = await function_to_call(
                            query=function_args.get("query"),
                            max_results=function_args.get("max_results")
                        )
                        if tool_output:
                            
                            return JSONResponse(content={
                                "answer": tool_output, 
                                "type": "arxiv_papers" 
                            })
                        else:
                            return JSONResponse(content={
                                "answer": "No papers found on Arxiv for your query.",
                                "type": "text"
                            })
                    elif function_name == "query_document":
                        rag_answer = await function_to_call(
                            question=function_args.get("question"),
                            filename=function_args.get("filename")
                        )
                        return JSONResponse(content={
                            "answer": rag_answer,
                            "type": "text" 
                        })
                    else:
                        tool_response_content = f"Tool '{function_name}' executed but no specific output formatting."

                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": tool_response_content,
                        }
                    )
                else:
                    return JSONResponse(content={"answer": f"Error: Backend attempted to call an unknown tool: {function_name}"}, status_code=500)

            second_response = await client.chat.completions.create(
                model="gpt-4o",
                messages=messages, 
                temperature=0.7
            )
            return JSONResponse(content={
                "answer": second_response.choices[0].message.content.strip(),
                "type": "text" 
            })        
        else:
            rag_answer = await query_document(user_question, target_filename)
            return JSONResponse(content={"answer": rag_answer, "type": "text"})

    except Exception as e:
        print(f"Error in query-document endpoint: {e}")
        return JSONResponse(content={"answer": f"An error occurred while processing your request: {e}"}, status_code=500)