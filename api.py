from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import numpy as np
import io
import os
import gc
from dotenv import load_dotenv
from pydub import AudioSegment
from utils import (
    authenticate,
    split_documents,
    build_vectorstore,
    retrieve_context,
    retrieve_context_approx,
    build_prompt,
    ask_gemini,
    load_documents_gradio,
    transcribe
)

load_dotenv()

app = FastAPI()

# Define the specific origins that are allowed to make requests to your API
origins = [
    "http://localhost:3000",  # For local development
    "https://chat-docx-ai-vercel.vercel.app", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = authenticate()
store = {"value": None}


@app.options("/upload")
async def options_upload():
    return JSONResponse(
        content={"status": "ok"},
        headers={
            "Access-Control-Allow-Origin": "https://chat-docx-ai-vercel.app",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        },
    )


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    headers = {
        "Access-Control-Allow-Origin": "https://chat-docx-ai-vercel.app"
    }
    try:
        if not files:
            return JSONResponse(
                content={"status": "error", "message": "No files uploaded."},
                status_code=400,
                headers=headers
            )
        
        # Explicitly clear memory before processing new files
        print("Clearing previous vector store from memory...")
        store["value"] = None
        gc.collect()
        print("Memory cleared.")

        print("Starting document processing...")
        raw_docs = load_documents_gradio(files)
        print("Documents loaded. Splitting documents...")
        chunks = split_documents(raw_docs)
        print("Documents split. Building vector store...")
        store["value"] = build_vectorstore(chunks)
        print("Vector store built successfully.")
        
        return JSONResponse(
            content={"status": "success", "message": "Document processed successfully! You can now ask questions."},
            headers=headers
        )
    except Exception as e:
        print(f"An error occurred during upload: {e}")
        return JSONResponse(
            content={"status": "error", "message": f"An internal server error occurred: {e}"},
            status_code=500,
            headers=headers
        )

@app.post("/ask")
async def ask(
    text: Optional[str] = Form(None),
    audio: Optional[UploadFile] = File(None)
):
    transcribed = None
    if store["value"] is None:
        return JSONResponse({"status": "error", "message": "Please upload and process a document first."}, status_code=400)
    if text and text.strip():
        query = text.strip()
    elif audio is not None:
        audio_bytes = await audio.read()
        try:
            audio_io = io.BytesIO(audio_bytes)
            audio_seg = AudioSegment.from_file(audio_io)
            y = np.array(audio_seg.get_array_of_samples()).astype(np.float32)
            if audio_seg.channels == 2:
                y = y.reshape((-1, 2)).mean(axis=1)  # Convert to mono
            y /= np.max(np.abs(y))  # Normalize to [-1, 1]
            sr = audio_seg.frame_rate
            transcribed = transcribe((sr, y))
            query = transcribed
        except FileNotFoundError as e:
            return JSONResponse({"status": "error", "message": "Audio decode failed: ffmpeg is not installed or not in PATH. Please install ffmpeg."}, status_code=400)
        except Exception as e:
            return JSONResponse({"status": "error", "message": f"Audio decode failed: {str(e)}"}, status_code=400)
    else:
        return JSONResponse({"status": "error", "message": "Please provide a question by typing or speaking."}, status_code=400)
    if store["value"]["chunks"] <= 50:
        top_chunks = retrieve_context(query, store["value"])
    else:
        top_chunks = retrieve_context_approx(query, store["value"])
    prompt = build_prompt(top_chunks, query)
    answer = ask_gemini(prompt, client)
    return {"status": "success", "answer": answer.strip(), "transcribed": transcribed}
