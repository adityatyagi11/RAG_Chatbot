from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from rag_engine import RAGEngine
from models import ChatRequest, ChatResponse, SearchRequest, SearchResponse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Wikipedia RAG Chatbot API",
    description="A Retrieval-Augmented Generation chatbot that queries Wikipedia",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG engine
rag_engine = RAGEngine()

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG engine on startup"""
    try:
        await rag_engine.initialize()
        logger.info("RAG engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG engine: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Wikipedia RAG Chatbot API",
        "status": "running",
        "version": "1.0.0"
    }

@app.post("/search", response_model=SearchResponse)
async def search_wikipedia(request: SearchRequest):
    """Search Wikipedia articles and index them"""
    try:
        results = await rag_engine.search_and_index(
            query=request.query,
            max_results=request.max_results
        )
        return SearchResponse(
            query=request.query,
            results=results,
            total_indexed=len(results)
        )
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the RAG system"""
    try:
        response = await rag_engine.chat(
            message=request.message,
            session_id=request.session_id,
            use_history=request.use_history
        )
        return response
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/history")
async def get_chat_history(session_id: str):
    """Get chat history for a session"""
    try:
        history = await rag_engine.get_chat_history(session_id)
        return {"session_id": session_id, "history": history}
    except Exception as e:
        logger.error(f"History retrieval error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear chat history for a session"""
    try:
        await rag_engine.clear_session(session_id)
        return {"message": f"Session {session_id} cleared successfully"}
    except Exception as e:
        logger.error(f"Session clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vector-store/stats")
async def get_vector_store_stats():
    """Get vector store statistics"""
    try:
        stats = await rag_engine.get_vector_store_stats()
        return stats
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )