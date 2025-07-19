from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class SearchRequest(BaseModel):
    """Request model for Wikipedia search"""
    query: str = Field(..., description="Search query for Wikipedia", min_length=1)
    max_results: int = Field(default=5, description="Maximum number of results", ge=1, le=20)

class SearchResult(BaseModel):
    """Model for individual Wikipedia search result"""
    title: str = Field(..., description="Wikipedia article title")
    url: str = Field(..., description="Wikipedia article URL")
    summary: str = Field(..., description="Article summary")
    chunks_created: int = Field(..., description="Number of text chunks created")

class SearchResponse(BaseModel):
    """Response model for Wikipedia search"""
    query: str = Field(..., description="Original search query")
    results: List[SearchResult] = Field(..., description="List of search results")
    total_indexed: int = Field(..., description="Total number of articles indexed")

class ChatRequest(BaseModel):
    """Request model for chat interaction"""
    message: str = Field(..., description="User message", min_length=1)
    session_id: str = Field(..., description="Session identifier")
    use_history: bool = Field(default=True, description="Whether to use conversation history")

class Source(BaseModel):
    """Model for source information"""
    title: str = Field(..., description="Source document title")
    url: str = Field(default="", description="Source document URL")
    chunk_content: str = Field(..., description="Relevant chunk content")

class ChatResponse(BaseModel):
    """Response model for chat interaction"""
    message: str = Field(..., description="Original user message")
    response: str = Field(..., description="AI generated response")
    session_id: str = Field(..., description="Session identifier")
    sources: List[Source] = Field(default=[], description="Source documents used")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")

class SessionInfo(BaseModel):
    """Model for session information"""
    session_id: str = Field(..., description="Session identifier")
    created_at: datetime = Field(default_factory=datetime.now, description="Session creation time")
    message_count: int = Field(default=0, description="Number of messages in session")

class VectorStoreStats(BaseModel):
    """Model for vector store statistics"""
    total_documents: int = Field(..., description="Total number of documents in vector store")
    collection_name: str = Field(..., description="Vector store collection name")
    embedding_model: str = Field(..., description="Embedding model used")

class ChatHistory(BaseModel):
    """Model for chat history"""
    session_id: str = Field(..., description="Session identifier")
    messages: List[Dict[str, Any]] = Field(..., description="List of chat messages")

class ErrorResponse(BaseModel):
    """Model for error responses"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")

class HealthCheck(BaseModel):
    """Model for health check response"""
    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Status message")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")

class WikipediaArticle(BaseModel):
    """Model for Wikipedia article data"""
    title: str = Field(..., description="Article title")
    url: str = Field(..., description="Article URL")
    content: str = Field(..., description="Article content")
    summary: str = Field(..., description="Article summary")
    categories: List[str] = Field(default=[], description="Article categories")
    links: List[str] = Field(default=[], description="Article links")
    images: List[str] = Field(default=[], description="Article images")
    language: str = Field(default="en", description="Article language")

class DocumentChunk(BaseModel):
    """Model for document chunks"""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    content: str = Field(..., description="Chunk content")
    metadata: Dict[str, Any] = Field(default={}, description="Chunk metadata")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")

class IndexingRequest(BaseModel):
    """Request model for document indexing"""
    documents: List[WikipediaArticle] = Field(..., description="Documents to index")
    chunk_size: int = Field(default=1000, description="Text chunk size")
    chunk_overlap: int = Field(default=200, description="Text chunk overlap")

class IndexingResponse(BaseModel):
    """Response model for document indexing"""
    indexed_documents: int = Field(..., description="Number of documents indexed")
    total_chunks: int = Field(..., description="Total number of chunks created")
    processing_time: float = Field(..., description="Processing time in seconds")

class QueryResponse(BaseModel):
    """Model for query response with sources"""
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    sources: List[Source] = Field(..., description="Source documents")
    confidence: Optional[float] = Field(None, description="Confidence score")
    processing_time: float = Field(..., description="Query processing time")

# Configuration models
class RAGConfig(BaseModel):
    """Configuration model for RAG system"""
    embedding_model: str = Field(default="text-embedding-ada-002", description="Embedding model name")
    llm_model: str = Field(default="gpt-3.5-turbo", description="Language model name")
    chunk_size: int = Field(default=1000, description="Text chunk size")
    chunk_overlap: int = Field(default=200, description="Text chunk overlap")
    max_retrieval_docs: int = Field(default=3, description="Maximum documents to retrieve")
    temperature: float = Field(default=0.7, description="LLM temperature")

class DatabaseConfig(BaseModel):
    """Configuration model for database settings"""
    collection_name: str = Field(default="wikipedia_docs", description="Vector store collection name")
    persist_directory: str = Field(default="./chroma_db", description="Database persistence directory")
    similarity_threshold: float = Field(default=0.7, description="Similarity search threshold")

# Utility models
class PaginationParams(BaseModel):
    """Model for pagination parameters"""
    page: int = Field(default=1, description="Page number", ge=1)
    size: int = Field(default=10, description="Page size", ge=1, le=100)

class SortParams(BaseModel):
    """Model for sorting parameters"""
    field: str = Field(..., description="Field to sort by")
    order: str = Field(default="asc", description="Sort order (asc/desc)")

class FilterParams(BaseModel):
    """Model for filtering parameters"""
    start_date: Optional[datetime] = Field(None, description="Start date filter")
    end_date: Optional[datetime] = Field(None, description="End date filter")
    session_id: Optional[str] = Field(None, description="Session ID filter")
    min_confidence: Optional[float] = Field(None, description="Minimum confidence score")