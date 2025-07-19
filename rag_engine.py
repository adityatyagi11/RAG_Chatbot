import os
import asyncio
from typing import List, Dict, Any, Optional
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from wikipedia_retriever import WikipediaRetriever
from models import ChatResponse, SearchResult
import logging
from dotenv import load_dotenv
load_dotenv()


logger = logging.getLogger(__name__)

class RAGEngine:
    """Core RAG engine for Wikipedia chatbot"""
    
    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.retriever = None
        self.wikipedia_retriever = WikipediaRetriever()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.sessions = {}  # Store conversation memories
        
    async def initialize(self):
        """Initialize the RAG engine components"""
        try:
            # Initialize OpenAI components
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            
            self.llm = ChatOpenAI(
                temperature=0.7,
                model_name="gpt-3.5-turbo",
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            
            # Initialize vector store
            self.vector_store = Chroma(
                collection_name="wikipedia_docs",
                embedding_function=self.embeddings,
                persist_directory="./chroma_db"
            )
            
            # Create retriever
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            
            logger.info("RAG engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG engine: {e}")
            raise
    
    async def search_and_index(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Search Wikipedia and index the results"""
        try:
            # Search Wikipedia
            wikipedia_results = await self.wikipedia_retriever.search(
                query=query,
                max_results=max_results
            )
            
            results = []
            documents = []
            
            for result in wikipedia_results:
                # Create documents for vector store
                content = result.get('content', '')
                if content:
                    # Split content into chunks
                    chunks = self.text_splitter.split_text(content)
                    
                    for i, chunk in enumerate(chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                "title": result.get('title', ''),
                                "url": result.get('url', ''),
                                "chunk_id": i,
                                "source": "wikipedia"
                            }
                        )
                        documents.append(doc)
                
                # Create search result
                search_result = SearchResult(
                    title=result.get('title', ''),
                    url=result.get('url', ''),
                    summary=result.get('summary', ''),
                    chunks_created=len(chunks) if content else 0
                )
                results.append(search_result)
            
            # Add documents to vector store
            if documents:
                self.vector_store.add_documents(documents)
                logger.info(f"Added {len(documents)} chunks to vector store")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in search_and_index: {e}")
            raise
    
    async def chat(self, message: str, session_id: str, use_history: bool = True) -> ChatResponse:
        """Chat with the RAG system"""
        try:
            # Get or create conversation memory for this session
            if session_id not in self.sessions:
                self.sessions[session_id] = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True,
                    output_key="answer"
                )
            
            memory = self.sessions[session_id] if use_history else None
            
            # Create conversation chain
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever,
                memory=memory,
                return_source_documents=True,
                verbose=True
            )
            
            # Get response
            result = qa_chain({"question": message})
            
            # Extract source information
            sources = []
            for doc in result.get("source_documents", []):
                source_info = {
                    "title": doc.metadata.get("title", "Unknown"),
                    "url": doc.metadata.get("url", ""),
                    "chunk_content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                }
                sources.append(source_info)
            
            return ChatResponse(
                message=message,
                response=result["answer"],
                session_id=session_id,
                sources=sources
            )
            
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            raise
    
    async def get_chat_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get chat history for a session"""
        if session_id not in self.sessions:
            return []
        
        memory = self.sessions[session_id]
        messages = memory.chat_memory.messages
        
        history = []
        for msg in messages:
            if hasattr(msg, 'content'):
                history.append({
                    "type": msg.__class__.__name__,
                    "content": msg.content
                })
        
        return history
    
    async def clear_session(self, session_id: str):
        """Clear chat history for a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    async def get_vector_store_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            collection = self.vector_store._collection
            count = collection.count()
            
            return {
                "total_documents": count,
                "collection_name": "wikipedia_docs",
                "embedding_model": "text-embedding-ada-002"
            }
        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}")
            return {"error": str(e)}