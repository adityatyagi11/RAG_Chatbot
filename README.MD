Wikipedia RAG Chatbot
A production-ready Retrieval-Augmented Generation (RAG) chatbot that integrates with Wikipedia to provide intelligent, context-aware responses. Built with FastAPI, Streamlit, LangChain, and ChromaDB.

🌟 Features
Wikipedia Integration: Dynamic search and indexing of Wikipedia articles

Real-time Chat: Interactive chatbot with conversation history

Vector Search: Semantic similarity search using ChromaDB

Source Citations: All responses include source references

Modern UI: Clean Streamlit interface with responsive design

API Documentation: Auto-generated FastAPI documentation

Session Management: Multi-user support with session isolation

Docker Ready: Containerized deployment configuration

🏗️ Architecture
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │     FastAPI     │    │   Wikipedia     │
│   Frontend      │◄──►│    Backend      │◄──►│     API         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   LangChain     │◄──►│    ChromaDB     │
                       │   RAG Engine    │    │ Vector Store    │
                       └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   OpenAI API    │
                       │   (GPT + Embed) │
                       └─────────────────┘
📊 Usage
1. Index Wikipedia Content
Use the sidebar to search for topics (e.g., "artificial intelligence")

The system will automatically fetch and index relevant Wikipedia articles

View indexed content in the sidebar panel

2. Chat with the Bot
Type questions in the chat interface

Get AI-generated responses based on indexed Wikipedia content

View source citations for transparency

3. Manage Sessions
Each browser session gets a unique identifier

Chat history is maintained per session

Clear history or export conversations as needed
Project Structure:
wikipedia-rag-chatbot/
├── main.py                 # FastAPI backend entry point
├── app.py                  # Streamlit frontend
├── rag_engine.py          # Core RAG logic
├── wikipedia_retriever.py # Wikipedia API integration
├── models.py              # Pydantic data models
├── utils.py               # Utility functions
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables
├── Dockerfile            # Container configuration
├── docker-compose.yml    # Multi-service deployment
└── README.md             # This file
Key Components
FastAPI Backend: RESTful API with automatic documentation

Streamlit Frontend: Interactive web interface

LangChain: Document processing and RAG pipeline

ChromaDB: Vector database for embeddings

Wikipedia API: Content source integration

OpenAI: LLM and embedding generation

API Endpoints
POST /search - Search and index Wikipedia articles

POST /chat - Send chat messages

GET /sessions/{session_id}/history - Get chat history

DELETE /sessions/{session_id} - Clear session

GET /vector-store/stats - Database statistics