import streamlit as st
import requests
import json
from typing import List, Dict, Any
import uuid
import time

# Configure Streamlit page
st.set_page_config(
    page_title="Wikipedia RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_BASE_URL = "http://localhost:8000"

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "indexed_topics" not in st.session_state:
    st.session_state.indexed_topics = []

def make_api_request(endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
    """Make API request to the backend"""
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "POST":
            response = requests.post(url, json=data)
        else:
            response = requests.get(url)
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return {"error": str(e)}

def search_and_index_wikipedia(query: str, max_results: int = 5) -> bool:
    """Search Wikipedia and index the results"""
    with st.spinner(f"Searching Wikipedia for '{query}' and indexing results..."):
        data = {
            "query": query,
            "max_results": max_results
        }
        
        result = make_api_request("/search", method="POST", data=data)
        
        if "error" not in result:
            st.session_state.indexed_topics.append({
                "query": query,
                "results": result.get("results", []),
                "total_indexed": result.get("total_indexed", 0),
                "timestamp": time.time()
            })
            return True
        
        return False

def send_chat_message(message: str) -> Dict:
    """Send chat message to the RAG system"""
    data = {
        "message": message,
        "session_id": st.session_state.session_id,
        "use_history": True
    }
    
    return make_api_request("/chat", method="POST", data=data)

def get_vector_store_stats() -> Dict:
    """Get vector store statistics"""
    return make_api_request("/vector-store/stats")

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("📚 Wikipedia RAG Chatbot")
    st.markdown("Chat with Wikipedia articles using Retrieval-Augmented Generation")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Wikipedia Search Section
        st.subheader("📖 Index Wikipedia Topics")
        search_query = st.text_input(
            "Enter a topic to search and index:",
            placeholder="e.g., artificial intelligence, quantum computing"
        )
        
        max_results = st.slider(
            "Maximum articles to index:",
            min_value=1,
            max_value=10,
            value=3
        )
        
        if st.button("🔍 Search & Index", type="primary"):
            if search_query:
                success = search_and_index_wikipedia(search_query, max_results)
                if success:
                    st.success(f"Successfully indexed articles for '{search_query}'!")
                else:
                    st.error("Failed to index articles. Please try again.")
            else:
                st.warning("Please enter a search query.")
        
        # Indexed Topics Display
        st.subheader("📋 Indexed Topics")
        if st.session_state.indexed_topics:
            for i, topic in enumerate(st.session_state.indexed_topics):
                with st.expander(f"📌 {topic['query']} ({topic['total_indexed']} articles)"):
                    for result in topic['results']:
                        st.write(f"**{result['title']}**")
                        st.write(f"Chunks: {result['chunks_created']}")
                        if result['url']:
                            st.write(f"[View on Wikipedia]({result['url']})")
                        st.write("---")
        else:
            st.info("No topics indexed yet. Search for a topic to get started!")
        
        # Vector Store Stats
        st.subheader("📊 Database Stats")
        if st.button("🔄 Refresh Stats"):
            stats = get_vector_store_stats()
            if "error" not in stats:
                st.metric("Total Documents", stats.get("total_documents", 0))
                st.write(f"Collection: {stats.get('collection_name', 'N/A')}")
                st.write(f"Embedding Model: {stats.get('embedding_model', 'N/A')}")
            else:
                st.error("Failed to get stats")
        
        # Session Management
        st.subheader("🗃️ Session")
        st.write(f"Session ID: `{st.session_state.session_id[:8]}...`")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
            st.rerun()
    
    # Main Chat Interface
    st.header("💬 Chat Interface")
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message["type"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
                    
                    # Display sources if available
                    if "sources" in message and message["sources"]:
                        with st.expander("📚 Sources"):
                            for j, source in enumerate(message["sources"], 1):
                                st.write(f"**{j}. {source['title']}**")
                                if source.get('url'):
                                    st.write(f"[View Article]({source['url']})")
                                st.write(f"*Excerpt:* {source['chunk_content']}")
                                st.write("---")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about the indexed Wikipedia articles..."):
        # Check if there are indexed topics
        if not st.session_state.indexed_topics:
            st.warning("⚠️ Please index some Wikipedia topics first using the sidebar!")
            st.stop()
        
        # Add user message to chat history
        st.session_state.chat_history.append({
            "type": "user",
            "content": prompt
        })
        
        # Display user message immediately
        st.chat_message("user").write(prompt)
        
        # Get response from RAG system
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = send_chat_message(prompt)
            
            if "error" not in response:
                assistant_message = response.get("response", "Sorry, I couldn't generate a response.")
                sources = response.get("sources", [])
                
                st.write(assistant_message)
                
                # Display sources
                if sources:
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(sources, 1):
                            st.write(f"**{i}. {source['title']}**")
                            if source.get('url'):
                                st.write(f"[View Article]({source['url']})")
                            st.write(f"*Excerpt:* {source['chunk_content']}")
                            st.write("---")
                
                # Add assistant message to chat history
                st.session_state.chat_history.append({
                    "type": "assistant",
                    "content": assistant_message,
                    "sources": sources
                })
            else:
                error_message = "Sorry, I encountered an error while processing your request."
                st.error(error_message)
                st.session_state.chat_history.append({
                    "type": "assistant",
                    "content": error_message
                })
        
        # Rerun to update the chat display
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>📚 Wikipedia RAG Chatbot - Powered by LangChain, OpenAI, and ChromaDB</p>
            <p>Built with ❤️ using Streamlit and FastAPI</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Additional utility functions
def export_chat_history():
    """Export chat history as JSON"""
    if st.session_state.chat_history:
        chat_data = {
            "session_id": st.session_state.session_id,
            "timestamp": time.time(),
            "messages": st.session_state.chat_history
        }
        return json.dumps(chat_data, indent=2)
    return None

# Add export functionality in sidebar
def add_export_option():
    """Add export option to sidebar"""
    with st.sidebar:
        st.subheader("💾 Export")
        if st.button("📥 Export Chat History"):
            exported_data = export_chat_history()
            if exported_data:
                st.download_button(
                    label="📁 Download JSON",
                    data=exported_data,
                    file_name=f"chat_history_{st.session_state.session_id[:8]}.json",
                    mime="application/json"
                )
            else:
                st.info("No chat history to export")

if __name__ == "__main__":
    main()
    add_export_option()
