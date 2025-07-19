import os
import hashlib
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import re
from functools import wraps

logger = logging.getLogger(__name__)

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[]
    )
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)

def validate_environment():
    """Validate required environment variables"""
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    logger.info("Environment validation successful")

def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def generate_session_id() -> str:
    """Generate a unique session ID"""
    timestamp = str(int(time.time()))
    random_data = os.urandom(8).hex()
    return f"session_{timestamp}_{random_data}"

def hash_content(content: str) -> str:
    """Generate hash for content deduplication"""
    return hashlib.md5(content.encode()).hexdigest()

def extract_urls_from_text(text: str) -> List[str]:
    """Extract URLs from text content"""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, text)

def format_response_time(start_time: float) -> str:
    """Format response time for logging"""
    elapsed = time.time() - start_time
    if elapsed < 1:
        return f"{elapsed*1000:.0f}ms"
    else:
        return f"{elapsed:.2f}s"

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations"""
    # Remove or replace unsafe characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')
    
    # Ensure filename is not empty
    if not filename:
        filename = "unnamed"
    
    return filename

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def merge_metadata(base_metadata: Dict[str, Any], additional_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Merge metadata dictionaries with additional data taking precedence"""
    merged = base_metadata.copy()
    merged.update(additional_metadata)
    return merged

def validate_query(query: str, min_length: int = 1, max_length: int = 1000) -> bool:
    """Validate user query parameters"""
    if not query or not isinstance(query, str):
        return False
    
    query = query.strip()
    
    if len(query) < min_length or len(query) > max_length:
        return False
    
    return True

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text for search optimization"""
    # Simple keyword extraction - remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'this', 'that', 'these', 'those', 'is', 'are', 
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 
        'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might'
    }
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # Filter out stop words and get unique keywords
    keywords = list(set(word for word in words if word not in stop_words))
    
    # Sort by length (longer words first) and return top N
    keywords.sort(key=len, reverse=True)
    
    return keywords[:max_keywords]

def timing_decorator(func):
    """Decorator to log function execution time"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = format_response_time(start_time)
            logger.debug(f"{func.__name__} completed in {execution_time}")
            return result
        except Exception as e:
            execution_time = format_response_time(start_time)
            logger.error(f"{func.__name__} failed after {execution_time}: {e}")
            raise
    return wrapper

def retry_decorator(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry function calls with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = delay * (2 ** attempt)
                        logger.warning(f"{func.__name__} attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries + 1} attempts: {e}")
            
            raise last_exception
        return wrapper
    return decorator

class PerformanceMonitor:
    """Simple performance monitoring utility"""
    
    def __init__(self):
        self.metrics = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.metrics[operation] = {"start": time.time()}
    
    def end_timer(self, operation: str):
        """End timing an operation and calculate duration"""
        if operation in self.metrics and "start" in self.metrics[operation]:
            duration = time.time() - self.metrics[operation]["start"]
            self.metrics[operation]["duration"] = duration
            self.metrics[operation]["end"] = time.time()
            return duration
        return None
    
    def get_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get all performance metrics"""
        return self.metrics
    
    def log_metrics(self):
        """Log performance metrics"""
        for operation, data in self.metrics.items():
            if "duration" in data:
                logger.info(f"Operation '{operation}' took {format_response_time(data['start'])}")

class ConfigManager:
    """Configuration management utility"""
    
    def __init__(self):
        self.config = {
            "chunk_size": int(os.getenv("CHUNK_SIZE", "1000")),
            "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "200")),
            "max_retrieval_docs": int(os.getenv("MAX_RETRIEVAL_DOCS", "3")),
            "temperature": float(os.getenv("TEMPERATURE", "0.7")),
            "max_tokens": int(os.getenv("MAX_TOKENS", "2000")),
            "wikipedia_lang": os.getenv("WIKIPEDIA_LANG", "en"),
            "vector_store_path": os.getenv("VECTOR_STORE_PATH", "./chroma_db"),
        }
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        self.config[key] = value
    
    def validate_config(self):
        """Validate configuration values"""
        if self.config["chunk_size"] <= 0:
            raise ValueError("chunk_size must be positive")
        
        if self.config["chunk_overlap"] < 0:
            raise ValueError("chunk_overlap cannot be negative")
        
        if self.config["temperature"] < 0 or self.config["temperature"] > 1:
            raise ValueError("temperature must be between 0 and 1")
        
        logger.info("Configuration validation successful")

# Global instances
config_manager = ConfigManager()
performance_monitor = PerformanceMonitor()

# Initialize utilities
def initialize_utils():
    """Initialize utility components"""
    setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    validate_environment()
    config_manager.validate_config()
    logger.info("Utilities initialized successfully")

if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test text cleaning
    test_text = "  This is    a test   text!!! @#$%  "
    cleaned = clean_text(test_text)
    print(f"Cleaned text: '{cleaned}'")
    
    # Test session ID generation
    session_id = generate_session_id()
    print(f"Generated session ID: {session_id}")
    
    # Test keyword extraction
    sample_text = "Artificial intelligence and machine learning are transforming modern technology"
    keywords = extract_keywords(sample_text)
    print(f"Extracted keywords: {keywords}")
    
    print("All tests completed!")