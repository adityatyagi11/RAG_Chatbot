import wikipedia
import asyncio
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class WikipediaRetriever:
    """Wikipedia API wrapper for content retrieval"""
    
    def __init__(self, language: str = "en", max_chars: int = 4000):
        """
        Initialize Wikipedia retriever
        
        Args:
            language: Wikipedia language code (default: "en")
            max_chars: Maximum characters per article (default: 4000)
        """
        self.language = language
        self.max_chars = max_chars
        wikipedia.set_lang(language)
    
    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search Wikipedia articles and return detailed content
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of dictionaries containing article information
        """
        try:
            # Search for article titles
            search_results = wikipedia.search(query, results=max_results)
            articles = []
            
            for title in search_results[:max_results]:
                try:
                    article_data = await self._get_article_content(title)
                    if article_data:
                        articles.append(article_data)
                except Exception as e:
                    logger.warning(f"Failed to retrieve article '{title}': {e}")
                    continue
            
            return articles
            
        except Exception as e:
            logger.error(f"Error searching Wikipedia: {e}")
            return []
    
    async def _get_article_content(self, title: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed content for a specific Wikipedia article
        
        Args:
            title: Wikipedia article title
            
        Returns:
            Dictionary with article information or None if failed
        """
        try:
            # Get Wikipedia page
            page = wikipedia.page(title)
            
            # Extract content (truncate if too long)
            content = page.content
            if len(content) > self.max_chars:
                content = content[:self.max_chars] + "..."
            
            # Get summary
            summary = page.summary
            if len(summary) > 500:
                summary = summary[:500] + "..."
            
            return {
                "title": page.title,
                "url": page.url,
                "content": content,
                "summary": summary,
                "categories": getattr(page, 'categories', []),
                "links": getattr(page, 'links', [])[:10],  # First 10 links
                "images": getattr(page, 'images', [])[:5],  # First 5 images
                "language": self.language
            }
            
        except wikipedia.exceptions.DisambiguationError as e:
            # Handle disambiguation pages
            logger.info(f"Disambiguation page for '{title}', trying first option")
            try:
                if e.options:
                    return await self._get_article_content(e.options[0])
            except Exception:
                pass
            return None
            
        except wikipedia.exceptions.PageError:
            logger.warning(f"Page not found: {title}")
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving article '{title}': {e}")
            return None
    
    async def get_article_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific Wikipedia article by exact title
        
        Args:
            title: Exact Wikipedia article title
            
        Returns:
            Dictionary with article information or None if not found
        """
        return await self._get_article_content(title)
    
    async def get_random_articles(self, count: int = 1) -> List[Dict[str, Any]]:
        """
        Get random Wikipedia articles
        
        Args:
            count: Number of random articles to retrieve
            
        Returns:
            List of random article dictionaries
        """
        try:
            articles = []
            for _ in range(count):
                try:
                    random_title = wikipedia.random()
                    article_data = await self._get_article_content(random_title)
                    if article_data:
                        articles.append(article_data)
                except Exception as e:
                    logger.warning(f"Failed to get random article: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            logger.error(f"Error getting random articles: {e}")
            return []
    
    def get_page_suggestions(self, query: str, max_suggestions: int = 10) -> List[str]:
        """
        Get Wikipedia page suggestions for a query
        
        Args:
            query: Search query
            max_suggestions: Maximum number of suggestions
            
        Returns:
            List of suggested page titles
        """
        try:
            suggestions = wikipedia.search(query, results=max_suggestions)
            return suggestions
        except Exception as e:
            logger.error(f"Error getting suggestions: {e}")
            return []
    
    def set_language(self, language: str):
        """
        Change Wikipedia language
        
        Args:
            language: Language code (e.g., 'en', 'es', 'fr')
        """
        self.language = language
        wikipedia.set_lang(language)
        logger.info(f"Wikipedia language set to: {language}")

# Example usage and testing
async def test_wikipedia_retriever():
    """Test function for WikipediaRetriever"""
    retriever = WikipediaRetriever()
    
    # Test search
    print("Testing Wikipedia search...")
    results = await retriever.search("artificial intelligence", max_results=3)
    
    for i, article in enumerate(results, 1):
        print(f"\n--- Article {i} ---")
        print(f"Title: {article['title']}")
        print(f"URL: {article['url']}")
        print(f"Summary: {article['summary'][:200]}...")
        print(f"Content length: {len(article['content'])} characters")
    
    # Test specific article
    print("\n\nTesting specific article retrieval...")
    article = await retriever.get_article_by_title("Machine Learning")
    if article:
        print(f"Retrieved: {article['title']}")
        print(f"Categories: {article['categories'][:5]}")
    
    # Test suggestions
    print("\n\nTesting suggestions...")
    suggestions = retriever.get_page_suggestions("python programming")
    print(f"Suggestions: {suggestions[:5]}")

if __name__ == "__main__":
    # Run test
    asyncio.run(test_wikipedia_retriever())