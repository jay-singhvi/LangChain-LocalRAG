import os
import logging
import time
from typing import List, Set
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# LangChain imports
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore

# Browser automation for JavaScript content
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class JavaScriptWebScraper:
    """Web scraper that handles JavaScript-rendered content using Selenium"""
    
    def __init__(self, headless=True):
        self.headless = headless
        self.driver = None
        
    def __enter__(self):
        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium not available. Install with: pip install selenium")
        
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            return self
        except Exception as e:
            logger.warning(f"Chrome driver failed: {e}. Trying with default browser.")
            self.driver = webdriver.Chrome(options=chrome_options)
            return self
            
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.driver:
            self.driver.quit()
            
    def get_page_content(self, url: str, wait_for_element: str = None, timeout: int = 10) -> str:
        """Get page content after JavaScript execution"""
        try:
            self.driver.get(url)
            
            # Wait for specific element if provided
            if wait_for_element:
                WebDriverWait(self.driver, timeout).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, wait_for_element))
                )
            else:
                # Default wait for body
                WebDriverWait(self.driver, timeout).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
            
            # Additional wait for dynamic content
            time.sleep(2)
            
            return self.driver.page_source
        except Exception as e:
            logger.error(f"Error getting page content for {url}: {e}")
            return ""


class WebCrawler:
    """Enhanced web crawler that handles both static and JavaScript content"""
    
    def __init__(self, start_url: str, breadth: int, depth: int, use_javascript: bool = True):
        self.start_url = start_url
        self.breadth = breadth if breadth != -1 else float('inf')
        self.depth = depth
        self.use_javascript = use_javascript and SELENIUM_AVAILABLE
        self.visited_urls: Set[str] = set()
        self.documents: List[Document] = []
        
        # Parse base domain
        parsed_url = urlparse(start_url)
        self.base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        logger.info(f"Initialized crawler - Breadth: {breadth}, Depth: {depth}, JavaScript: {self.use_javascript}")
        
    def is_valid_url(self, url: str) -> bool:
        """Check if URL should be crawled"""
        if not url or url in self.visited_urls:
            return False
            
        # Skip non-HTTP URLs
        if not url.startswith(('http://', 'https://')):
            return False
            
        # Skip binary files
        skip_extensions = ['.pdf', '.jpg', '.png', '.gif', '.mp4', '.zip', '.exe']
        if any(url.lower().endswith(ext) for ext in skip_extensions):
            return False
            
        # Stay within same domain
        return url.startswith(self.base_domain)
    
    def extract_links(self, html: str, base_url: str) -> List[str]:
        """Extract links from HTML content"""
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            if not href.startswith(('http://', 'https://')):
                href = urljoin(base_url, href)
            
            if self.is_valid_url(href):
                links.append(href)
        
        # Apply breadth limit
        if isinstance(self.breadth, int):
            return links[:self.breadth]
        return links
    
    def get_page_content(self, url: str) -> tuple:
        """Get page content and extract text"""
        try:
            if self.use_javascript and self._needs_javascript(url):
                logger.info(f"Using JavaScript scraper for: {url}")
                with JavaScriptWebScraper() as scraper:
                    html_content = scraper.get_page_content(url)
            else:
                logger.info(f"Using basic HTTP request for: {url}")
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                html_content = response.text
            
            if html_content:
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Extract text content
                for script in soup(["script", "style"]):
                    script.extract()
                
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                return text, html_content
            
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
        
        return "", ""
    
    def _needs_javascript(self, url: str) -> bool:
        """Determine if URL likely needs JavaScript rendering"""
        js_indicators = ['search', 'jobs', 'results', 'api', 'app', 'single-page']
        url_lower = url.lower()
        return any(indicator in url_lower for indicator in js_indicators)
    
    def crawl(self) -> List[Document]:
        """Main crawling method"""
        url_queue = [(self.start_url, 0)]
        
        with tqdm(desc="Crawling pages") as pbar:
            while url_queue:
                current_url, current_depth = url_queue.pop(0)
                
                if current_url in self.visited_urls or current_depth > self.depth:
                    continue
                
                self.visited_urls.add(current_url)
                logger.info(f"Crawling: {current_url} (depth: {current_depth})")
                
                # Get page content
                text_content, html_content = self.get_page_content(current_url)
                
                if text_content:
                    # Create document
                    doc = Document(
                        page_content=text_content,
                        metadata={
                            "source": current_url,
                            "depth": current_depth,
                            "title": self._extract_title(html_content)
                        }
                    )
                    self.documents.append(doc)
                    pbar.update(1)
                    logger.info(f"Scraped {len(self.documents)} documents so far")
                
                # Extract links for next depth level
                if current_depth < self.depth and html_content:
                    links = self.extract_links(html_content, current_url)
                    for link in links:
                        if link not in self.visited_urls:
                            url_queue.append((link, current_depth + 1))
        
        logger.info(f"Crawling complete. Collected {len(self.documents)} documents from {len(self.visited_urls)} pages.")
        return self.documents
    
    def _extract_title(self, html: str) -> str:
        """Extract page title"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            title_tag = soup.find('title')
            return title_tag.text.strip() if title_tag else "No Title"
        except:
            return "No Title"


class RAGSystem:
    """Local RAG system using Ollama"""
    
    def __init__(self, documents: List[Document], llm_model: str = "llama3:8b", embedding_model: str = "nomic-embed-text"):
        self.documents = documents
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        
        # Initialize embeddings
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        
        # Initialize vector store
        self._init_vector_store()
        
        # Initialize LLM
        self.llm = ChatOllama(model=llm_model)
        
        # Initialize retrieval chain
        self._init_retrieval_chain()
    
    def _init_vector_store(self):
        """Initialize FAISS vector store"""        
        # Create FAISS index
        embedding_size = len(self.embeddings.embed_query("test"))
        index = faiss.IndexFlatL2(embedding_size)
        
        self.vector_store = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        
        # Add documents if available
        if self.documents:
            self.vector_store.add_documents(self.documents)
            logger.info(f"Added {len(self.documents)} documents to vector store")
        else:
            logger.warning("No documents provided - vector store created empty")
    
    def _init_retrieval_chain(self):
        """Initialize retrieval chain"""
        if not hasattr(self, 'vector_store'):
            logger.warning("Vector store not initialized yet")
            return
            
        template = """You are a helpful assistant. Use the following context to answer the question.
If you don't know the answer based on the context, say so.

Context:
{context}

Question: {question}

Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        self.document_chain = create_stuff_documents_chain(self.llm, prompt)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        
        self.retrieval_chain = RunnablePassthrough.assign(
            context=lambda x: self.retriever.invoke(x["question"])
        ).assign(
            answer=self.document_chain
        )
    
    def save_index(self, path: str):
        """Save vector store to disk"""
        self.vector_store.save_local(path)
        logger.info(f"Index saved to {path}")
    
    def load_index(self, path: str):
        """Load vector store from disk"""
        if os.path.exists(path):
            self.vector_store = FAISS.load_local(
                path, self.embeddings, allow_dangerous_deserialization=True
            )
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
            self._init_retrieval_chain()
            logger.info(f"Index loaded from {path}")
        else:
            raise FileNotFoundError(f"Index not found at {path}")
    
    def query(self, question: str) -> str:
        """Query the RAG system"""
        result = self.retrieval_chain.invoke({"question": question})
        answer = result["answer"]
        return answer.content if hasattr(answer, 'content') else str(answer)


def main():
    """Example usage"""
    # Test with a simple website
    url = "https://example.com"
    crawler = WebCrawler(url, breadth=5, depth=2)
    documents = crawler.crawl()
    
    if documents:
        rag_system = RAGSystem(documents)
        rag_system.save_index("test_index")
        
        # Test query
        answer = rag_system.query("What is this website about?")
        print(f"Answer: {answer}")
    else:
        print("No documents collected")


if __name__ == "__main__":
    main()