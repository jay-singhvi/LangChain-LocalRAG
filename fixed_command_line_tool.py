#!/usr/bin/env python3
"""
Enhanced Local RAG System - Command Line Interface

This tool crawls websites (including JavaScript-rendered content) and creates a local RAG system
for question answering using Ollama LLM and local embeddings.
"""

import argparse
import os
import logging
from typing import List

# Import our enhanced scraper
from web_scraper import WebCrawler, RAGSystem

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_index(url: str, breadth: int, depth: int, index_path: str, 
                llm_model: str = "llama3:8b", embedding_model: str = "nomic-embed-text",
                use_javascript: bool = True) -> RAGSystem:
    """Create a new index by crawling the given URL"""
    logger.info(f"Starting crawl of {url}")
    logger.info(f"Parameters: breadth={breadth}, depth={depth}, javascript={use_javascript}")
    
    # Initialize crawler
    crawler = WebCrawler(
        start_url=url,
        breadth=breadth,
        depth=depth,
        use_javascript=use_javascript
    )
    
    # Crawl and collect documents
    documents = crawler.crawl()
    
    if not documents:
        raise ValueError("No documents were collected during crawling. Please check the URL and parameters.")
    
    logger.info(f"Collected {len(documents)} documents")
    
    # Create RAG system
    rag_system = RAGSystem(
        documents=documents,
        llm_model=llm_model,
        embedding_model=embedding_model
    )
    
    # Save index
    rag_system.save_index(index_path)
    logger.info(f"Index saved to {index_path}")
    
    return rag_system


def load_existing_index(index_path: str, llm_model: str = "llama3:8b", 
                       embedding_model: str = "nomic-embed-text") -> RAGSystem:
    """Load an existing index"""
    logger.info(f"Loading index from {index_path}")
    
    # Create RAG system with empty documents (will be loaded from index)
    rag_system = RAGSystem([], llm_model=llm_model, embedding_model=embedding_model)
    rag_system.load_index(index_path)
    
    return rag_system


def interactive_mode(rag_system: RAGSystem):
    """Run interactive Q&A session"""
    print("\n" + "="*60)
    print("🤖 LOCAL RAG SYSTEM - Interactive Mode")
    print("="*60)
    print("Ask questions about the crawled content.")
    print("Type 'exit', 'quit', or 'q' to stop.")
    print("="*60)
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['exit', 'quit', 'q']:
                break
            
            if not question:
                print("Please enter a question.")
                continue
            
            print("\n🤔 Thinking...")
            answer = rag_system.query(question)
            print(f"\n💡 Answer: {answer}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    print("\n👋 Goodbye!")


def main():
    parser = argparse.ArgumentParser(
        description="Local RAG System with JavaScript Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Crawl Amazon jobs with JavaScript support
  python %(prog)s --crawl --url "https://www.amazon.jobs/en/search?offset=0&result_limit=10" --breadth -1 --depth 3 --index-path amazon_jobs_docs

  # Load existing index and ask questions
  python %(prog)s --load --index-path amazon_jobs_docs --question "What software engineer jobs are available?"

  # Interactive mode
  python %(prog)s --load --index-path amazon_jobs_docs
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--crawl", action="store_true", 
                           help="Crawl a website and create new index")
    mode_group.add_argument("--load", action="store_true", 
                           help="Load existing index")
    
    # Crawling options
    crawl_group = parser.add_argument_group("Crawling options")
    crawl_group.add_argument("--url", type=str, 
                            help="URL to crawl (required with --crawl)")
    crawl_group.add_argument("--breadth", type=int, default=5,
                            help="Max links per page (default: 5, -1 for unlimited)")
    crawl_group.add_argument("--depth", type=int, default=3,
                            help="Max depth to crawl (default: 3)")
    crawl_group.add_argument("--no-javascript", action="store_true",
                            help="Disable JavaScript rendering (faster but misses dynamic content)")
    
    # Common options
    parser.add_argument("--index-path", type=str, default="faiss_index",
                       help="Path to save/load the index (default: faiss_index)")
    parser.add_argument("--llm-model", type=str, default="llama3:8b",
                       help="Ollama LLM model name (default: llama3:8b)")
    parser.add_argument("--embedding-model", type=str, default="nomic-embed-text",
                       help="Ollama embedding model name (default: nomic-embed-text)")
    parser.add_argument("--question", type=str,
                       help="Ask a single question and exit")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize RAG system
        if args.crawl:
            if not args.url:
                parser.error("--url is required when using --crawl")
            
            print(f"🕷️  Starting web crawl...")
            print(f"   URL: {args.url}")
            print(f"   Breadth: {args.breadth} (-1 = unlimited)")
            print(f"   Depth: {args.depth}")
            print(f"   JavaScript: {'No' if args.no_javascript else 'Yes'}")
            print(f"   Index path: {args.index_path}")
            
            rag_system = create_index(
                url=args.url,
                breadth=args.breadth,
                depth=args.depth,
                index_path=args.index_path,
                llm_model=args.llm_model,
                embedding_model=args.embedding_model,
                use_javascript=not args.no_javascript
            )
            print("✅ Crawling and indexing complete!")
            
        elif args.load:
            if not os.path.exists(args.index_path):
                parser.error(f"Index path '{args.index_path}' does not exist. Use --crawl to create it.")
            
            print(f"📂 Loading existing index from {args.index_path}...")
            rag_system = load_existing_index(
                index_path=args.index_path,
                llm_model=args.llm_model,
                embedding_model=args.embedding_model
            )
            print("✅ Index loaded successfully!")
        
        # Handle queries
        if args.question:
            print(f"\n❓ Question: {args.question}")
            print("🤔 Thinking...")
            answer = rag_system.query(args.question)
            print(f"\n💡 Answer: {answer}")
        else:
            interactive_mode(rag_system)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Operation interrupted by user.")
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())