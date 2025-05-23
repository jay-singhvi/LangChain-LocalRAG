# Web Scrapping and Local RAG System Setup Guide

## Quick Setup

### 1. Install Dependencies
```bash
# Install Python packages
pip install -r requirements_fixed.txt

# Install ChromeDriver for JavaScript scraping
# Option A: System package manager (Linux)
sudo apt-get install chromium-browser chromium-chromedriver

# Option B: Download manually
# Go to https://chromedriver.chromium.org/ and download for your system
```

### 2. Install Ollama
```bash
# Download and install Ollama from https://ollama.ai/

# Pull required models
ollama pull llama3:8b
ollama pull nomic-embed-text
```

### 3. Test the System
```bash
# Test with a simple website
python fixed_command_line_tool.py --crawl --url "https://example.com" --breadth 3 --depth 1 --index-path test_index

# Ask a question
python fixed_command_line_tool.py --load --index-path test_index --question "What is this website about?"
```

## Your Amazon Jobs Use Case

### Crawl Amazon Jobs
```bash
python fixed_command_line_tool.py --crawl \
  --url "https://www.amazon.jobs/en/search?offset=0&result_limit=10&sort=relevant&job_type%5B%5D=Full-Time&country%5B%5D=USA&distanceType=Mi&radius=24km&is_manager%5B%5D=0" \
  --breadth -1 \
  --depth 3 \
  --index-path amazon_jobs_docs
```

### Query for Jobs
```bash
# Interactive mode
python fixed_command_line_tool.py --load --index-path amazon_jobs_docs

# Single question
python fixed_command_line_tool.py --load --index-path amazon_jobs_docs --question "What software engineer jobs are available in Seattle?"
```

## Key Features

✅ **JavaScript Support**: Uses Selenium to handle dynamically loaded content  
✅ **Local Everything**: No cloud dependencies, uses Ollama for LLM and embeddings  
✅ **Configurable Crawling**: Set breadth (-1 for unlimited) and depth  
✅ **Persistent Storage**: FAISS vector database saved locally  
✅ **Command Line Interface**: Easy to use and automate  

## Troubleshooting

### ChromeDriver Issues
If you get ChromeDriver errors, install webdriver-manager:
```bash
pip install webdriver-manager
```

Then modify `web_scraper.py` to use auto-downloading:
```python
from webdriver_manager.chrome import ChromeDriverManager
# In JavaScriptWebScraper.__enter__():
self.driver = webdriver.Chrome(ChromeDriverManager().install(), options=chrome_options)
```

### Ollama Connection Issues
Make sure Ollama is running:
```bash
ollama serve
```

### No JavaScript Needed?
For static sites, disable JavaScript for faster crawling:
```bash
python fixed_command_line_tool.py --crawl --url "..." --no-javascript
```

## Architecture

```
URL → WebCrawler → Documents → RAGSystem → FAISS Index
                                     ↓
Question → RAGSystem → Ollama LLM → Answer
```

The system handles JavaScript-rendered content by:
1. Detecting dynamic URLs (search, jobs, etc.)
2. Using Selenium WebDriver to render pages
3. Extracting content after JavaScript execution
4. Falling back to basic HTTP for static content
