# Script Documentation

## Overview

The provided script performs web scraping, data cleaning, and relevance evaluation for content based on user queries. It leverages various libraries and tools to fetch, process, and evaluate web content.

## Functionality Breakdown

### 1. Importing Libraries

- **Utility Libraries**: 
  - `logging`, `os`, `shutil`, `re`, `time`, `hashlib`, `portalocker`
  - `shelve`: For simple persistent storage (caching).
- **HTML/XML Parsing**:
  - `BeautifulSoup` from `bs4`: Parses and cleans HTML content.
  - `etree` and `html` from `lxml`: Parses and manipulates HTML/XML content.
- **Web Search**:
  - `search` from `googlesearch`: Performs Google search queries.
- **Language Models and Embeddings**:
  - `ollama`, `chromadb`, `HuggingFaceEmbedding`, `torch`
- **Web Scraping**:
  - `webdriver`, `Service`, `Options`, `ChromeDriverManager`
- **Tokenization**:
  - `AutoTokenizer` from `transformers`
- **Authentication**:
  - `login` from `huggingface_hub`

### 2. Configuration and Initialization

- **Authentication**: Authenticates with Hugging Face using a token.
- **Device Setup**: Specifies the device for Torch operations (`cpu`).
- **Configurable Parameters**: A dictionary `CONFIG` defines various parameters like model names, cache files, directories, etc.
- **Tokenizer**: Initializes the tokenizer with a pre-trained model.
- **Directories Setup**: Ensures necessary directories exist and are clean.

### 3. Logging Setup

- Configures logging to console and file with a specific format.

### 4. Cache Management

- **Opening Cache**: Opens cache files for long-term memory.
- **Saving Cache**: Saves changes to cache files with locking to prevent concurrent write issues.

### 5. Content Handling

- **Saving Raw Content**: Saves raw fetched content to the filesystem.
- **Saving Cleaned Content**: Saves cleaned and evaluated content to the filesystem.
- **Fetching Content**: Uses Selenium to fetch content from URLs.
- **Cleaning HTML**: Cleans HTML content by removing unnecessary elements.
- **Extracting Text**: Extracts readable text from cleaned HTML.
- **Cleaning Extracted Text**: Further cleans extracted text by removing extra spaces and lines.

### 6. Content Evaluation

- **Evaluating Relevance**: Uses a model to evaluate the relevance of content based on a given query.
- **Summarizing Content**: Summarizes long content to fit within model constraints.
- **Truncating Content**: Ensures content fits within the token limits of the model.

### 7. Processing URLs and Queries

- **Processing URL**: Fetches, cleans, and evaluates content from a URL.
- **Google Search**: Performs Google searches with retries and caching.
- **Vector Store Query**: Queries a vector store for relevant documents.

### 8. Main Processing Logic

- **Processing Subquestions**: Processes each subquestion by fetching and evaluating content from URLs and vector store.
- **Search and Extract**: Manages the overall workflow of processing subquestions and compiling contexts and answers.

### 9. Response Generation

- **Generating Response**: Uses a model to generate responses for given prompts.
- **Rephrasing Queries**: Generates subquestions from the main query to break down the search process.

### 10. Main Execution

- Defines the main query and starts processing.

## How to Run the Code

### Environment Setup

Ensure that ollama is installed and you have gemma2:27b-instruct-fp16 available:

Ensure you have Python installed. It is recommended to create a virtual environment:

Use pip to install the required libraries:
```sh
pip install logging os shelve hashlib shutil re time bs4 lxml googlesearch-python ollama chromadb transformers torch portalocker urllib3 selenium webdriver-manager
```
