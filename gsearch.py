import logging
import os
import shelve
import hashlib
import shutil
import re
import time
import random
from bs4 import BeautifulSoup
from lxml import etree, html
from googlesearch import search
import ollama
import chromadb
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch
import portalocker
from urllib.parse import urlparse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from transformers import AutoTokenizer
from huggingface_hub import login

# Authenticate to HuggingFace using the token
hf_token = "?"
if hf_token:
    login(token=hf_token)
else:
    logging.error("HuggingFace API token is not set.")
    exit(1)

device = torch.device("cpu")

# Configurable parameters
CONFIG = {
    "MODEL": "gemma2:27b-instruct-fp16",
    "TOKENIZER": "google/gemma-2-27b",
    "NUM_SUBQUESTIONS": 10,
    "NUM_SEARCH_RESULTS_GOOGLE": 10,
    "NUM_SEARCH_RESULTS_VECTOR": 5,
    "LOG_FILE": 'logs/app.log',
    "LLM_CACHE_FILE": 'cache/llm_cache.db',
    "GOOGLE_CACHE_FILE": 'cache/google_cache.db',
    "URL_CACHE_FILE": 'cache/url_cache.db',
    "EXTRACTED_CONTENT_DIR": 'extracted_content',
    "REQUEST_DELAY": 5,
    "INITIAL_RETRY_DELAY": 60,
    "MAX_RETRIES": 3,
    "MAX_REDIRECTS": 3,
    "VECTOR_STORE_PATH": "./vector_store",
    "VECTOR_STORE_COLLECTION": "documents",
    "EMBEDDING_MODEL": "mixedbread-ai/mxbai-embed-large-v1",
    "TEXT_SNIPPET_LENGTH": 200,
    "MAX_DOCUMENT_LENGTH": 2000,  # Maximum length of document before summarizing
    "CONTEXT_LENGTH_TOKENS": 8000  # Context length in tokens for the model
}

# Define the tokenizer outside the config map
tokenizer = AutoTokenizer.from_pretrained(CONFIG["TOKENIZER"])

RAW_CONTENT_DIR = os.path.join(CONFIG["EXTRACTED_CONTENT_DIR"], 'raw')
CLEANED_CONTENT_DIR = os.path.join(CONFIG["EXTRACTED_CONTENT_DIR"], 'cleaned')
RELEVANT_DIR = os.path.join(CLEANED_CONTENT_DIR, 'relevant')
NOT_RELEVANT_DIR = os.path.join(CLEANED_CONTENT_DIR, 'not_relevant')

for directory in [os.path.dirname(CONFIG["LOG_FILE"]), os.path.dirname(CONFIG["LLM_CACHE_FILE"]), RAW_CONTENT_DIR, RELEVANT_DIR, NOT_RELEVANT_DIR]:
    os.makedirs(directory, exist_ok=True)

def clean_directories(*dirs):
    for dir in dirs:
        for filename in os.listdir(dir):
            file_path = os.path.join(dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logging.error(f"Failed to delete {file_path}. Reason: {e}")

clean_directories(RELEVANT_DIR, NOT_RELEVANT_DIR, RAW_CONTENT_DIR)

log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
file_handler = logging.FileHandler(CONFIG["LOG_FILE"], mode='w')
file_handler.setFormatter(log_formatter)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

def open_cache(cache_file):
    try:
        return shelve.open(cache_file, writeback=True)
    except Exception as e:
        logging.error(f"Failed to open cache file: {e}")
        return None

llm_cache = open_cache(CONFIG["LLM_CACHE_FILE"])
google_cache = open_cache(CONFIG["GOOGLE_CACHE_FILE"])
url_cache = open_cache(CONFIG["URL_CACHE_FILE"])

def save_cache(cache, cache_file):
    try:
        with portalocker.Lock(cache_file + '.lock', 'w'):
            cache.sync()
    except Exception as e:
        logging.error(f"Failed to save cache: {e}")

def save_raw_content(subquestion, url, raw_content, content_type):
    filename_hash = hashlib.md5((subquestion + url).encode('utf-8')).hexdigest()
    raw_filename = f"{filename_hash}_raw.{content_type}"
    metadata_filename = f"{filename_hash}_raw_meta.txt"

    mode = 'wb' if content_type == 'pdf' else 'w'
    with open(os.path.join(RAW_CONTENT_DIR, raw_filename), mode) as f:
        f.write(raw_content)

    with open(os.path.join(RAW_CONTENT_DIR, metadata_filename), 'w', encoding='utf-8') as f:
        metadata = {
            'url': url,
            'subquestion': subquestion,
            'status': 'raw',
            'content_type': content_type
        }
        f.write(str(metadata))

def save_cleaned_content(subquestion, url, cleaned_text, is_relevant, reason, source="web", vector_metadata=None):
    filename_hash = hashlib.md5((subquestion + url).encode('utf-8')).hexdigest()
    cleaned_filename = f"{filename_hash}_cleaned.txt"
    metadata_filename = f"{filename_hash}_cleaned_meta.txt"

    directory = RELEVANT_DIR if is_relevant else NOT_RELEVANT_DIR
    with open(os.path.join(directory, cleaned_filename), 'w', encoding='utf-8') as f:
        f.write(cleaned_text)

    with open(os.path.join(directory, metadata_filename), 'w', encoding='utf-8') as f:
        metadata = {
            'url': url,
            'subquestion': subquestion,
            'status': 'relevant' if is_relevant else 'not_relevant',
            'reason': reason,
            'source': source,
            'vector_metadata': vector_metadata
        }
        f.write(str(metadata))

# List of user agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59"
]

def get_random_user_agent():
    return random.choice(USER_AGENTS)

def fetch_content_with_browser(url):
    options = Options()
    options.add_argument("--incognito")
    options.add_argument("--headless")  # Run in headless mode
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(f"user-agent={get_random_user_agent()}")
    options.add_argument("--enable-javascript")
    options.add_argument("--enable-cookies")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    
    try:
        driver.get(url)
        time.sleep(random.uniform(CONFIG["REQUEST_DELAY"], CONFIG["REQUEST_DELAY"] + 2))  # Random delay to mimic human behavior
        page_source = driver.page_source
        driver.quit()
        return page_source, 'html'
    except Exception as e:
        logging.error(f"Failed to fetch content with browser for URL {url}: {e}")
        driver.quit()
        return "", ""

def clean_html_content(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'form']):
        element.extract()

    for element in soup.find_all(attrs={"class": ["sidebar", "advertisement", "promo", "footer", "header"]}):
        element.extract()
    for element in soup.find_all(attrs={"id": ["sidebar", "advertisement", "promo", "footer", "header"]}):
        element.extract()

    cleaned_html = soup.prettify()
    try:
        tree = html.fromstring(cleaned_html)
        for element in tree.xpath('//script|//style|//header|//footer|//nav|//aside|//form'):
            element.drop_tree()

        return etree.tostring(tree, method='html', encoding='unicode')
    except Exception as e:
        logging.error(f"Error in cleaning HTML: {e}")
        return ""

def extract_text_from_html(cleaned_html):
    soup = BeautifulSoup(cleaned_html, 'html.parser')
    return soup.get_text(separator='\n')

def cleanup_extracted_text(text):
    # Remove multiple empty lines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    # Remove leading and trailing spaces from each line
    text = '\n'.join([line.strip() for line in text.split('\n')])
    # Remove multiple spaces and tabs
    text = re.sub(r'\s+', ' ', text)
    return text

def evaluate_content_relevance(content, query_context, model):
    prompt = (
        f"Given the following query context: {query_context}\n"
        f"Evaluate the relevance and trustworthiness of the provided content. Respond strictly with 'yes' or 'no', followed by a brief and concise explanation.\n\n"
        f"Content:"
    )
    truncated_prompt = truncate_content_to_fit_prompt(prompt, content)
    logging.info(f"Evaluating content relevance for query context: {query_context[:200]}...")
    response = generate_response_with_ollama(truncated_prompt, model)
    if response:
        is_relevant = response.lower().startswith("yes")
        reason = response[4:].strip()
        logging.info(f"Content relevance evaluation result for context {query_context[:200]}: {is_relevant}, Reason: {reason}")
        return is_relevant, reason
    else:
        logging.error("Failed to evaluate content relevance.")
        return False, "Evaluation failed"
    
def summarize_content(content, subquestion, model):
    prompt = (
        f"Summarize the following content focusing on key points and information relevant to answering the subquestion. Do not describe the form of the content. Be concise and specific and only say what is needed: {subquestion}\n\n"
        f"Content:"
    )
    truncated_prompt = truncate_content_to_fit_prompt(prompt, content)
    logging.info(f"Summarizing content for subquestion: {subquestion[:200]}...")
    response = generate_response_with_ollama(truncated_prompt, model)
    if response:
        logging.info(f"Content summary: {response[:200]}")
        return response.strip()
    else:
        logging.error("Failed to summarize content.")
        return content  # Return original content if summarization fails

def truncate_content_to_fit_prompt(prompt, content):
    prompt_tokens = tokenizer.encode(prompt, return_tensors='pt')
    content_tokens = tokenizer.encode(content, return_tensors='pt')
    available_tokens = CONFIG["CONTEXT_LENGTH_TOKENS"] - prompt_tokens.size(1)

    if content_tokens.size(1) > available_tokens:
        truncated_content_tokens = content_tokens[:, :available_tokens]
        truncated_content = tokenizer.decode(truncated_content_tokens[0], skip_special_tokens=True)
        truncated_prompt = f"{prompt}\n\n{truncated_content}"
        logging.info("Content has been truncated to fit the prompt.")
    else:
        truncated_prompt = f"{prompt}\n\n{content}"
        logging.info("Original complete content is used in the prompt.")

    return truncated_prompt

def process_url(subquestion, url, model):
    logging.info(f"Fetching content for URL: {url}")
    if url in url_cache:
        logging.info(f"Using cached content for URL: {url}")
        content, content_type = url_cache[url]
    else:
        content, content_type = fetch_content_with_browser(url)
        url_cache[url] = (content, content_type)
        save_cache(url_cache, CONFIG["URL_CACHE_FILE"])
    
    if content:
        save_raw_content(subquestion, url, content, content_type)
        if content_type == 'html':
            logging.info(f"Cleaning HTML content for URL: {url}")
            cleaned_html = clean_html_content(content)
            extracted_text = extract_text_from_html(cleaned_html)
        else:
            return "", ""

        cleaned_text = cleanup_extracted_text(extracted_text)
        if cleaned_text:
            if len(cleaned_text) > CONFIG["MAX_DOCUMENT_LENGTH"]:
                cleaned_text = summarize_content(cleaned_text, subquestion, model)
            logging.info(f"Evaluating content relevance for URL: {url}")
            is_relevant, reason = evaluate_content_relevance(cleaned_text, subquestion, model)
            save_cleaned_content(subquestion, url, cleaned_text, is_relevant, reason)
            # Log at least 200 characters of the cleaned text
            logging.info(f"Cleaned text for URL {url}: {cleaned_text[:200]}")
            return cleaned_text, url
        else:
            save_cleaned_content(subquestion, url, "", False, "Failed to extract cleaned text")
    return "", ""

def search_google_with_retries(query, num_results):
    for attempt in range(CONFIG["MAX_RETRIES"]):
        try:
            if query in google_cache:
                logging.info(f"Using cached Google search results for query: {query}")
                return google_cache[query]
            results = list(search(query, num_results=num_results))  # Convert generator to list
            google_cache[query] = results
            save_cache(google_cache, CONFIG["GOOGLE_CACHE_FILE"])
            return results
        except Exception as e:
            if e.response.status_code == 429:
                retry_after = CONFIG["INITIAL_RETRY_DELAY"] * (2 ** attempt)
                logging.info(f"Received 429 error. Retrying after {retry_after} seconds.")
                time.sleep(retry_after)
            else:
                logging.error(f"Failed to search for query '{query}': {e}")
                return []
    logging.error(f"Exhausted retries for query: {query}")
    return []

def query_vector_store(collection, query, top_k=5):
    embed_model = HuggingFaceEmbedding(model_name=CONFIG["EMBEDDING_MODEL"], device=device)
    embedding = embed_model.get_text_embedding(query)
    try:
        results = collection.query(query_embeddings=[embedding], n_results=top_k, include=["documents", "metadatas"])
        logging.info(f"Found {len(results['documents'])} documents in vector store for query: {query}")
        return results
    except Exception as e:
        logging.error(f"Failed to query vector store: {e}")
        return []

def process_subquestion(subquestion, model, num_search_results_google, num_search_results_vector, original_query):
    visited_urls = set()
    domain_timestamps = {}
    all_contexts = ""
    all_references = []
    urls_to_process = []
    documents_to_process = []
    subquestion_answers = []

    logging.info(f"Processing subquestion: {subquestion}")

    results = search_google_with_retries(subquestion, num_search_results_google)
    if results:
        urls_to_process.extend(results)

    if os.path.exists(CONFIG["VECTOR_STORE_PATH"]):
        vector_store_client = chromadb.PersistentClient(path=CONFIG["VECTOR_STORE_PATH"])
        collection = vector_store_client.get_collection(name=CONFIG["VECTOR_STORE_COLLECTION"])
        vector_results = query_vector_store(collection, subquestion, top_k=num_search_results_vector)
    
        for docs, metas in zip(vector_results['documents'], vector_results['metadatas']):
            for doc, meta in zip(docs, metas):
                documents_to_process.append((doc, meta))

    logging.info(f"Initial items to process: URLs = {len(urls_to_process)}, Documents = {len(documents_to_process)}")

    while urls_to_process or documents_to_process:
        if urls_to_process:
            current_url = urls_to_process.pop(0)
            domain = urlparse(current_url).netloc
            current_time = time.time()
            
            if current_url in visited_urls:
                continue

            if domain in domain_timestamps and current_time - domain_timestamps[domain] < CONFIG["REQUEST_DELAY"]:
                wait_time = CONFIG["REQUEST_DELAY"] - (current_time - domain_timestamps[domain])
                logging.info(f"Waiting for {wait_time} seconds before making another request to {domain}")
                time.sleep(wait_time)
                current_time = time.time()  # Update current time after sleep

            extracted_text, reference = process_url(subquestion, current_url, model)
            if extracted_text:
                all_contexts += f"Content from {current_url}:\n{extracted_text}\n\n"
                all_references.append(reference)
                subquestion_answers.append(f"Answer from {current_url}: {extracted_text[:500]}")
                visited_urls.add(current_url)
                domain_timestamps[domain] = current_time

        if documents_to_process:
            doc, meta = documents_to_process.pop(0)
            combined_context = f"Document: {doc}\nMetadata: {meta}"
            logging.info(f"Evaluating content relevance for document from vector store with metadata.")
            is_relevant, reason = evaluate_content_relevance(combined_context, subquestion, model)
            source = meta.get('source', 'vector_store')
            if is_relevant:
                if len(doc) > CONFIG["MAX_DOCUMENT_LENGTH"]:
                    doc = summarize_content(doc, subquestion, model)
                all_contexts += f"Content from {source}:\n{doc}\nMetadata: {meta}\n\n"
                all_references.append(source)
                subquestion_answers.append(f"Answer from vector store document: {doc[:500]}")
                filename_hash = hashlib.md5((subquestion + source).encode('utf-8')).hexdigest()
                save_cleaned_content(subquestion, source, combined_context, is_relevant, reason, source="vector_store", vector_metadata=meta)
                logging.info(f"Document from vector store: {doc[:200]}")

        time.sleep(random.uniform(CONFIG["REQUEST_DELAY"], CONFIG["REQUEST_DELAY"] + 2))  # Random delay between processing

    logging.info(f"Context gathered for subquestion '{subquestion}': {all_contexts}")
    return all_contexts, all_references, subquestion_answers

def search_and_extract(subquestions, model, num_search_results_google, num_search_results_vector, original_query):
    all_contexts = ""
    all_references = []
    all_subquestion_answers = []

    while subquestions:
        logging.info(f"Number of subquestions to process: {len(subquestions)}")
        subquestion = subquestions.pop()
        context, references, subquestion_answers = process_subquestion(subquestion, model, num_search_results_google, num_search_results_vector, original_query)

        if context:
            all_contexts += context
            all_references.extend(references)
            all_subquestion_answers.append(f"Subquestion: {subquestion}\n{subquestion_answers}")
        else:
            logging.info(f"Insufficient context, refining subquestion: {subquestion}")
            refined_subquestions = rephrase_query_to_subquestions(subquestion, model, CONFIG["NUM_SUBQUESTIONS"])
            logging.info(f"Number of refined subquestions generated: {len(refined_subquestions)}")
            subquestions.extend(refined_subquestions)

    if all_contexts:
        prompt = (
            f"Given the following context, answer the question: {original_query}\n\n"
            f"Context provided contains both factual data and opinions. Use the context to formulate a comprehensive answer to the main question.\n\n"
            f"Context:\n{all_contexts}\n\n"
            f"Subquestions and their answers:\n{all_subquestion_answers}"
        )
        truncated_prompt = truncate_content_to_fit_prompt(prompt, all_contexts)
        logging.info(f"Requesting final answer for: Input:\n{original_query}\n\nContext:\n{all_contexts}\nSubquestions and their answers:\n{all_subquestion_answers}")
        response = generate_response_with_ollama(truncated_prompt, model)
        logging.info(f"Output:\n{response}")

def generate_response_with_ollama(prompt, model):
    if prompt in llm_cache:
        logging.info(f"Using cached response for prompt: {prompt[:100]}...")
        return llm_cache[prompt]
    try:
        response = ollama.generate(model=model, prompt=prompt)
        response_content = response.get('response', "")
        if not response_content:
            logging.error(f"Unexpected response structure: {response}")
            return ""

        llm_cache[prompt] = response_content
        save_cache(llm_cache, CONFIG["LLM_CACHE_FILE"])
        return response_content
    except Exception as e:
        logging.error(f"Failed to generate response with Ollama: {e}")
        return ""

def rephrase_query_to_subquestions(query, model, num_subquestions):
    prompt = (
        f"Given the following main question: {query}\n\n"
        f"Generate {num_subquestions} concise subquestions that can be used in a Google search query to find pages likely containing relevant information to answer the subquestion or main question. "
        f"Include sufficient information and keywords to make the answer likely relevant to the main question. "
        f"Ensure each subquestion is self-contained and does not reference information not available in the subquestion itself. "
        f"Only reply with the subquestions, each on a new line without using a list format."
    )
    logging.info(f"Requesting rephrased subquestions for query: {query}")
    response = generate_response_with_ollama(prompt, model)
    if response:
        subquestions = response.split('\n')
        unique_subquestions = list(set([sq.strip() for sq in subquestions if sq.strip()]))
        return unique_subquestions
    else:
        logging.error("Failed to generate subquestions.")
        return []

if __name__ == "__main__":
    original_query = "Suggest several suitable level 1 to level 3 spells for a Dungeons & Dragons 5th Edition Eldritch Knight Elf who specializes in ranged combat. The character has a Dexterity score of 20 and an Intelligence score of 16. Please recommend spells exclusively from officially published D&D sources, such as the Player's Handbook, Xanathar's Guide to Everything, Tasha's Cauldron of Everything, and the Dungeon Master's Guide. Do not include any homebrew or unofficial content in your suggestions."

    logging.info(f"Starting script with original query: {original_query}")
    subquestions = rephrase_query_to_subquestions(original_query, CONFIG["MODEL"], CONFIG["NUM_SUBQUESTIONS"])
    if subquestions:
        search_and_extract(subquestions, CONFIG["MODEL"], CONFIG["NUM_SEARCH_RESULTS_GOOGLE"], CONFIG["NUM_SEARCH_RESULTS_VECTOR"], original_query)
    logging.info("Script completed.")
    if llm_cache:
        llm_cache.close()
    if google_cache:
        google_cache.close()
    if url_cache:
        url_cache.close()
