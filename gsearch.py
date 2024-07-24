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
import googlesearch.user_agents
import ollama
import chromadb
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document
import torch
import portalocker
from urllib.parse import urlparse
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
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
    "MODEL": "gemma2:27b-instruct-q8_0",
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
    "MAX_DOCUMENT_LENGTH": 2000,
    "CONTEXT_LENGTH_TOKENS": 8000,
    "NUM_USER_AGENTS": 10
}

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

def save_chunk_content(subquestion, url, chunk, chunk_summary, is_relevant, reason, chunk_id):
    filename_hash = hashlib.md5((subquestion + url + str(chunk_id)).encode('utf-8')).hexdigest()
    chunk_filename = f"{filename_hash}_chunk_{chunk_id}.txt"
    chunk_summary_filename = f"{filename_hash}_chunk_{chunk_id}_summary.txt"
    metadata_filename = f"{filename_hash}_chunk_{chunk_id}_meta.txt"

    directory = RELEVANT_DIR if is_relevant else NOT_RELEVANT_DIR
    with open(os.path.join(RAW_CONTENT_DIR, chunk_filename), 'w', encoding='utf-8') as f:
        f.write(chunk)
    
    with open(os.path.join(directory, chunk_summary_filename), 'w', encoding='utf-8') as f:
        f.write(chunk_summary)

    with open(os.path.join(directory, metadata_filename), 'w', encoding='utf-8') as f:
        metadata = {
            'url': url,
            'subquestion': subquestion,
            'status': 'relevant' if is_relevant else 'not_relevant',
            'reason': reason,
            'chunk_id': chunk_id
        }
        f.write(str(metadata))

def save_cleaned_content(subquestion, url, cleaned_text, summarized_text, is_relevant, reason, source="web", vector_metadata=None):
    filename_hash = hashlib.md5((subquestion + url).encode('utf-8')).hexdigest()
    cleaned_filename = f"{filename_hash}_cleaned.txt"
    summarized_filename = f"{filename_hash}_summarized.txt"
    metadata_filename = f"{filename_hash}_cleaned_meta.txt"

    directory = RELEVANT_DIR if is_relevant else NOT_RELEVANT_DIR
    with open(os.path.join(directory, cleaned_filename), 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    
    with open(os.path.join(directory, summarized_filename), 'w', encoding='utf-8') as f:
        f.write(summarized_text)

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

def save_final_output(main_question, subquestions, contexts, answers):
    # Create directory structure based on the main question hash
    main_question_hash = hashlib.md5(main_question.encode('utf-8')).hexdigest()
    output_dir = os.path.join("output", main_question_hash)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save main question
    with open(os.path.join(output_dir, "main_question.txt"), 'w', encoding='utf-8') as f:
        f.write(main_question)
    
    # Save subquestions
    subquestions_file = os.path.join(output_dir, "subquestions.txt")
    with open(subquestions_file, 'w', encoding='utf-8') as f:
        for subquestion in subquestions:
            f.write(f"{subquestion}\n")
    
    # Save contexts and answers
    for i, (context, answer) in enumerate(zip(contexts, answers)):
        context_file = os.path.join(output_dir, f"context_{i+1}.txt")
        answer_file = os.path.join(output_dir, f"answer_{i+1}.txt")
        with open(context_file, 'w', encoding='utf-8') as f:
            f.write(context)
        with open(answer_file, 'w', encoding='utf-8') as f:
            f.write(answer)

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

def generate_user_agents(num_user_agents, model):
    prompt = (
        f"Generate {num_user_agents} realistic and commonly used user-agent strings for web browsers. "
        f"Include a variety of operating systems (e.g., Windows, macOS, Linux, Android, iOS) and browser versions "
        f"(e.g., Chrome, Firefox, Safari, Edge) to simulate different types of users. "
        f"Output each user-agent string on a new line without any additional text or formatting."
    )
    response = generate_response_with_ollama(prompt, model)
    if response:
        user_agents = [ua.strip() for ua in response.split('\n') if ua.strip()]
        return user_agents
    else:
        logging.error("Failed to generate user agents.")
        return []

class UserAgentManager:
    def __init__(self, model, num_user_agents):
        self.model = model
        self.num_user_agents = num_user_agents
        self.user_agents = []
        self.current_index = 0
        logging.info(f"Initializing UserAgentManager with model {self.model} and {self.num_user_agents} user agents.")
        self._generate_user_agents()

    def _generate_user_agents(self):
        logging.info("Generating new user agents...")
        self.user_agents = generate_user_agents(self.num_user_agents, self.model)
        self.current_index = 0
        if self.user_agents:
            logging.info(f"Generated {len(self.user_agents)} new user agents.")
        else:
            logging.error("Failed to generate new user agents.")

    def get_next_user_agent(self):
        if not self.user_agents or self.current_index >= len(self.user_agents):
            logging.warning("User agent list is empty or index out of range, regenerating user agents.")
            self._generate_user_agents()
        if self.user_agents:
            user_agent = self.user_agents[self.current_index]
            self.current_index += 1
            logging.info(f"Returning user agent: {user_agent}")
            return user_agent
        else:
            logging.error("Failed to get a new user agent. User agent list is empty after regeneration.")
            return None

user_agent_manager = UserAgentManager(CONFIG["MODEL"], CONFIG["NUM_USER_AGENTS"])

def init_browser():
    options = Options()
    options.add_argument("--incognito")
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(f"user-agent={user_agent_manager.get_next_user_agent()}")
    options.add_argument("--enable-javascript")
    options.add_argument("--enable-cookies")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver

browser = None

def ensure_browser():
    global browser
    if browser is None or not browser.service.is_connectable():
        logging.info("Initializing new browser instance.")
        if browser is not None:
            try:
                browser.quit()
            except Exception as e:
                logging.error(f"Failed to quit the browser: {e}")
        browser = init_browser()

def fetch_content_with_browser(url):
    ensure_browser()
    try:
        logging.info(f"Navigating to URL: {url}")
        browser.get(url)
        WebDriverWait(browser, CONFIG["REQUEST_DELAY"]).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        logging.info("Page loaded successfully")

        try:
            cookie_buttons = WebDriverWait(browser, 5).until(
                EC.presence_of_all_elements_located((
                    By.XPATH, 
                    "//*[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'accept') or contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'agree') or contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'proceed') or contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'allow') or contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'consent') or contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'continue')]"
                ))
            )
            for cookie_button in cookie_buttons:
                try:
                    if WebDriverWait(browser, 5).until(EC.element_to_be_clickable(cookie_button)):
                        logging.info("Found clickable cookie pop-up button, attempting to click it")
                        cookie_button.click()
                        logging.info("Clicked cookie pop-up button, waiting for it to disappear")
                        
                        WebDriverWait(browser, 5).until(EC.invisibility_of_element(cookie_button))
                        logging.info("Cookie pop-up disappeared successfully")
                except Exception as e:
                    logging.info(f"Could not click the button")
        except Exception as e:
            logging.info(f"No cookie pop-up found or could not locate buttons")

        time.sleep(random.uniform(CONFIG["REQUEST_DELAY"], CONFIG["REQUEST_DELAY"] + 2))
        page_source = browser.page_source
        logging.info("Fetched page source successfully")
        return page_source, 'html'
    except Exception as e:
        logging.error(f"Failed to fetch content with browser for URL {url}: {e}")
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
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = '\n'.join([line.strip() for line in text.split('\n')])
    text = re.sub(r'\s+', ' ', text)
    return text

def evaluate_and_summarize_content(content, query_context, subquestion, model):
    prompt = (
        f"Context: {query_context}\n\n"
        f"Content: {content}\n\n"
        f"Subquestion: {subquestion}\n\n"
        f"Determine if the provided content is relevant to the context for answering the subquestion. "
        f"Relevance should be based on specific, factual information, and direct connections to the context. "
        f"Then, if relevant, provide a concise summary of the content focusing on key points and critical insights related to the subquestion. "
        f"Respond with 'Relevant: [reason]\n\nSummary: [summary]' or 'Not Relevant: [reason]'."
    )
    logging.info(f"Evaluating relevance and summarizing content for subquestion: {subquestion[:200]}...")
    response = generate_response_with_ollama(prompt, model)
    
    if response:
        response_lower = response.lower()
        if response_lower.startswith("relevant:"):
            is_relevant = True
            split_response = response.split('\n\nSummary:')
            reason = split_response[0][response_lower.find(":") + 1:].strip()
            summary = split_response[1].strip() if len(split_response) > 1 else ""
        elif response_lower.startswith("not relevant:"):
            is_relevant = False
            reason = response[response_lower.find(":") + 1:].strip()
            summary = ""
        else:
            is_relevant = False
            reason = "Unclear response structure"
            summary = ""
        
        logging.info(f"Content relevance and summary result for context {query_context[:200]}: {is_relevant}, Reason: {reason}, Summary: {summary[:200]}")
        return is_relevant, reason, summary
    else:
        logging.error("Failed to evaluate and summarize content.")
        return False, "Evaluation failed", ""

def split_and_process_chunks(subquestion, url, text, model):
    embed_model = HuggingFaceEmbedding(model_name=CONFIG["EMBEDDING_MODEL"], device=device)
    splitter = SemanticSplitterNodeParser(chunk_size=CONFIG["TEXT_SNIPPET_LENGTH"], chunk_overlap=50, embed_model=embed_model)
    
    document = Document(text=text)
    nodes = splitter.build_semantic_nodes_from_documents([document])
    
    chunk_summaries = []
    chunk_relevance = []

    for chunk_id, node in enumerate(nodes, 1):
        chunk = node.text
        is_relevant, reason, summary = evaluate_and_summarize_content(chunk, subquestion, subquestion, model)
        if is_relevant:
            save_chunk_content(subquestion, url, chunk, summary, is_relevant, reason, chunk_id)
            chunk_summaries.append(summary)
            chunk_relevance.append(is_relevant)
    
    summarized_text = " ".join(chunk_summaries)
    is_relevant = any(chunk_relevance)
    reason = "At least one chunk is relevant" if is_relevant else "None of the chunks are relevant"

    return summarized_text, is_relevant, reason

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
            summarized_text, is_relevant, reason = split_and_process_chunks(subquestion, url, cleaned_text, model)
            save_cleaned_content(subquestion, url, cleaned_text, summarized_text, is_relevant, reason)
            logging.info(f"Cleaned text for URL {url}: {cleaned_text[:200]}")
            return summarized_text, url
        else:
            save_cleaned_content(subquestion, url, "", "", False, "Failed to extract cleaned text")
    return "", ""

def search_google_with_retries(query, num_results):
    for attempt in range(CONFIG["MAX_RETRIES"]):
        try:
            if query in google_cache:
                logging.info(f"Using cached Google search results for query: {query}")
                return google_cache[query]
            
            googlesearch.user_agents.user_agents = [user_agent_manager.get_next_user_agent()]
            results = list(search(query, num_results=num_results))
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

def generate_search_terms(subquestion, model):
    prompt = (
        f"Translate the following subquestion into a set of concise Google search terms that cover the subquestion effectively and conform to Google's search best practices. "
        f"Ensure the search terms are specific, include relevant keywords, use quotation marks for exact phrases, and use the minus sign to exclude unwanted terms. "
        f"The output should be a single line of text that can be directly used in Google search. Avoid providing explanations or additional tips. "
        f"Subquestion: {subquestion}"
    )

    logging.info(f"Generating search terms for subquestion: {subquestion}")
    response = generate_response_with_ollama(prompt, model)
    if response:
        search_terms = response.strip()
        logging.info(f"Generated search terms: {search_terms}")
        return search_terms
    else:
        logging.error("Failed to generate search terms.")
        return subquestion

def rephrase_query_to_subquestions(query, model, num_subquestions):
    prompt = (
        f"Given the following main question: {query}\n\n"
        f"Generate {num_subquestions} detailed and specific subquestions that can be used in a Google search query to find pages likely containing relevant information to answer the subquestion or the main question. "
        f"Ensure the subquestions collectively address all aspects of the main question. "
        f"Each subquestion should include enough context and keywords to make the answer relevant to the main question. "
        f"Ensure each subquestion is self-contained and does not reference information not available in the subquestion itself. "
        f"Only reply with the subquestions, each on a new line without using a list format."
    )
    logging.info(f"Requesting rephrased subquestions for query: {query}")
    response = generate_response_with_ollama(prompt, model)
    if response:
        subquestions = response.split('\n')
        unique_subquestions = list(set([sq.strip() for sq in subquestions if sq.strip()]))
        logging.info(f"Rephrased subquestions: {unique_subquestions}")
        return unique_subquestions
    else:
        logging.error("Failed to generate subquestions.")
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

    search_terms = generate_search_terms(subquestion, model)
    logging.info(f"Translated search terms: {search_terms}")

    results = search_google_with_retries(search_terms, num_search_results_google)
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
                current_time = time.time()

            extracted_text, reference = process_url(subquestion, current_url, model)
            if extracted_text:
                all_contexts += f"Content from {current_url}:\n{extracted_text}\n\n"
                all_references.append(reference)
                subquestion_answers.append(f"Answer from {current_url}: {extracted_text[:500]}")
                visited_urls.add(current_url)
                domain_timestamps[domain] = current_time

        if documents_to_process:
            doc, meta = documents_to_process.pop(0)
            logging.info(f"Evaluating content relevance for document from vector store with metadata.")
            summarized_text, is_relevant, reason = split_and_process_chunks(subquestion, meta.get('source', 'vector_store'), doc, model)
            source = meta.get('source', 'vector_store')
            
            all_contexts += f"Content from {source}:\n{summarized_text}\nMetadata: {meta}\n\n"
            all_references.append(source)
            subquestion_answers.append(f"Answer from vector store document: {summarized_text[:500]}")
            save_cleaned_content(subquestion, source, doc, summarized_text, is_relevant, reason, source="vector_store", vector_metadata=meta)
            logging.info(f"Document from vector store: {doc[:200]}")

        time.sleep(random.uniform(CONFIG["REQUEST_DELAY"], CONFIG["REQUEST_DELAY"] + 2))

    logging.info(f"Context gathered for subquestion '{subquestion}': {all_contexts}")
    logging.info(f"Answers gathered for subquestion '{subquestion}': {subquestion_answers}")
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
        logging.info(f"Requesting final answer for: Input:\n{original_query}\n\nContext:\n{all_contexts}\nSubquestions and their answers:\n{all_subquestion_answers}")
        response = generate_response_with_ollama(prompt, model)
        logging.info(f"Output:\n{response}")

        save_final_output(original_query, subquestions, all_contexts, response)
        
if __name__ == "__main__":
    original_query = (
        "I am playing as an Eldritch Knight Elf in Dungeons & Dragons 5th Edition, focusing on ranged combat. My character does not have access to homebrew spells and has a Dexterity score of 20 and an Intelligence score of 16. Please provide a list of effective level 1 to level 3 spells that would be beneficial for my character to have. Include specific details and explanations for why each spell is beneficial."
    )

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
        google_cache.close()
    if browser:
        browser.quit()
