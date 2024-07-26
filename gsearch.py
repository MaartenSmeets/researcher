import hashlib
import logging
import os
import random
import re
import shelve
import shutil
import time
from urllib.parse import urlparse

import chromadb
from bs4 import BeautifulSoup
from googlesearch import search
from googlesearch import user_agents as google_user_agents
from huggingface_hub import login
from lxml import etree, html
from ollama import generate
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from torch import device as torch_device

from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import portalocker
import json

# Configurable parameters
CONFIG = {
    # Model configuration
    "MODEL_NAME": "gemma2:27b-instruct-q8_0",  # Name of the model used for generating responses
    "NUM_INITIAL_SUBQUESTIONS": 5,  # Number of initial subquestions to generate
    "NUM_FOLLOWUP_SUBQUESTIONS": 3,  # Number of follow-up subquestions to generate when more information is needed
    "NUM_SEARCH_RESULTS_GOOGLE": 5,  # Number of search results to fetch from Google for each subquestion
    "NUM_SEARCH_RESULTS_VECTOR": 5,  # Number of search results to fetch from the vector store for each subquestion
    "EMBEDDING_MODEL_NAME": "mixedbread-ai/mxbai-embed-large-v1",  # Embedding model name used for vector store queries

    # Logging configuration
    "LOG_FILE_PATH": 'logs/app.log',  # Path to the log file

    # Cache configuration
    "LLM_CACHE_FILE_PATH": 'cache/llm_cache.db',  # File path for the LLM response cache
    "GOOGLE_CACHE_FILE_PATH": 'cache/google_cache.db',  # File path for the Google search results cache
    "URL_CACHE_FILE_PATH": 'cache/url_cache.db',  # File path for the URL content cache
    "CHUNK_CACHE_FILE_PATH": 'cache/chunk_cache.db',  # File path for the document chunk cache

    # Content extraction and processing configuration
    "EXTRACTED_CONTENT_DIRECTORY": 'extracted_content',  # Directory to store extracted content
    "RAW_CONTENT_DIRECTORY": 'extracted_content/raw',  # Directory to store raw extracted content
    "CLEANED_CONTENT_DIRECTORY": 'extracted_content/cleaned',  # Directory to store cleaned content
    "RELEVANT_CONTENT_DIRECTORY": 'extracted_content/cleaned/relevant',  # Directory to store relevant cleaned content
    "NOT_RELEVANT_CONTENT_DIRECTORY": 'extracted_content/cleaned/not_relevant',  # Directory to store not relevant cleaned content

    # Request and retry configuration
    "REQUEST_DELAY_SECONDS": 5,  # Delay between requests to avoid rate limiting
    "INITIAL_RETRY_DELAY_SECONDS": 60,  # Initial delay before retrying a failed request
    "MAX_RETRIES": 3,  # Maximum number of retries for failed requests
    "MAX_REDIRECTS": 3,  # Maximum number of redirects to follow

    # Vector store configuration
    "VECTOR_STORE_DIRECTORY": "./vector_store",  # Directory path for the vector store
    "VECTOR_STORE_COLLECTION_NAME": "documents",  # Collection name in the vector store

    # Text processing configuration
    "TEXT_SNIPPET_LENGTH": 1000,  # Length of text snippets for chunking
    "CONTEXT_LENGTH_TOKENS": 8000,  # Length of context in tokens for generating responses

    # User agent configuration
    "NUM_USER_AGENTS": 10,  # Number of user agents to generate
    "OUTPUT_DIRECTORY": "output"  # Directory to store final output
}

# Authenticate to HuggingFace using the token
hf_token = "?"
if hf_token:
    login(token=hf_token)
else:
    logging.error("HuggingFace API token is not set.")
    exit(1)

device = torch_device("cpu")

for directory in [
    os.path.dirname(CONFIG["LOG_FILE_PATH"]),
    os.path.dirname(CONFIG["LLM_CACHE_FILE_PATH"]),
    CONFIG["RAW_CONTENT_DIRECTORY"],
    CONFIG["RELEVANT_CONTENT_DIRECTORY"],
    CONFIG["NOT_RELEVANT_CONTENT_DIRECTORY"],
    CONFIG["OUTPUT_DIRECTORY"]
]:
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

clean_directories(CONFIG["RELEVANT_CONTENT_DIRECTORY"], CONFIG["NOT_RELEVANT_CONTENT_DIRECTORY"], CONFIG["RAW_CONTENT_DIRECTORY"], CONFIG["OUTPUT_DIRECTORY"])

log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
file_handler = logging.FileHandler(CONFIG["LOG_FILE_PATH"], mode='w')
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

llm_cache = open_cache(CONFIG["LLM_CACHE_FILE_PATH"])
google_cache = open_cache(CONFIG["GOOGLE_CACHE_FILE_PATH"])
url_cache = open_cache(CONFIG["URL_CACHE_FILE_PATH"])
chunk_cache = open_cache(CONFIG["CHUNK_CACHE_FILE_PATH"])

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
    with open(os.path.join(CONFIG["RAW_CONTENT_DIRECTORY"], raw_filename), mode) as f:
        f.write(raw_content)

    with open(os.path.join(CONFIG["RAW_CONTENT_DIRECTORY"], metadata_filename), 'w', encoding='utf-8') as f:
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

    directory = CONFIG["RELEVANT_CONTENT_DIRECTORY"] if is_relevant else CONFIG["NOT_RELEVANT_CONTENT_DIRECTORY"]
    with open(os.path.join(CONFIG["RAW_CONTENT_DIRECTORY"], chunk_filename), 'w', encoding='utf-8') as f:
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
    if not summarized_text.strip():
        return  # Avoid saving empty summaries
    filename_hash = hashlib.md5((subquestion + url).encode('utf-8')).hexdigest()
    cleaned_filename = f"{filename_hash}_cleaned.txt"
    summarized_filename = f"{filename_hash}_summarized.txt"
    metadata_filename = f"{filename_hash}_cleaned_meta.txt"

    directory = CONFIG["RELEVANT_CONTENT_DIRECTORY"] if is_relevant else CONFIG["NOT_RELEVANT_CONTENT_DIRECTORY"]
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

def append_to_file(output_dir, filename, content):
    with open(os.path.join(output_dir, filename), 'a', encoding='utf-8') as f:
        f.write(content + '\n')

def save_final_output(main_question, subquestions, contexts, answers):
    main_question_hash = hashlib.md5(main_question.encode('utf-8')).hexdigest()
    output_dir = os.path.join(CONFIG["OUTPUT_DIRECTORY"], main_question_hash)
    os.makedirs(output_dir, exist_ok=True)
    
    append_to_file(output_dir, "main_question.txt", main_question)
    subquestions_content = "\n".join(subquestions)
    append_to_file(output_dir, "subquestions.txt", subquestions_content)
    
    combined_contexts = "\n\n".join(contexts)
    combined_answers = "\n\n".join(answers)
    
    append_to_file(output_dir, "combined_contexts.txt", combined_contexts)
    append_to_file(output_dir, "combined_answers.txt", combined_answers)
    append_to_file(output_dir, "final_answer.txt", answers[-1])  # Save the final answer separately

def generate_response_with_ollama(prompt, model):
    if prompt in llm_cache:
        logging.info(f"Using cached response for prompt: {prompt[:100]}...")
        return llm_cache[prompt]
    try:
        response = generate(model=model, prompt=prompt)
        response_content = response.get('response', "")
        if not response_content:
            logging.error(f"Unexpected response structure: {response}")
            return ""

        llm_cache[prompt] = response_content
        save_cache(llm_cache, CONFIG["LLM_CACHE_FILE_PATH"])
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
            logging.debug(f"Returning user agent: {user_agent}")
            return user_agent
        else:
            logging.error("Failed to get a new user agent. User agent list is empty after regeneration.")
            return None

user_agent_manager = UserAgentManager(CONFIG["MODEL_NAME"], CONFIG["NUM_USER_AGENTS"])

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
        WebDriverWait(browser, CONFIG["REQUEST_DELAY_SECONDS"]).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
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
                        logging.debug("Found clickable cookie pop-up button, attempting to click it")
                        cookie_button.click()
                        logging.debug("Clicked cookie pop-up button, waiting for it to disappear")
                        WebDriverWait(browser, 5).until(EC.invisibility_of_element(cookie_button))
                        logging.info("Cookie pop-up disappeared successfully")
                except Exception as e:
                    logging.debug(f"Could not click the button: {e}")
        except Exception as e:
            logging.debug(f"No cookie pop-up found or could not locate buttons: {e}")

        time.sleep(random.uniform(CONFIG["REQUEST_DELAY_SECONDS"], CONFIG["REQUEST_DELAY_SECONDS"] + 2))
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

def generate_search_terms(subquestion, model):
    if not subquestion.strip():  # Ensure subquestion is not empty
        logging.error("Subquestion is empty, skipping search term generation.")
        return subquestion

    prompt = (
        f"Transform the following subquestion into a set of concise Google search terms that cover the subquestion effectively and conform to Google's search best practices. "
        f"Ensure the search terms are specific, include relevant keywords, use quotation marks for exact phrases, and use the minus sign to exclude unwanted terms. "
        f"The output should be a single line of text that can be directly used in Google search. Avoid providing explanations or additional tips. "
        f"Only use the provided subquestion to generate search terms and do not use any other knowledge."
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

def search_google_with_retries(query, num_results):
    for attempt in range(CONFIG["MAX_RETRIES"]):
        try:
            if query in google_cache and attempt == 0:
                logging.info(f"Using cached Google search results for query: {query}")
                return google_cache[query]
            
            google_user_agents.user_agents = [user_agent_manager.get_next_user_agent()]
            results = list(search(query, num_results=num_results))
            google_cache[query] = results
            save_cache(google_cache, CONFIG["GOOGLE_CACHE_FILE_PATH"])
            return results
        except Exception as e:
            if e.response.status_code == 429:
                retry_after = CONFIG["INITIAL_RETRY_DELAY_SECONDS"] * (2 ** attempt)
                logging.info(f"Received 429 error. Retrying after {retry_after} seconds.")
                time.sleep(retry_after)
            else:
                logging.error(f"Failed to search for query '{query}': {e}")
                return []
    logging.error(f"Exhausted retries for query: {query}")
    return []

def query_vector_store(collection, query, top_k=5):
    embed_model = HuggingFaceEmbedding(model_name=CONFIG["EMBEDDING_MODEL_NAME"], device=device)
    embedding = embed_model.get_text_embedding(query)
    try:
        results = collection.query(query_embeddings=[embedding], n_results=top_k, include=["documents", "metadatas"])
        logging.info(f"Found {len(results['documents'])} documents in vector store for query: {query}")
        return results
    except Exception as e:
        logging.error(f"Failed to query vector store: {e}")
        return []

def rephrase_query_to_initial_subquestions(query, model, num_subquestions):
    prompt = (
        f"Given the following main question: {query}\n\n"
        f"Generate {num_subquestions} detailed and specific subquestions that can be used in a Google search query to find pages likely containing relevant information to answer the subquestion or the main question. "
        f"Ensure the subquestions collectively address all aspects of the main question. "
        f"Each subquestion should include enough context and keywords to make the answer relevant to the main question. "
        f"Ensure each subquestion is self-contained and does not reference information not available in the subquestion itself. Only use the provided main question to generate subquestions and do not use any other knowledge."
        f"Only reply with the subquestions, each on a new line without using a list format."
    ) 
    response = generate_response_with_ollama(prompt, model)
    if response:
        subquestions = response.split('\n')
        unique_subquestions = list(set([sq.strip() for sq in subquestions if sq.strip() and not sq.isspace()]))
        return unique_subquestions
    else:
        logging.error("Failed to generate initial subquestions.")
        return []

def rephrase_query_to_followup_subquestions(query, model, num_subquestions, context):
    prompt = (
        f"Given the following main question: {query}\n\n"
        f"Current context: {context}\n\n"
        f"Generate {num_subquestions} detailed and specific follow-up subquestions that can help answer the main question, focusing on missing information required to complete the answer. "
        f"Each subquestion should be self-contained and does not reference information not available in the subquestion itself. Only use the provided context to generate follow-up subquestions and do not use any other knowledge."
        f"Only reply with the subquestions, each on a new line without using a list format."
    )
    response = generate_response_with_ollama(prompt, model)
    if response:
        subquestions = response.split('\n')
        unique_subquestions = list(set([sq.strip() for sq in subquestions if sq.strip() and not sq.isspace()]))
        return subquestions
    else:
        logging.error("Failed to generate follow-up subquestions.")
        return []

def parse_json_response(response, json_format, model, max_retries=3):
    error_history = set()
    for attempt in range(max_retries):
        try:
            response = response.strip('```json').strip('```').strip()
            result = json.loads(response)
            return result
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response: {e}\nResponse: {response}")
            error_message = str(e)
            if error_message in error_history:
                response = request_llm_to_fix_json_creatively(response, error_message, json_format, model)
            else:
                response = request_llm_to_fix_json(response, error_message, json_format, model)
            error_history.add(error_message)
    return None
   
def request_llm_to_fix_json(response, error, json_format, model):
    prompt = (
        f"The following JSON response contains errors:\n\n"
        f"{response}\n\n"
        f"Error details: {error}\n\n"
        f"Please correct the JSON response to match the expected format provided below and reply only with the fixed JSON. Ensure the corrected JSON can be parsed successfully.\n\n"
        f"Expected JSON format: {json_format}"
    )
    fixed_response = generate_response_with_ollama(prompt, model)
    return fixed_response

def request_llm_to_fix_json_creatively(response, error, json_format, model):
    prompt = (
        f"The following JSON response contains errors and previous attempts to fix it have failed:\n\n"
        f"{response}\n\n"
        f"Error details: {error}\n\n"
        f"Please correct the JSON response creatively to match the expected format provided below and reply only with the fixed JSON. Ensure the corrected JSON can be parsed successfully and consider different approaches.\n\n"
        f"Expected JSON format: {json_format}"
    )
    fixed_response = generate_response_with_ollama(prompt, model)
    return fixed_response

def evaluate_and_summarize_content(content, query_context, subquestion, model):
    json_format = (
        '{\n  "relevant": true/false,\n  "reason": "<reason for relevance>",\n  "summary": "<concise detailed summary if relevant>"\n}'
    )
    prompt = (
        f"Context: {query_context}\n\n"
        f"Content: {content}\n\n"
        f"Subquestion: {subquestion}\n\n"
        f"Determine if the provided content is directly relevant to the context for answering the subquestion. "
        f"Relevance should be based on specific, factual information contained within the provided context. Do not use any other knowledge. "
        f"If relevant, provide a very detailed summary of the content that is self-contained and can stand on its own. Ensure the summary includes specific details and mentions all relevant information explicitly. Do not make general statements or reference any information not present in the provided content. "
        f"Provide the response in the following plain JSON format without any Markdown formatting:\n"
        f"{json_format}"
    )
    logging.info(f"Evaluating relevance and summarizing content for subquestion: {subquestion[:200]}...")
    response = generate_response_with_ollama(prompt, model)

    result = parse_json_response(response, json_format, model)
    if result:
        is_relevant = result.get("relevant", False)
        reason = result.get("reason", "No reason provided")
        summary = result.get("summary", "") or ""
        logging.info(f"Content relevance and summary result for context {query_context[:200]}: {is_relevant}, Reason: {reason}, Summary: {summary[:200]}")
        return is_relevant, reason, summary
    else:
        logging.error(f"Failed to evaluate and summarize content for subquestion: {subquestion[:200]}")
        return False, "Failed to parse JSON response", ""

def split_and_process_chunks(subquestion, url, text, model):
    # Generate a hash of the text to use as a cache key
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()

    # Check if the chunks are already in the cache
    if text_hash in chunk_cache:
        logging.info(f"Using cached chunks for text hash: {text_hash}")
        chunk_summaries, chunk_relevance = chunk_cache[text_hash]
    else:
        embed_model = HuggingFaceEmbedding(model_name=CONFIG["EMBEDDING_MODEL_NAME"], device=device)
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

        # Save the chunks to the cache
        chunk_cache[text_hash] = (chunk_summaries, chunk_relevance)
        save_cache(chunk_cache, CONFIG["CHUNK_CACHE_FILE_PATH"])

    summarized_text = " ".join(chunk_summaries)
    is_relevant = any(chunk_relevance)
    reason = "At least one chunk is relevant" if is_relevant else "None of the chunks are relevant"

    return summarized_text, is_relevant, reason

def detect_automation_denial(content, model):
    prompt = (
        f"Determine if the following content indicates a denial due to automation, scraping, or similar reasons. "
        f"Respond only with 'True' if there is any indication of denial and only with 'False' otherwise.\n\n"
        f"Content: {content}"
    )
    response = generate_response_with_ollama(prompt, model).strip()
    if response.lower().startswith('true'):
        return True
    return False

def process_url(subquestion, url, model):
    logging.info(f"Fetching content for URL: {url}")
    retries = 0
    while retries < CONFIG["MAX_RETRIES"]:
        content, content_type = fetch_content_with_browser(url)
        url_cache[url] = (content, content_type)
        save_cache(url_cache, CONFIG["URL_CACHE_FILE_PATH"])

        if content:
            save_raw_content(subquestion, url, content, content_type)
            if content_type == 'html':
                logging.info(f"Cleaning HTML content for URL: {url}")
                cleaned_html = clean_html_content(content)
                extracted_text = extract_text_from_html(cleaned_html)
            else:
                return "", ""

            cleaned_text = cleanup_extracted_text(extracted_text)
            if detect_automation_denial(cleaned_text, CONFIG["MODEL_NAME"]):
                retries += 1
                logging.warning(f"Detected automation denial. Retrying {retries}/{CONFIG['MAX_RETRIES']}")
                time.sleep(CONFIG["INITIAL_RETRY_DELAY_SECONDS"])
                ensure_browser()  # Reinitialize browser with a new user agent
                continue

            if cleaned_text:
                summarized_text, is_relevant, reason = split_and_process_chunks(subquestion, url, cleaned_text, model)
                save_cleaned_content(subquestion, url, cleaned_text, summarized_text, is_relevant, reason)
                logging.info(f"Cleaned text for URL {url}: {cleaned_text[:200]}")
                return summarized_text, url
            else:
                save_cleaned_content(subquestion, url, "", "", False, "Failed to extract cleaned text")
        retries += 1
        time.sleep(CONFIG["INITIAL_RETRY_DELAY_SECONDS"])
    return "", ""

def process_subquestion(subquestion, model, num_search_results_google, num_search_results_vector, original_query):
    visited_urls = set()
    domain_timestamps = {}
    all_contexts = []
    all_references = []
    urls_to_process = []
    documents_to_process = []
    relevant_answers = []

    logging.info(f"Processing subquestion: {subquestion}")

    search_terms = generate_search_terms(subquestion, model)
    logging.info(f"Translated search terms: {search_terms}")

    results = search_google_with_retries(search_terms, num_search_results_google)
    if results:
        urls_to_process.extend(results)

    if os.path.exists(CONFIG["VECTOR_STORE_DIRECTORY"]):
        vector_store_client = chromadb.PersistentClient(path=CONFIG["VECTOR_STORE_DIRECTORY"])
        collection = vector_store_client.get_collection(name=CONFIG["VECTOR_STORE_COLLECTION_NAME"])
        vector_results = query_vector_store(collection, subquestion, top_k=num_search_results_vector)

        sources_processed = set()
        combined_documents = {}

        for doc, meta in zip(vector_results['documents'], vector_results['metadatas']):
            logging.debug(f"Processing doc: {doc[:200]} with meta: {meta}")
            for m, d in zip(meta, doc):  # Iterating through each meta-doc pair if they are lists
                source = m.get('source', '')
                if source not in combined_documents:
                    combined_documents[source] = []
                combined_documents[source].append((m, d))

        for source, docs in combined_documents.items():
            logging.debug(f"Combining documents for source: {source}")
            combined_docs_sorted = sorted(docs, key=lambda x: x[0].get('chunk_id', 0))  # Sort by chunk_id if available
            combined_doc = " ".join([d[1] for d in combined_docs_sorted])
            meta = combined_docs_sorted[0][0]
            logging.debug(f"Combined doc: {combined_doc[:200]} with meta: {meta}")
            documents_to_process.append((combined_doc, meta))

    logging.info(f"Initial items to process: URLs = {len(urls_to_process)}, Documents = {len(documents_to_process)}")

    while urls_to_process or documents_to_process:
        if urls_to_process:
            current_url = urls_to_process.pop(0)
            domain = urlparse(current_url).netloc
            current_time = time.time()

            if current_url in visited_urls:
                continue

            if domain in domain_timestamps and current_time - domain_timestamps[domain] < CONFIG["REQUEST_DELAY_SECONDS"]:
                wait_time = CONFIG["REQUEST_DELAY_SECONDS"] - (current_time - domain_timestamps[domain])
                logging.info(f"Waiting for {wait_time} seconds before making another request to {domain}")
                time.sleep(wait_time)
                current_time = time.time()

            extracted_text, reference = process_url(subquestion, current_url, model)
            if extracted_text:
                all_contexts.append(f"Content from {current_url}:\n{extracted_text}")
                all_references.append(reference)
                relevant_answers.append(extracted_text)
                visited_urls.add(current_url)
                domain_timestamps[domain] = current_time

        if documents_to_process:
            doc, meta = documents_to_process.pop(0)
            logging.info(f"Evaluating content relevance for document from vector store with metadata: {meta}")
            summarized_text, is_relevant, reason = split_and_process_chunks(subquestion, meta.get('source', 'vector_store'), doc, model)
            source = meta.get('source', 'vector_store')

            all_contexts.append(f"Content from {source}:\n{summarized_text}")
            all_references.append(source)
            relevant_answers.append(summarized_text)
            save_cleaned_content(subquestion, source, doc, summarized_text, is_relevant, reason, source="vector_store", vector_metadata=meta)
            logging.info(f"Document from vector store: {doc[:200]}")

        time.sleep(random.uniform(CONFIG["REQUEST_DELAY_SECONDS"], CONFIG["REQUEST_DELAY_SECONDS"] + 2))

    if relevant_answers:
        prompt = (
            f"Given the following relevant excerpts, provide a single, concise, and detailed answer to the subquestion: {subquestion}\n\n"
            f"Relevant excerpts:\n{{relevant_answers}}\n\n"
            f"Your task is to synthesize these excerpts into a cohesive answer. Ensure that your response is:\n\n"
            f"1. Comprehensive: Address all aspects of the subquestion thoroughly.\n"
            f"2. Detailed: Include specific details and evidence from the excerpts to support your answer.\n"
            f"3. Accurate: Base your answer solely on the provided excerpts without introducing external information.\n"
            f"4. Clear and Structured: Organize your response logically and ensure clarity.\n\n"
            f"Additionally, highlight the significance of each detail included in your answer and how it directly relates to the subquestion. Avoid generalizations and ensure that your response is self-contained and can stand alone without requiring further context."
        )
        final_subquestion_answer = generate_response_with_ollama(prompt, model)
        all_subquestion_answers = [final_subquestion_answer]
    else:
        all_subquestion_answers = ["No relevant information found."]

    logging.info(f"Context gathered for subquestion '{subquestion}': {all_contexts}")
    logging.info(f"Answer gathered for subquestion '{subquestion}': {all_subquestion_answers}")

    main_question_hash = hashlib.md5(original_query.encode('utf-8')).hexdigest()
    output_dir = os.path.join(CONFIG["OUTPUT_DIRECTORY"], main_question_hash)
    os.makedirs(output_dir, exist_ok=True)
    append_to_file(output_dir, f"{hashlib.md5(subquestion.encode('utf-8')).hexdigest()}_subquestion.txt", subquestion)
    append_to_file(output_dir, f"{hashlib.md5(subquestion.encode('utf-8')).hexdigest()}_context.txt", "\n\n".join(all_contexts))
    append_to_file(output_dir, f"{hashlib.md5(subquestion.encode('utf-8')).hexdigest()}_answer.txt", "\n\n".join(all_subquestion_answers))

    return all_contexts, all_references, all_subquestion_answers

def search_and_extract(subquestions, model, num_search_results_google, num_search_results_vector, original_query, context=""):
    all_contexts = []
    all_references = []
    all_subquestion_answers = []
    main_question_answered = False

    # New list to keep track of all subquestions used
    all_subquestions_used = subquestions[:]

    while subquestions:
        subquestion = subquestions.pop()
        subquestion_context, references, subquestion_answers = process_subquestion(subquestion, model, num_search_results_google, num_search_results_vector, original_query)

        if subquestion_context:
            all_contexts.extend(subquestion_context)
            all_references.extend(references)
            all_subquestion_answers.extend(subquestion_answers)

            main_question_answered = check_if_main_question_answered(all_contexts, all_subquestion_answers, original_query, subquestions)
            if main_question_answered:
                break
        else:
            context_str = "\n\n".join(all_contexts)
            refined_subquestions = rephrase_query_to_followup_subquestions(subquestion, model, CONFIG["NUM_INITIAL_SUBQUESTIONS"], context_str)
            subquestions.extend(refined_subquestions)
            all_subquestions_used.extend(refined_subquestions)  # Add newly generated subquestions

    if not main_question_answered:
        context_str = "\n\n".join(all_contexts)
        remaining_subquestions = rephrase_query_to_followup_subquestions(original_query, model, CONFIG["NUM_INITIAL_SUBQUESTIONS"], context_str)
        subquestions.extend(remaining_subquestions)
        all_subquestions_used.extend(remaining_subquestions)  # Add newly generated subquestions
        search_and_extract(subquestions, model, num_search_results_google, num_search_results_vector, original_query, context_str)
    else:
        if all_contexts:
            prompt = (
                f"Given the following context, subquestions, and their answers, answer the main question: {original_query}\n\n"
                f"The provided context encompasses both factual information and a range of opinions. In addition, several subquestions related to the main question have been answered, and these answers should be considered when formulating your response. Your task is to use solely the supplied context, along with the subquestions and their answers, to construct a comprehensive and detailed response to the main question. Refrain from incorporating any external knowledge or information.\n\n"
                f"Context:\n{all_contexts}\n\n"
                f"Subquestions and their answers:\n{all_subquestion_answers}\n\n"
                f"Ensure your answer is exhaustive, drawing upon all relevant details from the context and the subquestion answers. Provide a clear and well-structured response that fully addresses the main question."
            )
            response = generate_response_with_ollama(prompt, model)
            save_final_output(original_query, all_subquestions_used, all_contexts, [response])  # Use the updated list

def check_if_main_question_answered(contexts, subquestion_answers, main_question, subquestions):
    combined_contexts = "\n\n".join(contexts)
    combined_subquestion_answers = "\n\n".join(subquestion_answers)
    json_format = (
        '{\n  "answered": true/false,\n  "reason": "<reason if not fully answered>",\n  "additional_information_needed": "<details of what additional information is needed if any>"\n}'
    )
    prompt = (
        f"Given the following context and answers to subquestions, determine if the main question has been fully answered:\n\n"
        f"Main question: {main_question}\n\n"
        f"Context:\n{combined_contexts}\n\n"
        f"Subquestions and their answers:\n{combined_subquestion_answers}\n\n"
        f"Respond with a JSON object containing the following keys:\n"
        f"{json_format}"
    )
    logging.info(f"Checking if main question is answered. Main question: {main_question}")
    response = generate_response_with_ollama(prompt, CONFIG["MODEL_NAME"])
    result = parse_json_response(response, json_format, CONFIG["MODEL_NAME"])
    if result:
        answered = result.get("answered", False)
        reason = result.get("reason", "")
        additional_information_needed = result.get("additional_information_needed", "")

        if answered:
            return True
        else:
            logging.info(f"Reason: {reason}")
            logging.info(f"Additional information needed: {additional_information_needed}")
            refined_subquestions = rephrase_query_to_followup_subquestions(main_question, CONFIG["MODEL_NAME"], CONFIG["NUM_FOLLOWUP_SUBQUESTIONS"], additional_information_needed)
            subquestions.extend(refined_subquestions)
            return False
    return False

if __name__ == "__main__":
    try:
        original_query = (
            "I am playing as an Eldritch Knight Elf in Dungeons & Dragons 5th Edition, focusing on ranged combat. My character does not have access to homebrew spells and has a Dexterity score of 20 and an Intelligence score of 16. Please provide a list of effective level 1 to level 3 spells that would be beneficial for my character to have. Include specific details and explanations for why each spell is beneficial."
        )

        logging.info(f"Starting script with original query: {original_query}")
        subquestions = rephrase_query_to_initial_subquestions(original_query, CONFIG["MODEL_NAME"], CONFIG["NUM_INITIAL_SUBQUESTIONS"])
        if subquestions:
            search_and_extract(subquestions, CONFIG["MODEL_NAME"], CONFIG["NUM_SEARCH_RESULTS_GOOGLE"], CONFIG["NUM_SEARCH_RESULTS_VECTOR"], original_query)
        logging.info("Script completed.")
    except Exception as e:
        logging.error(f"Uncaught exception: {e}")
    finally:
        if llm_cache:
            llm_cache.close()
        if google_cache:
            google_cache.close()
        if url_cache:
            google_cache.close()
        if chunk_cache:
            chunk_cache.close()
        if browser:
            browser.quit()
