import logging
from googlesearch import search
import ollama
import requests
from bs4 import BeautifulSoup
import os
import shelve
import fcntl
import hashlib
import random
import shutil
from lxml import etree, html

# Configurable parameters
MODEL = "gemma2:27b-instruct-fp16"
NUM_SUBQUESTIONS = 5
NUM_SEARCH_RESULTS = 5
LOG_FILE = 'logs/app.log'
CACHE_FILE = 'cache/cache.db'
EXTRACTED_CONTENT_DIR = 'extracted_content'
RAW_CONTENT_DIR = os.path.join(EXTRACTED_CONTENT_DIR, 'raw')
CLEANED_CONTENT_DIR = os.path.join(EXTRACTED_CONTENT_DIR, 'cleaned')
RELEVANT_DIR = os.path.join(CLEANED_CONTENT_DIR, 'relevant')
NOT_RELEVANT_DIR = os.path.join(CLEANED_CONTENT_DIR, 'not_relevant')

# List of User Agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    # Add more user agents if necessary
]

# Create directories if they don't exist
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
os.makedirs(RAW_CONTENT_DIR, exist_ok=True)
os.makedirs(RELEVANT_DIR, exist_ok=True)
os.makedirs(NOT_RELEVANT_DIR, exist_ok=True)

# Clean relevant and not relevant directories
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

# Configure logging
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

file_handler = logging.FileHandler(LOG_FILE, mode='w')
file_handler.setFormatter(log_formatter)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

# Load or initialize the cache
def open_cache(cache_file):
    try:
        return shelve.open(cache_file, writeback=True)
    except Exception as e:
        logging.error(f"Failed to open cache file: {e}")
        return None

cache = open_cache(CACHE_FILE)

def save_cache():
    try:
        with open(CACHE_FILE + '.lock', 'w') as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            cache.sync()
            fcntl.flock(lock_file, fcntl.LOCK_UN)
    except Exception as e:
        logging.error(f"Failed to save cache: {e}")

def save_raw_content(subquestion, url, raw_text):
    filename_hash = hashlib.md5((subquestion + url).encode('utf-8')).hexdigest()
    raw_filename = f"{filename_hash}_raw.txt"

    with open(os.path.join(RAW_CONTENT_DIR, raw_filename), 'w', encoding='utf-8') as f:
        f.write(f"URL: {url}\n\n{subquestion}\n\n{raw_text}")

def get_page_content(subquestion, url):
    try:
        logging.debug(f"Fetching page content from URL: {url}")
        headers = {'User-Agent': random.choice(USER_AGENTS)}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        if 'text/html' in response.headers.get('Content-Type', ''):
            raw_content = response.content.decode('utf-8')
            save_raw_content(subquestion, url, raw_content)
            return raw_content
        else:
            logging.info(f"Skipped non-HTML content at {url}")
            return ""
    except requests.RequestException as e:
        logging.error(f"Failed to retrieve {url}: {e}")
        return ""

def clean_html_content(html_content):
    logging.info("Cleaning HTML content using BeautifulSoup and lxml...")
    soup = BeautifulSoup(html_content, 'html.parser')

    for script_or_style in soup(['script', 'style']):
        script_or_style.extract()

    clean_html = soup.prettify()

    tree = html.fromstring(clean_html)
    for element in tree.xpath('//script|//style'):
        element.drop_tree()

    clean_text = etree.tostring(tree, method='html', encoding='unicode')
    return clean_text

def extract_valuable_information_from_html(html_content):
    logging.info("Extracting valuable information from HTML content...")
    cleaned_text = clean_html_content(html_content)
    return cleaned_text

def evaluate_content_relevance(content, query_context, model):
    logging.info("Evaluating content relevance and trustworthiness...")
    prompt = (
        f"Evaluate the relevance and trustworthiness of the following content in the context of the query: {query_context}. "
        f"Respond with 'yes' or 'no' followed by a brief explanation:\n\nContent:\n{content}"
    )
    response = generate_response_with_ollama(prompt, model)
    if response:
        logging.info(f"Evaluation response: {response}")
        return response.lower().startswith("yes")
    else:
        logging.error("Failed to evaluate content relevance.")
        return False

def save_cleaned_content(subquestion, url, cleaned_text, is_relevant):
    filename_hash = hashlib.md5((subquestion + url).encode('utf-8')).hexdigest()
    cleaned_filename = f"{filename_hash}_cleaned.txt"

    directory = RELEVANT_DIR if is_relevant else NOT_RELEVANT_DIR
    with open(os.path.join(directory, cleaned_filename), 'w', encoding='utf-8') as f:
        f.write(f"URL: {url}\n\n{subquestion}\n\n{cleaned_text}")

def search_and_extract(subquestions, model, num_search_results, original_query):
    sufficient_context = False

    while not sufficient_context:
        all_contexts = ""
        all_references = []

        for subquestion in subquestions:
            logging.info(f"Processing subquestion: {subquestion}")
            try:
                results = search(subquestion, num_results=num_search_results)
            except Exception as e:
                logging.error(f"Failed to search for subquestion '{subquestion}': {e}")
                continue

            context = ""
            references = []

            for url in results:
                logging.info(f"Retrieving content from {url}")
                html_content = get_page_content(subquestion, url)
                if html_content:
                    logging.info(f"Extracting valuable information from {url}")
                    cleaned_text = extract_valuable_information_from_html(html_content)
                    if cleaned_text:
                        logging.info(f"Evaluating content from {url}")
                        is_relevant = evaluate_content_relevance(cleaned_text, original_query, model)
                        save_cleaned_content(subquestion, url, cleaned_text, is_relevant)
                        if is_relevant:
                            context += cleaned_text + "\n\n"
                            references.append(url)
                        else:
                            logging.info(f"Content from {url} deemed irrelevant or untrustworthy.")

            if context:
                all_contexts += context
                all_references.extend(references)

        if all_contexts:
            logging.info("Sufficient context retrieved.")
            sufficient_context = True
            prompt = f"Given the following context, answer the question: {original_query}\n\nContext:\n{all_contexts}\n\nReferences:\n" + "\n".join(all_references)
            response = generate_response_with_ollama(prompt, model)
            logging.info(f"Final response:\n{response}\n")
        else:
            logging.warning("No relevant or trustworthy context retrieved. Generating additional subquestions.")
            subquestions = rephrase_query_to_subquestions(original_query, model, NUM_SUBQUESTIONS)

def generate_response_with_ollama(prompt, model):
    if prompt in cache:
        logging.info(f"Using cached response for prompt: {prompt[:50]}...")
        return cache[prompt]
    try:
        logging.info(f"Generating response with Ollama for prompt: {prompt[:50]}...")
        response = ollama.chat(model=model, messages=[{'role': 'user', 'content': prompt}])
        response_content = response['message']['content']
        cache[prompt] = response_content
        save_cache()
        return response_content
    except Exception as e:
        logging.error(f"Failed to generate response with Ollama: {e}")
        return ""

def rephrase_query_to_subquestions(query, model, num_subquestions):
    logging.info(f"Rephrasing query into subquestions: {query}")
    prompt = (
        f"Given the following main question: {query}\n\n"
        f"Generate {num_subquestions} concise subquestions that include sufficient information and keywords to make the answer likely relevant to the main question. "
        f"Ensure each subquestion is self-contained and does not reference information not available in the subquestion itself. "
        f"Only reply with the subquestions, each on a new line without using a list format."
    )
    response = generate_response_with_ollama(prompt, model)
    if response:
        subquestions = response.split('\n')
        subquestions = [sq for sq in subquestions if sq.strip()]
        logging.info(f"Generated subquestions: {subquestions}")
        return subquestions
    else:
        logging.error("Failed to generate subquestions.")
        return []

if __name__ == "__main__":
    original_query = "Suggest feats for a dnd 5th edition eldritch knight elf focused on ranged combat with dex 20 and int 16 who already has crossbow expert and sharpshooter. Do not use homebrew content but only from verifiable official DnD books like Xanathar's Guide to Everything and the Player's Handbook, Tasha's Cauldron of Everything, Player's Handbook, Dungeon Master's Guide."
    
    logging.info(f"Starting script with original query: {original_query}")
    subquestions = rephrase_query_to_subquestions(original_query, MODEL, NUM_SUBQUESTIONS)
    if subquestions:
        search_and_extract(subquestions, MODEL, NUM_SEARCH_RESULTS, original_query)
    logging.info("Script completed.")
    if cache:
        cache.close()
