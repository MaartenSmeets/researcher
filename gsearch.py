import logging
import requests
import os
import shelve
import fcntl
import hashlib
import random
import shutil
from bs4 import BeautifulSoup
from lxml import etree, html
from googlesearch import search
import ollama

# Configurable parameters
CONFIG = {
    "MODEL": "gemma2unc",
    "NUM_SUBQUESTIONS": 5,
    "NUM_SEARCH_RESULTS": 5,
    "LOG_FILE": 'logs/app.log',
    "CACHE_FILE": 'cache/cache.db',
    "EXTRACTED_CONTENT_DIR": 'extracted_content',
    "USER_AGENTS": [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/91.0.864.59",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    ]
}

RAW_CONTENT_DIR = os.path.join(CONFIG["EXTRACTED_CONTENT_DIR"], 'raw')
CLEANED_CONTENT_DIR = os.path.join(CONFIG["EXTRACTED_CONTENT_DIR"], 'cleaned')
RELEVANT_DIR = os.path.join(CLEANED_CONTENT_DIR, 'relevant')
NOT_RELEVANT_DIR = os.path.join(CLEANED_CONTENT_DIR, 'not_relevant')

# Create directories if they don't exist
for directory in [os.path.dirname(CONFIG["LOG_FILE"]), os.path.dirname(CONFIG["CACHE_FILE"]), RAW_CONTENT_DIR, RELEVANT_DIR, NOT_RELEVANT_DIR]:
    os.makedirs(directory, exist_ok=True)

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
file_handler = logging.FileHandler(CONFIG["LOG_FILE"], mode='w')
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

cache = open_cache(CONFIG["CACHE_FILE"])

def save_cache():
    try:
        with open(CONFIG["CACHE_FILE"] + '.lock', 'w') as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            cache.sync()
            fcntl.flock(lock_file, fcntl.LOCK_UN)
    except Exception as e:
        logging.error(f"Failed to save cache: {e}")

def save_raw_content(subquestion, url, raw_text):
    filename_hash = hashlib.md5((subquestion + url).encode('utf-8')).hexdigest()
    raw_filename = f"{filename_hash}_raw.txt"
    metadata_filename = f"{filename_hash}_raw_meta.txt"

    with open(os.path.join(RAW_CONTENT_DIR, raw_filename), 'w', encoding='utf-8') as f:
        f.write(raw_text)

    with open(os.path.join(RAW_CONTENT_DIR, metadata_filename), 'w', encoding='utf-8') as f:
        metadata = {
            'url': url,
            'subquestion': subquestion,
            'status': 'raw'
        }
        f.write(str(metadata))

def save_cleaned_content(subquestion, url, cleaned_text, summary, is_relevant, reason):
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
            'summary': summary
        }
        f.write(str(metadata))

def fetch_page_content(url):
    try:
        headers = {'User-Agent': random.choice(CONFIG["USER_AGENTS"])}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        if 'text/html' in response.headers.get('Content-Type', ''):
            try:
                return response.content.decode('utf-8')
            except UnicodeDecodeError:
                return response.content.decode('ISO-8859-1')
        else:
            logging.info(f"Skipped non-HTML content at {url}")
            return ""
    except requests.RequestException as e:
        logging.error(f"Failed to retrieve {url}: {e}")
        return ""

def get_page_content(subquestion, url, visited_urls):
    raw_content = fetch_page_content(url)
    if raw_content:
        save_raw_content(subquestion, url, raw_content)
        
        # Find and evaluate links in the page
        soup = BeautifulSoup(raw_content, 'html.parser')
        links = [link['href'] for link in soup.find_all('a', href=True) if link['href'].startswith('http') and link['href'] not in visited_urls]
        return raw_content, links
    return "", []

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

def generate_summary(text, query_context, model):
    prompt = (
        f"Given the following query context: {query_context}\n\n"
        f"Summarize the following content, focusing on parts relevant for answering the main question. "
        f"Make sure the summary is elaborate on relevant parts to avoid losing important information.\n\n"
        f"Content:\n{text}\n\n"
        f"Summary:"
    )
    logging.info(f"Requesting summary for text from URL. Query context: {query_context[:100]}...")
    response = generate_response_with_ollama(prompt, model)
    if response:
        logging.info(f"Summary generated successfully for context: {query_context[:100]}...")
        return response.strip()
    else:
        logging.error("Failed to generate summary.")
        return "Summary generation failed."

def evaluate_content_relevance(content, query_context, model):
    prompt = (
        f"Given the following query context: {query_context}\n"
        f"Evaluate the relevance and trustworthiness of the provided content. Respond strictly with 'yes' or 'no', followed by a brief and concise explanation.\n\n"
        f"Content:\n{content[:1000]}\n"
        f"Response (yes or no):"
    )
    logging.info(f"Evaluating content relevance for query context: {query_context[:100]}...")
    response = generate_response_with_ollama(prompt, model)
    if response:
        is_relevant = response.lower().startswith("yes")
        reason = response[4:].strip()
        logging.info(f"Content relevance evaluation result for context {query_context[:100]}: {is_relevant}, Reason: {reason}")
        return is_relevant, reason
    else:
        logging.error("Failed to evaluate content relevance.")
        return False, "Evaluation failed"

def process_subquestion(subquestion, model, num_search_results, original_query):
    visited_urls = set()
    all_contexts = ""
    all_references = []

    logging.info(f"Processing subquestion: {subquestion}")
    try:
        results = search(subquestion, num_results=num_search_results)
    except Exception as e:
        logging.error(f"Failed to search for subquestion '{subquestion}': {e}")
        return "", []

    urls_to_process = set(results)

    while urls_to_process:
        logging.info(f"Number of URLs to process: {len(urls_to_process)}")
        current_url = urls_to_process.pop()
        if current_url in visited_urls:
            continue

        logging.info(f"Fetching page content for URL: {current_url}")
        html_content, _ = get_page_content(subquestion, current_url, visited_urls)
        if html_content:
            logging.info(f"Cleaning HTML content for URL: {current_url}")
            cleaned_html = clean_html_content(html_content)
            if cleaned_html:
                extracted_text = extract_text_from_html(cleaned_html)
                if extracted_text:
                    logging.info(f"Evaluating content relevance for URL: {current_url}")
                    is_relevant, reason = evaluate_content_relevance(extracted_text, subquestion, model)
                    if is_relevant:
                        logging.info(f"Content from {current_url} is relevant. Generating summary.")
                        summary = generate_summary(extracted_text, subquestion, model)
                        save_cleaned_content(subquestion, current_url, extracted_text, summary, is_relevant, reason)
                        all_contexts += f"Summary of document from {current_url}:\n{summary}\n\n"
                        all_references.append(current_url)
                        visited_urls.add(current_url)

    return all_contexts, all_references

def search_and_extract(subquestions, model, num_search_results, original_query):
    all_contexts = ""
    all_references = []

    while subquestions:
        logging.info(f"Number of subquestions to process: {len(subquestions)}")
        subquestion = subquestions.pop()
        context, references = process_subquestion(subquestion, model, num_search_results, original_query)

        if context:
            all_contexts += context
            all_references.extend(references)
        else:
            logging.info(f"Insufficient context, refining subquestion: {subquestion}")
            refined_subquestions = rephrase_query_to_subquestions(subquestion, model, CONFIG["NUM_SUBQUESTIONS"])
            subquestions.extend(refined_subquestions)

    if all_contexts:
        prompt = f"Given the following context, answer the question: {original_query}\n\nContext:\n{all_contexts}\n\nReferences:\n" + "\n".join(all_references)
        logging.info(f"Requesting final answer for the original query. Query: {original_query}")
        response = generate_response_with_ollama(prompt, model)
        logging.info(f"Final response:\n{response}\n")

def generate_response_with_ollama(prompt, model):
    if prompt in cache:
        logging.info(f"Using cached response for prompt: {prompt[:100]}...")
        return cache[prompt]
    try:
        response = ollama.generate(model=model, prompt=prompt)
        response_content = response.get('response', "")
        if not response_content:
            logging.error(f"Unexpected response structure: {response}")
            return ""

        cache[prompt] = response_content
        save_cache()
        return response_content
    except Exception as e:
        logging.error(f"Failed to generate response with Ollama: {e}")
        return ""

def rephrase_query_to_subquestions(query, model, num_subquestions):
    prompt = (
        f"Given the following main question: {query}\n\n"
        f"Generate {num_subquestions} concise subquestions that include sufficient information and keywords to make the answer likely relevant to the main question. "
        f"Ensure each subquestion is self-contained and does not reference information not available in the subquestion itself. "
        f"Only reply with the subquestions, each on a new line without using a list format."
    )
    logging.info(f"Requesting rephrased subquestions for query: {query}")
    response = generate_response_with_ollama(prompt, model)
    if response:
        subquestions = response.split('\n')
        return [sq for sq in subquestions if sq.strip()]
    else:
        logging.error("Failed to generate subquestions.")
        return []

if __name__ == "__main__":
    original_query = "Suggest appropriate feats for a Dungeons & Dragons 5th Edition Eldritch Knight Elf who specializes in ranged combat. The character has a Dexterity score of 20 and an Intelligence score of 16. The character already possesses the Crossbow Expert and Sharpshooter feats. Please recommend feats exclusively from officially published D&D sources, such as the Player's Handbook, Xanathar's Guide to Everything, Tasha's Cauldron of Everything, and the Dungeon Master's Guide. Do not include any homebrew or unofficial content in your suggestions."
    
    logging.info(f"Starting script with original query: {original_query}")
    subquestions = rephrase_query_to_subquestions(original_query, CONFIG["MODEL"], CONFIG["NUM_SUBQUESTIONS"])
    if subquestions:
        search_and_extract(subquestions, CONFIG["MODEL"], CONFIG["NUM_SEARCH_RESULTS"], original_query)
    logging.info("Script completed.")
    if cache:
        cache.close()
