import logging
from googlesearch import search
import ollama
import requests

# Configurable parameters
MODEL = "gemma2unc"  # Specify the model to use with Ollama
NUM_SUBQUESTIONS = 10  # Number of subquestions to generate
NUM_SEARCH_RESULTS = 10  # Number of search results per subquestion
LOG_FILE = 'app.log'  # Log file name

# Configure logging to log to both console and file
log_formatter = logging.Formatter('%(asctime)s - %(message)s')

# Set up logging to console
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

# Set up logging to file, overwriting on each run
file_handler = logging.FileHandler(LOG_FILE, mode='w')
file_handler.setFormatter(log_formatter)

# Get the root logger and configure it
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

# Function to retrieve the HTML content of a webpage
def get_page_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        if 'text/html' in response.headers.get('Content-Type', ''):
            return response.content.decode('utf-8')
        else:
            logging.info(f"Skipped non-HTML content at {url}")
            return ""
    except requests.RequestException as e:
        logging.error(f"Failed to retrieve {url}: {e}")
        return ""

# Function to interact with Ollama API and generate response
def generate_response_with_ollama(prompt, model):
    logging.info(f"Generating response with Ollama for prompt: {prompt[:50]}...")
    response = ollama.chat(model=model, messages=[{
        'role': 'user',
        'content': prompt,
    }])
    return response['message']['content']

# Function to rephrase the original query into complete subquestions
def rephrase_query_to_subquestions(query, model, num_subquestions):
    logging.info(f"Rephrasing query into subquestions: {query}")
    prompt = f"Rephrase the following query into a set of {num_subquestions} subquestions that can be used as complete Google search queries. Only reply with the subquestions each on a new line without using a list: {query}"
    response = generate_response_with_ollama(prompt, model)
    subquestions = response.split('\n')
    subquestions = [sq for sq in subquestions if sq.strip()]
    logging.info(f"Generated subquestions: {subquestions}")
    return subquestions

# Function to extract valuable information from HTML content while retaining structure
def extract_valuable_information_from_html(html_content, model):
    logging.info("Extracting valuable information from HTML content...")
    prompt = f"Extract valuable information from the following HTML content while retaining as much of the original structure as possible but without technical information. The output should be clean valuable text:\n\nHTML Content:\n{html_content}"
    response = generate_response_with_ollama(prompt, model)
    logging.info(f"Extracted valuable information: {response[:200]}...")  # Log the beginning of the extracted text
    return response

# Function to evaluate the relevance and trustworthiness of the content
def evaluate_content_relevance(content, query_context, model):
    logging.info("Evaluating content relevance and trustworthiness...")
    prompt = f"Evaluate whether the following content is relevant and trustworthy in the context of the query: {query_context}. Answer with 'yes' or 'no' followed by an explanation:\n\nContent:\n{content}"
    response = generate_response_with_ollama(prompt, model)
    logging.info(f"Evaluation response: {response}")
    if response.lower().startswith("yes"):
        return True
    return False

# Function to search and extract information for each subquestion
def search_and_extract(subquestions, model, num_search_results, original_query):
    for subquestion in subquestions:
        logging.info(f"Processing subquestion: {subquestion}")
        results = search(subquestion, num_results=num_search_results)
        context = ""
        references = []  # List to store references

        for url in results:
            logging.info(f"Retrieving content from {url}")
            html_content = get_page_content(url)
            if html_content:
                logging.info(f"Extracting valuable information from {url}")
                extracted_text = extract_valuable_information_from_html(html_content, model)
                if extracted_text:
                    logging.info(f"Evaluating content from {url}")
                    if evaluate_content_relevance(extracted_text, original_query, model):
                        context += extracted_text + "\n\n"
                        references.append(url)  # Add URL to references
                    else:
                        logging.info(f"Content from {url} deemed irrelevant or untrustworthy.")
        
        if context:
            logging.info(f"Context retrieved for subquestion '{subquestion}'")
            # Provide context along with the subquestion to the LLM
            prompt = f"Given the following context, answer the question: {subquestion}\n\nContext:\n{context}\n\nReferences:\n" + "\n".join(references)
            response = generate_response_with_ollama(prompt, model)
            logging.info(f"Response for subquestion '{subquestion}':\n{response}\n")
        else:
            logging.warning(f"No relevant or trustworthy context retrieved for subquestion '{subquestion}'")

# Main script
if __name__ == "__main__":
    original_query = "Suggest feats for a dnd 5th edition eldritch knight elf focused on ranged combat with dex 20 and int 16 who already has crossbow expert and sharpshooter. Do not use homebrew content. Only use Xanathar's Guide to Everything and the Player's Handbook, Tasha's Cauldron of Everything, Player's Handbook, Dungeon Master's Guide."
    
    logging.info(f"Starting script with original query: {original_query}")
    subquestions = rephrase_query_to_subquestions(original_query, MODEL, NUM_SUBQUESTIONS)
    search_and_extract(subquestions, MODEL, NUM_SEARCH_RESULTS, original_query)
    logging.info("Script completed.")
