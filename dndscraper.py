import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
import os
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup, Comment

class LinkSpider(scrapy.Spider):
    name = "link_spider"
    allowed_domains = ["dnd5e.wikidot.com"]  # Replace with the target domain
    start_urls = ["http://dnd5e.wikidot.com/"]  # Replace with the target URL
    excluded_sources = [
        'unearthed arcana', 'strixhaven', 'eberron', 'ravenloft', 'critical role', 
        'wildemount', 'mythic odysseys of theros', 'ravnica', 'acquisitions inc', 
        'planescape', 'spelljammer', 'theros', 'volo', 'exploring', 'mordenkainen'
    ]  # Add sources to exclude
    excluded_paths = ['feed', 'forum']  # Add paths to exclude
    visited_urls = {}  # Dictionary to keep track of visited URLs and their local filenames
    counter = 0  # Counter for unique filenames
    excluded_counter = 0  # Counter for unique filenames for excluded pages

    def parse(self, response):
        # Extract the title from <div class="page-title page-header"><span>...</span></div>
        page_title = response.css('div.page-title.page-header span::text').get()

        # Check for excluded sources in the page content
        page_content = response.css('div#page-content').get()

        if page_content:
            if self.contains_excluded_source(page_content):
                self.excluded_counter += 1
                filename = f'excluded_pages/page_{self.excluded_counter}.html'
            else:
                self.counter += 1
                filename = f'crawled_pages/page_{self.counter}.html'
                self.visited_urls[response.url] = filename
            
            # Add title and URL to the combined content
            cleaned_content = self.clean_html(page_content)
            final_content = f"""
            <html>
                <head>
                    <title>{page_title}</title>
                </head>
                <body>
                    <h1>{page_title}</h1>
                    <p><strong>Original URL:</strong> <a href="{response.url}">{response.url}</a></p>
                    {cleaned_content}
                </body>
            </html>
            """
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(final_content)
            self.log(f'Saved file {filename}')
        
        # Extract all links from the page
        links = response.css('a::attr(href)').extract()
        for link in links:
            full_url = response.urljoin(link)
            # Check if the link is within the allowed domain and has not been visited
            if full_url.startswith('http://dnd5e.wikidot.com') and full_url not in self.visited_urls:
                # Exclude links that contain any of the excluded paths
                if not any(excluded in full_url for excluded in self.excluded_paths):
                    self.visited_urls[full_url] = None  # Placeholder to avoid revisits
                    yield response.follow(full_url, self.parse)

    def contains_excluded_source(self, page_content):
        """Check if the page content contains any of the excluded sources."""
        soup = BeautifulSoup(page_content, 'html.parser')
        source_paragraphs = soup.find_all('p')
        for p in source_paragraphs:
            if p.text.lower().startswith('source:'):
                if any(source in p.text.lower() for source in self.excluded_sources):
                    return True
        return False

    def clean_html(self, html_content):
        """Remove scripts, styles, and unnecessary tags from the HTML content."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()
        
        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # Clean up unnecessary tags or attributes if needed
        # Here we can add more logic to clean up other non-essential tags

        return str(soup)

# Run the spider
if __name__ == "__main__":
    # Create folders for the results
    if not os.path.exists('crawled_pages'):
        os.makedirs('crawled_pages')
    if not os.path.exists('excluded_pages'):
        os.makedirs('excluded_pages')
    
    process = CrawlerProcess(get_project_settings())
    process.crawl(LinkSpider)
    process.start()