import requests
import sqlite3
import random
from bs4 import BeautifulSoup
from langgraph.graph import StateGraph, END, Graph
from langchain_ollama import ChatOllama
from typing import TypedDict, Annotated, List, Dict, Optional
import logging
import asyncio
from IPython.display import display, Image as IPImage
from langchain_core.pydantic_v1 import BaseModel, Field
import time
import pandas as pd

logging.basicConfig(
    filename='scraper_errors.log',
    level=logging.INFO,
    format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S'
)

# Initialize LLM
class RealLLM:
    def __init__(self, model_name):
        self.llm = ChatOllama(model=model_name)

model_name = "llama3.2:latest"
RealLLM_obj = RealLLM(model_name)
llm = RealLLM_obj.llm

# Step 1: Define State Management
class ScraperState(BaseModel):
    homepage_url: str
    product_urls: Optional[List[str]] = []
    current_product: Optional[str] = None
    product_details: Optional[List[Dict[str, str]]] = []
    status: str = "INITIAL"  # Possible states: INITIAL, SCRAPING_URLS, SCRAPING_PRODUCTS, SAVING_TO_DB, DONE
    errors: Optional[List[str]] = []
    retries: int = 0  # Track retries for scraping
    max_retries: int = 5  # Max retry attempts

# Step 2: User-Agent Headers
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
]

headers = {
    "User-Agent": random.choice(USER_AGENTS),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
    "Accept-Encoding": "gzip, deflate",
    "DNT": "1",
    "Connection": "keep-alive",
}

def get_number_of_pages(search_url: str, max_retries: int = 5, backoff_factor: int = 2) -> int:
    try:
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        page_numbers = soup.find("span", class_="s-pagination-item s-pagination-disabled") 
        page_count = page_numbers.get_text(strip=True) if page_numbers else 1
        logging.info(f"Total number of pages: {page_count}")
        return page_count
    except requests.exceptions.HTTPError as err:
        if err.response.status_code == 503:
            if max_retries > 0:
                retries = max_retries
                wait_time = random.uniform(1, backoff_factor ** retries)
                time.sleep(wait_time)
                return get_number_of_pages(search_url, max_retries - 1, backoff_factor)
            else:
                logging.error(f"Max retries reached: {max_retries}")
                return 1
        else:
            logging.error(f"An error occurred: {str(err)} in get_number_of_pages")
            return 1
    except Exception as e:
        logging.error(f"An error occurred: {str(e)} in get_number_of_pages")
        return 1
    
# Step 3: Scrape All Product URLs from Homepage
def scrape_product_urls(state: ScraperState) -> ScraperState:
    max_retries = 5
    backoff_factor = 2
    retries = 0
    while retries < max_retries:
        try:
            logging.info(f"Scraping product URLs Started")
            response = requests.get(state.homepage_url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")

            product_links = []
            for a_tag in soup.select("a[href*='/dp/']"):
                href = a_tag["href"]
                url_parts = href.split("/")
                product_id = url_parts[url_parts.index("dp") + 1] if "dp" in url_parts else None
                product_url = f"https://www.amazon.in/dp/{product_id}/"
                product_links.append(product_url)

            product_links = list(set(product_links))  # Remove duplicates
            state.product_urls = product_links
            state.status = "SCRAPING_PRODUCTS"
            logging.info(f"Scraping product URLs Completed")
            logging.info(f"Total product URLs found: {len(product_links)}")
            return state
        except requests.exceptions.HTTPError as err:
            if err.response.status_code == 503:
                retries += 1
                wait_time = random.uniform(1, backoff_factor ** retries)
                logging.warning(f"503 Error encountered. Retrying in {wait_time:.2f} seconds... (Attempt {retries}/{max_retries})")
                time.sleep(wait_time)
            else:
                logging.error(f"HTTPError encountered: {err}")
                state.errors.append(str(err))
                state.status = "DONE"
                return state
        except Exception as e:
            logging.error(f"An error occurred: {str(e)} in scrape_product_urls")
            state.errors.append(str(e))
            state.status = "DONE"
            return state
    logging.info(f"Failed to fetch {state.homepage_url} after {max_retries} retries.")
    return state

# Step 4: Scrape Product Details
def scrape_product_details(state: ScraperState) -> ScraperState:
    if not state.product_urls:
        logging.info("No product URLs found")
        state.status = "DONE"
        return state

    logging.info(f"Scraping URLs Count: {len(state.product_urls)}")
    for ind in range(0, len(state.product_urls)):
        if(ind%50 == 0 and ind != 0):
            break
        product_url = state.product_urls.pop(0)  # Process one URL at a time
        state.current_product = product_url
        
        max_retries = 3
        backoff_factor = 2
        retries = state.retries
        while retries < max_retries:
            try:
                response = requests.get(product_url, headers=headers)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, "html.parser")

                product_name = soup.find("span", id="productTitle")
                price = soup.find("span", class_="a-price-whole")
                offer = soup.find("span", class_="savingsPercentage")
                rating = soup.find("span", class_="a-size-base a-color-base")
                brand = soup.find("tr", class_="po-brand")
                description_list = soup.select("#feature-bullets ul li span.a-list-item")
                description = soup.find("div", id="productDescription")
                image = soup.find("img", id="landingImage")
                
                product_id = product_url.split("/dp/")[1].split("/")[0]

                product_data = {
                    "product_id": product_id,
                    "product_url": product_url,
                    "product_name": product_name.get_text(strip=True) if product_name else None,
                    "price": f"₹{price.get_text(strip=True)}" if price else None,
                    "offer": offer.get_text(strip=True) if offer else None,
                    "rating": (
                        float(rating.get_text(strip=True)) 
                        if rating and rating.get_text(strip=True).replace(".", "", 1).isdigit() 
                        else None
                    ),
                    "brand": brand.find_all("td")[1].get_text(strip=True) if brand else None,
                    "description1": ", ".join([desc.get_text(strip=True) for desc in description_list]) if description_list else None,
                    "product_description": description.get_text(strip=True) if description else None,
                    "image_url": image["src"] if image else None,
                }

                state.product_details.append(product_data)
                state.status = "SAVING_TO_DB"
                break
            except requests.exceptions.HTTPError as err:
                if err.response.status_code == 503:
                    retries += 1
                    wait_time = random.uniform(1, backoff_factor ** retries)
                    logging.warning(f"503 Error encountered. Retrying in {wait_time:.2f} seconds... (Attempt {retries}/{max_retries})")
                    state.retries = retries
                    time.sleep(wait_time)
                else:
                    logging.error(f"HTTPError encountered: {err}")
                    state.errors.append(str(err))
                    state.status = "DONE"
                    return state
            except Exception as e:
                logging.error(f"An error occurred: {str(e)} in scrape_product_details")
                state.errors.append(str(e))
                state.status = "DONE"
                return state
    return state

# Step 5: Save Data to SQLite Database
def save_to_db(state: ScraperState):
    if not state.product_details:
        logging.info("No product details found to save.")
        state.status = "DONE"
        return state

    conn = sqlite3.connect("products.db")
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS product_data (
        product_id TEXT,
        product_url TEXT,
        product_name TEXT,
        price TEXT,
        offer TEXT,
        rating REAL,
        brand TEXT,
        description1 TEXT,
        product_description TEXT,
        image_url TEXT)''')

    for product in state.product_details:
        c.execute('''INSERT INTO product_data (
            product_id, product_url, product_name, price, offer, rating, brand,
            description1, product_description, image_url
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', (
            product["product_id"], product["product_url"], product["product_name"], 
            product["price"], product["offer"], product["rating"], product["brand"], 
            product["description1"], product["product_description"], product["image_url"]
        ))

    conn.commit()
    conn.close()
    state.status = "DONE"
    return state

# Step 6: Decision-Making Function
def decision_maker(state: ScraperState) -> str:
    if state.status == "SCRAPING_PRODUCTS" and state.retries >= state.max_retries:
        logging.warning(f"Max retries reached for {state.current_product}. Moving to next step.")
        return "END"  # End the process after reaching retry limit

    if state.errors:
        logging.warning(f"Errors encountered: {state.errors}. Retrying...")
        return "scrape_product_details"  # Retry the scraping step

    if state.product_urls:
        logging.info(f"Scraping more product details...")
        return "scrape_product_details"  # Continue scraping if URLs exist

    if state.product_details:
        logging.info(f"Saving data to DB...")
        return "save_to_db"  # Proceed to save if product details are found

    return "END"  # End the process if none of the conditions match

# Step 7: Workflow Execution (Asynchronous)
async def run_workflow(query: str):
    initial_state = ScraperState(
        homepage_url=query,
        product_urls=[], 
        current_product=None,
        product_details=[],
        status="INITIAL",
        errors=[],
        retries=0,
        max_retries=5
    )
    
    # Initialize graph and nodes
    graph = StateGraph("product_scraper")

    # Add nodes and edges
    graph.add_node("scrape_product_urls", scrape_product_urls)
    graph.add_node("scrape_product_details", scrape_product_details)
    graph.add_node("save_to_db", save_to_db)

    graph.set_entry_point("scrape_product_urls")
    
    graph.add_conditional_edges(
        "scrape_product_urls",
        lambda state: decision_maker(state),
        {
            "scrape_product_details": "scrape_product_details",
            "END": END
        }
    )
    
    graph.add_conditional_edges(
        "scrape_product_details",
        lambda state: decision_maker(state),
        {
            "scrape_product_details": "scrape_product_details",
            "save_to_db": "save_to_db",
            "END": END
        }
    )
    
    graph.add_conditional_edges(
        "save_to_db",
        lambda state: decision_maker(state),
        {
            "scrape_product_details": "scrape_product_details",
            "END": END
        }
    )
    
    app = graph.compile()

    workflow_diagram = app.get_graph().draw_mermaid_png()

    # Save the image file
    with open("product_scrape.png", "wb") as f:
        f.write(workflow_diagram)

    # Run the workflow
    # result = await graph.run(initial_state)
    result = await app.ainvoke(initial_state,{"recursion_limit": 100})  # Make sure to pass the ScraperState object

    return {
        "status": result.status,
        "errors": result.errors,
        "product_urls": result.product_urls,
        "product_details": result.product_details
    }

# Example Usage
query = "https://www.amazon.in/s?k=laptop"  # Example URL to start the scraping process
loop = asyncio.get_event_loop()
result = loop.run_until_complete(run_workflow(query))
print(result)
