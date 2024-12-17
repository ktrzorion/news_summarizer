import os
import re
import html
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import xml.etree.ElementTree as ET
import httpx
import asyncio
from bs4 import BeautifulSoup as bs
from requests_html import AsyncHTMLSession
from LightRAG.lightrag.llm import gpt_4o_complete
from langchain_openai import OpenAI
from datetime import datetime, timedelta
import json
from LightRAG.lightrag import LightRAG
import logging
from dotenv import load_dotenv
import email.utils

# Load environment variables
load_dotenv()

# API Key and Model Initialization
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="News RAG Pipeline API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WORKING_DIR = "./company_data"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# Initialize RAG with async wrapper
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_complete
)

class CompanyReport(BaseModel):
    company_name: str
    summary: str
    key_points: List[str]
    source_urls: List[str]

class PipelineRequest(BaseModel):
    company_name: str
    hours_ago: int = Field(default=24, ge=1, le=2000)
    max_articles: int = Field(default=5, ge=1, le=10)

class NewsItem(BaseModel):
    title: str
    link: str
    published_date: str | None
    source: str | None
    description: str | None

class NewsResponse(BaseModel):
    query: str
    items: List[NewsItem]
    total_results: int

def parse_date(date_str: str) -> datetime:
    """Parse RFC 2822 date format commonly used in RSS feeds"""
    return datetime(*email.utils.parsedate(date_str)[:6])

def get_time_ago(published_date: datetime, now: datetime) -> str:
    """Return a human-readable string of how long ago the article was published"""
    diff = now - published_date

    if diff.days > 0:
        return f"{diff.days} days ago"
    hours = diff.seconds // 3600
    if hours > 0:
        return f"{hours} hours ago"
    minutes = (diff.seconds % 3600) // 60
    return f"{minutes} minutes ago"

async def search_news(
    query: str,
    hours: int = Query(default=24, ge=1, le=2000, description="Fetch news from the last X hours")
):
    """
    Search for news articles based on the provided query and time filter.
    Returns news items from the last specified hours.
    """
    try:
        base_url = "https://news.google.com/rss/search"
        params = {
            "q": query,
            "hl": "en-US",
            "gl": "US",
            "ceid": "US:en"
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(base_url, params=params)
            response.raise_for_status()

        root = ET.fromstring(response.text)
        channel = root.find('channel')
        now = datetime.utcnow()
        cutoff_time = now - timedelta(hours=hours)

        news_items = []
        for item in channel.findall('item'):
            pub_date = item.find('pubDate')
            if pub_date is not None:
                pub_date_str = pub_date.text
                pub_date_dt = parse_date(pub_date_str)
                if pub_date_dt < cutoff_time:
                    continue
                time_ago = get_time_ago(pub_date_dt, now)
            else:
                pub_date_str = None
                time_ago = None

            title = html.unescape(item.find('title').text) if item.find('title') is not None else None
            source = title.split(' - ')[-1] if title else None
            cleaned_title = ' - '.join(title.split(' - ')[:-1]) if title else None

            news_item = NewsItem(
                title=cleaned_title,
                link=item.find('link').text if item.find('link') is not None else None,
                published_date=pub_date_str,
                source=source,
                description=html.unescape(item.find('description').text) if item.find('description') is not None else None
            )
            news_items.append(news_item)

        return NewsResponse(
            query=query,
            items=news_items,
            total_results=len(news_items)
        )

    except httpx.HTTPError as e:
        raise HTTPException(status_code=500, detail=f"Error fetching news: {str(e)}")
    except ET.ParseError as e:
        raise HTTPException(status_code=500, detail=f"Error parsing RSS feed: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

async def scrape_article_content(url: str, max_retries: int = 3) -> str:
    """Scrape the main content from a news article URL."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }

    for attempt in range(max_retries):
        try:
            session = AsyncHTMLSession()
            response = await session.get(url, headers=headers, timeout=30)

            if response.status_code != 200:
                logger.warning(f"Failed to fetch {url}: Status {response.status_code}")
                continue

            try:
                await response.html.arender(timeout=30000, sleep=2)
            except Exception as e:
                logger.warning(f"Render failed for {url}: {e}")
                content = response.html.html
            else:
                content = response.html.html

            soup = bs(content, "html.parser")
            paragraphs = soup.find_all('p')
            content = ' '.join(
                p.get_text().strip() 
                for p in paragraphs 
                if len(p.get_text().strip()) > 20
            )

            if content:
                return content

        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
            await asyncio.sleep(1)
        finally:
            await session.close()

    return ""

@app.post("/analyze-company/", response_model=CompanyReport)
async def analyze_company(request: PipelineRequest):
    """
    Complete pipeline to analyze a company based on recent news
    """
    try:
        news_response = await search_news(request.company_name, request.hours_ago)

        if not news_response.items:
            raise HTTPException(status_code=404, detail="No news articles found")

        articles = news_response.items[:request.max_articles]
        logger.info(f"Processing {len(articles)} articles for {request.company_name}")

        sem = asyncio.Semaphore(3)

        async def bounded_scrape(article):
            async with sem:
                if article.link:
                    content = await scrape_article_content(article.link)
                    if content:
                        return {
                            "title": article.title or "No title",
                            "source": article.source or "Unknown source",
                            "date": article.published_date or "No date",
                            "url": article.link,
                            "content": content
                        }
            return None

        scraping_tasks = [bounded_scrape(article) for article in articles]
        article_contents = await asyncio.gather(*scraping_tasks, return_exceptions=True)

        valid_articles = [
            article for article in article_contents 
            if article is not None 
            and not isinstance(article, Exception)
            and article.get("content")
            and len(article["content"].strip()) > 100
        ]

        if not valid_articles:
            raise HTTPException(status_code=404, detail="Could not extract valid content from any articles")

        logger.info(f"Successfully processed {len(valid_articles)} articles")

        source_urls = []
        for article in valid_articles:
            try:
                full_content = f"""
                Title: {article['title']}
                Source: {article['source']}
                Date: {article['date']}
                URL: {article['url']}

                Content:
                {article['content']}
                """

                await rag.ainsert(full_content)
                source_urls.append(article['url'])
                logger.info(f"Successfully inserted content from {article['url']}")

            except Exception as e:
                logger.error(f"Failed to insert content: {e}")
                continue

        if not source_urls:
            raise HTTPException(status_code=500, detail="Failed to process any articles through RAG system")

        report_prompt = f"""
        Based on the recent news articles about {request.company_name}, create a comprehensive analysis.
        Focus only on factual information from the provided articles about:
        1. Recent developments and announcements
        2. Current business activities and strategies
        3. Market position and performance
        4. Challenges, opportunities, and future outlook

        Format the response as a JSON object with:
        {{
            "summary": "A concise summary of the key findings (2-3 paragraphs)",
            "key_points": ["List of 3-5 most important points, each 1-2 sentences"]
        }}
        """
        
        try:
            report_result = await asyncio.wait_for(
                rag.aquery(report_prompt),
                timeout=200
            )

            print(report_result)

            report_result = re.sub(r'^```json\s*', '', report_result, flags=re.MULTILINE)
            report_result = re.sub(r'\s*```$', '', report_result, flags=re.MULTILINE)
            report_result = report_result.strip()

        except asyncio.TimeoutError:
            logger.error("Report generation timed out")
            raise HTTPException(status_code=500, detail="Report generation timed out")
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Report generation failed")

        # Parse and validate the report JSON
        try:
            report_data = json.loads(report_result)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse report result as JSON: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Failed to generate properly formatted report"
            )

        # Validate report structure
        if not isinstance(report_data, dict):
            raise HTTPException(
                status_code=500,
                detail="Invalid report format: not a JSON object"
            )

        required_fields = {'summary', 'key_points'}
        missing_fields = required_fields - set(report_data.keys())
        if missing_fields:
            raise HTTPException(
                status_code=500,
                detail=f"Invalid report format: missing fields {missing_fields}"
            )

        if not isinstance(report_data['key_points'], list):
            raise HTTPException(
                status_code=500,
                detail="Invalid report format: key_points must be a list"
            )

        # Sanitize and validate the content
        summary = str(report_data['summary']).strip()
        key_points = [str(point).strip() for point in report_data['key_points'] if point]

        if not summary or not key_points:
            raise HTTPException(
                status_code=500,
                detail="Invalid report content: empty summary or key points"
            )

        # Create the final report
        company_report = CompanyReport(
            company_name=request.company_name,
            summary=summary,
            key_points=key_points,
            source_urls=source_urls
        )

        return company_report

    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise

    except Exception as e:
        # Log unexpected errors and convert to HTTP exception
        logger.error(f"Unexpected error in analyze_company: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)