import os
import re
import json
import time
import nltk
import faiss
import spacy
import pickle
import asyncio
import logging
import aiohttp
import uvicorn
import requests
import traceback
import numpy as np
import unicodedata
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import traceback
import asyncio
import aiohttp
from urllib.parse import quote
from typing import List
import networkx as nx
from enum import Enum
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from datetime import datetime
from bs4 import BeautifulSoup
from textblob import TextBlob
from pydantic import BaseModel
from urllib.parse import quote
from dataclasses import dataclass
from datetime import datetime, timezone
from dateutil import parser as date_parser
from requests.adapters import HTTPAdapter, Retry
from fastapi import FastAPI, HTTPException
from LightRAG.lightrag import LightRAG, QueryParam 
from typing import Optional, List, Dict, Any, Callable
from sentence_transformers import SentenceTransformer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SearchMode(Enum):
    """Available search modes for document retrieval"""
    NAIVE = "naive"     
    LOCAL = "local"     
    GLOBAL = "global"   
    HYBRID = "hybrid"   
    MIX = "mix"         

@dataclass
class QueryParam:
    """Parameters for controlling search behavior"""
    mode: str = "hybrid"           
    top_k: int = 3
    min_similarity: float = 0.6    
    context_window: int = 2
    use_knowledge_graph: bool = True

class CompanySearchResponse(BaseModel):
    company_name: str
    summary: str
    related_entities: Optional[List[str]]
    relevant_documents: List[Dict[str, Any]]
    business_insights: Dict[str, List[Dict[str, Any]]]  # New field
    metadata: Dict[str, Any]
    timestamp: str

class CompanySearchRequest(BaseModel):
    """Request model for company search"""
    company_name: str
    search_depth: Optional[int] = 2
    max_results: Optional[int] = 5
    include_related: Optional[bool] = True
    search_mode: Optional[str] = "hybrid"

class CompanySearchResponse(BaseModel):
    """Response model for company search results"""
    company_name: str
    summary: str
    related_entities: Optional[List[str]]
    relevant_documents: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    timestamp: str

class NewsArticle:
    def __init__(self, title: str, content: str, date: str, url: str):
        self.title = title
        self.content = content
        self.date = date
        self.url = url
        self.insights: List[BusinessInsight] = []

class Document:
    """Represents a document with content, embeddings, and metadata"""
    def __init__(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None
    ):
        self.content = content
        self.metadata = metadata or {}
        self.embedding = embedding
        self.id = self.metadata.get('id') or str(hash(content))
        self.timestamp = datetime.now().isoformat()
        
        # Extract any image references from HTML content
        self.images = []
        if '<img' in content:
            soup = BeautifulSoup(content, 'html.parser')
            self.images = [img['src'] for img in soup.find_all('img')]

class KnowledgeGraph:
    """Manages the knowledge graph for semantic relationships"""
    def __init__(self):
        # Initialize spaCy for NLP tasks
        self.nlp = spacy.load("en_core_web_sm")
        self.graph = nx.Graph()
        
    def extract_entities(self, text: str) -> List[tuple]:
        """Extract named entities and their relationships from text"""
        doc = self.nlp(text)
        entities = []
        
        # Extract entity pairs and their relationships
        for sent in doc.sents:
            sent_entities = [(ent.text, ent.label_) for ent in sent.ents]
            
            # Create relationships between entities in the same sentence
            for i, (ent1, label1) in enumerate(sent_entities):
                for ent2, label2 in sent_entities[i+1:]:
                    # Find the shortest dependency path between entities
                    relationship = self._get_relationship(sent, ent1, ent2)
                    entities.append((ent1, relationship, ent2))
                    
        return entities
    
    def _get_relationship(self, sent, ent1: str, ent2: str) -> str:
        """Extract the relationship between two entities based on dependency parsing"""
        # Find token spans for entities
        tokens = [token.text for token in sent]
        try:
            start1 = tokens.index(ent1.split()[0])
            start2 = tokens.index(ent2.split()[0])
            
            # Get path between entities through dependency tree
            path = []
            current = sent[start1]
            while current.text != ent2.split()[0]:
                if current.dep_ != '':
                    path.append(current.dep_)
                current = current.head
                
            return '_'.join(path) if path else 'related_to'
        except ValueError:
            return 'related_to'
    
    def add_document(self, doc: Document):
        """Add document content to knowledge graph"""
        # Extract entities and relationships
        relationships = self.extract_entities(doc.content)
        
        # Add to graph
        for ent1, rel, ent2 in relationships:
            self.graph.add_edge(ent1, ent2, relationship=rel)
            
    def find_related_concepts(self, query: str, max_depth: int = 2) -> List[str]:
        """Find concepts related to query terms in the knowledge graph"""
        query_entities = [ent.text for ent in self.nlp(query).ents]
        related_concepts = []
        
        for entity in query_entities:
            if entity in self.graph:
                # Get subgraph around entity
                subgraph = nx.ego_graph(self.graph, entity, radius=max_depth)
                related_concepts.extend(subgraph.nodes())
                
        return list(set(related_concepts))

class LightRAG:
    """Enhanced RAG system with multiple search modes and knowledge graph integration"""
    
    def __init__(
        self,
        working_dir: str,
        llm_model_func: Callable,
        embedding_dimension: int = 384,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        self.working_dir = working_dir
        self.llm_model_func = llm_model_func
        self.embedding_dimension = embedding_dimension
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.documents = {}
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.knowledge_graph = KnowledgeGraph()
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(embedding_dimension)
        
        # Load existing data if available
        self._load_state()
        
    def _load_state(self):
        """Load saved state from disk"""
        index_path = os.path.join(self.working_dir, "index.faiss")
        docs_path = os.path.join(self.working_dir, "documents.pkl")
        
        if os.path.exists(index_path) and os.path.exists(docs_path):
            self.index = faiss.read_index(index_path)
            with open(docs_path, 'rb') as f:
                self.documents = pickle.load(f)
                
    def _save_state(self):
        """Save current state to disk"""
        index_path = os.path.join(self.working_dir, "index.faiss")
        docs_path = os.path.join(self.working_dir, "documents.pkl")
        
        faiss.write_index(self.index, index_path)
        with open(docs_path, 'wb') as f:
            pickle.dump(self.documents, f)
            
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = ' '.join(words[i:i + self.chunk_size])
            chunks.append(chunk)
            
        return chunks

    def insert(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Insert document content into the system"""
        # Split into chunks
        chunks = self._chunk_text(content)
        
        # Process each chunk
        for chunk in chunks:
            # Get embedding
            embedding = self.embedding_model.encode([chunk])[0]
            
            # Create document
            doc = Document(chunk, metadata, embedding.tolist())
            
            # Store the document first
            self.documents[doc.id] = doc
            
            # Then add to index
            self.index.add(np.array([embedding], dtype=np.float32))
            
            # Add to knowledge graph
            self.knowledge_graph.add_document(doc)
            
        # Save state
        self._save_state()

    def _naive_search(self, query: str, param: QueryParam) -> List[Document]:
        """Perform keyword-based search"""
        query_terms = set(query.lower().split())
        scored_docs = []
        
        for doc in self.documents.values():
            doc_terms = set(doc.content.lower().split())
            score = len(query_terms & doc_terms) / len(query_terms)
            if score >= param.min_similarity:
                scored_docs.append((score, doc))
                
        return [doc for _, doc in sorted(scored_docs, reverse=True)[:param.top_k]]

    def _local_search(self, query: str, param: QueryParam) -> List[Document]:
        """Perform local context-aware search"""
        # Get initial matches
        query_embedding = self.embedding_model.encode([query])[0]
        distances, indices = self.index.search(
            np.array([query_embedding], dtype=np.float32),
            param.top_k
        )
        
        # Expand context window
        expanded_docs = set()
        for idx in indices[0]:
            # Instead of using reconstruct, we'll store embeddings in documents
            matching_docs = [doc for doc in self.documents.values() 
                            if np.allclose(np.array(doc.embedding), 
                                        query_embedding, 
                                        rtol=1e-5, 
                                        atol=1e-8)]
            for doc in matching_docs:
                expanded_docs.add(doc)
                # Add surrounding documents
                surrounding = self._get_surrounding_docs(doc, param.context_window)
                expanded_docs.update(surrounding)
                
        return list(expanded_docs)[:param.top_k]

    def _global_search(self, query: str, param: QueryParam) -> List[Document]:
        """Perform global semantic search"""
        query_embedding = self.embedding_model.encode([query])[0]
        distances, indices = self.index.search(
            np.array([query_embedding], dtype=np.float32),
            param.top_k
        )
        
        # Find documents with matching embeddings
        results = []
        for doc in self.documents.values():
            if np.allclose(np.array(doc.embedding), query_embedding, rtol=1e-5, atol=1e-8):
                results.append(doc)
                if len(results) >= param.top_k:
                    break
                    
        return results

    def _hybrid_search(self, query: str, param: QueryParam) -> List[Document]:
        """Combine multiple search strategies"""
        # Get results from different methods
        naive_results = self._naive_search(query, param)
        local_results = self._local_search(query, param)
        global_results = self._global_search(query, param)
        
        # Combine and deduplicate results
        all_docs = {}
        for doc in naive_results + local_results + global_results:
            if doc and doc.id not in all_docs:
                all_docs[doc.id] = doc
                
        return list(all_docs.values())[:param.top_k]

    def _mix_search(self, query: str, param: QueryParam) -> List[Document]:
        """Combine knowledge graph and vector search"""
        # Get related concepts from knowledge graph
        related_concepts = self.knowledge_graph.find_related_concepts(query)
        
        # Expand query with related concepts
        expanded_query = f"{query} {' '.join(related_concepts)}"
        
        # Perform vector search with expanded query
        expanded_results = self._global_search(expanded_query, param)
        
        return expanded_results

    def _get_doc_by_embedding(self, index: int) -> Optional[Document]:
        """
        Retrieve document by its index in FAISS.
        
        Args:
            index (int): The integer index in the FAISS database
            
        Returns:
            Optional[Document]: The corresponding document if found, None otherwise
        """
        try:
            # Convert to numpy int64 to ensure correct type for FAISS
            faiss_idx = np.int64(index)
            embedding = self.index.reconstruct(faiss_idx)
            
            # Search through documents with numpy's allclose for float comparison
            for doc in self.documents.values():
                if np.allclose(np.array(doc.embedding), embedding, rtol=1e-5):
                    return doc
            return None
        except RuntimeError:
            # Handle case where index is out of bounds
            return None

    def _get_surrounding_docs(
        self,
        doc: Document,
        window: int
    ) -> List[Document]:
        """Get documents surrounding the given document"""
        # This is a simple implementation that could be enhanced based on
        # document relationships, timestamps, or other metadata
        doc_ids = list(self.documents.keys())
        try:
            current_idx = doc_ids.index(doc.id)
            start_idx = max(0, current_idx - window)
            end_idx = min(len(doc_ids), current_idx + window + 1)
            
            return [
                self.documents[doc_id]
                for doc_id in doc_ids[start_idx:end_idx]
            ]
        except ValueError:
            return []
    
    def _preprocess_document(self, content: str) -> str:
        """
        Preprocess document content to improve quality of embeddings and search.
        Handles common text cleanup tasks while preserving meaningful structure.
        """
        # Remove excessive whitespace while preserving paragraph breaks
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'\n\s*\n', '\n\n', content)
        
        # Remove special characters but keep important punctuation
        content = re.sub(r'[^\w\s.,!?;:()\-\'\"]+', ' ', content)
        
        # Normalize quotes and dashes
        content = content.replace('"', '"').replace('"', '"')
        content = content.replace('--', '—')
        
        return content.strip()

    def _analyze_document_structure(self, content: str) -> Dict[str, Any]:
        """
        Analyze document structure to identify sections, headings, and key elements.
        This helps in maintaining context during chunking and retrieval.
        """
        structure = {
            'sections': [],
            'headings': [],
            'paragraphs': [],
            'metadata': {}
        }
        
        # Split into lines while preserving empty lines
        lines = content.split('\n')
        current_section = {'heading': '', 'content': []}
        
        for line in lines:
            # Identify headings (lines that are shorter and often capitalized)
            if len(line.strip()) > 0 and len(line.strip()) < 100 and line.strip().isupper():
                if current_section['content']:
                    structure['sections'].append(current_section)
                current_section = {'heading': line.strip(), 'content': []}
                structure['headings'].append(line.strip())
            else:
                current_section['content'].append(line)
                
        # Add the last section
        if current_section['content']:
            structure['sections'].append(current_section)
            
        # Analyze paragraph structure
        paragraphs = [p for p in content.split('\n\n') if p.strip()]
        structure['paragraphs'] = paragraphs
        
        # Extract potential metadata (e.g., dates, authors, references)
        structure['metadata'] = self._extract_metadata(content)
        
        return structure

    def _extract_metadata(self, content: str) -> Dict[str, Any]:
        """
        Extract metadata from document content using pattern matching and NLP.
        Identifies dates, authors, references, and other structured information.
        """
        metadata = {}
        
        # Extract dates using regex patterns
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # ISO format
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # Common date format
            r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}'
        ]
        
        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, content))
        if dates:
            metadata['dates'] = dates
            
        # Extract potential author names (capitalized names following common patterns)
        author_pattern = r'(?:By|Author[s]?:?|Written by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        authors = re.findall(author_pattern, content)
        if authors:
            metadata['authors'] = authors
            
        # Extract references or citations
        ref_pattern = r'\[[\d,\s]+\]|\(\d{4}\)'
        references = re.findall(ref_pattern, content)
        if references:
            metadata['references'] = references
            
        return metadata

    def _smart_chunking(self, content: str) -> List[Dict[str, Any]]:
        """
        Implement smart document chunking that preserves context and structure.
        Creates chunks that maintain semantic coherence and document hierarchy.
        """
        # Analyze document structure
        structure = self._analyze_document_structure(content)
        chunks = []
        
        # Process each section while maintaining context
        for section in structure['sections']:
            section_content = '\n'.join(section['content'])
            section_chunks = self._create_contextual_chunks(
                section_content,
                section['heading']
            )
            chunks.extend(section_chunks)
            
        # Add document-level metadata to all chunks
        for chunk in chunks:
            chunk['metadata'].update(structure['metadata'])
            
        return chunks

    def _create_contextual_chunks(
        self,
        content: str,
        heading: str
    ) -> List[Dict[str, Any]]:
        """
        Create chunks that maintain context and semantic boundaries.
        Avoids breaking apart coherent ideas or important relationships.
        """
        chunks = []
        sentences = nltk.sent_tokenize(content)
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            # Check if adding this sentence would exceed chunk size
            if current_length + sentence_length > self.chunk_size:
                # Create new chunk if we have content
                if current_chunk:
                    chunks.append({
                        'content': ' '.join(current_chunk),
                        'metadata': {
                            'heading': heading,
                            'position': len(chunks),
                            'complete_sentence': True
                        }
                    })
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
                
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append({
                'content': ' '.join(current_chunk),
                'metadata': {
                    'heading': heading,
                    'position': len(chunks),
                    'complete_sentence': True
                }
            })
            
        return chunks

    def insert(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Enhanced document insertion with preprocessing and smart chunking.
        Maintains document structure and context during processing.
        """
        # Preprocess content
        processed_content = self._preprocess_document(content)
        
        # Create smart chunks
        chunks = self._smart_chunking(processed_content)
        
        # Process each chunk
        for chunk_data in chunks:
            # Combine chunk metadata with provided metadata
            combined_metadata = chunk_data['metadata']
            if metadata:
                combined_metadata.update(metadata)
            
            # Get embedding for chunk
            chunk_embedding = self.embedding_model.encode(
                [chunk_data['content']]
            )[0]
            
            # Create document
            doc = Document(
                chunk_data['content'],
                combined_metadata,
                chunk_embedding.tolist()
            )
            
            # Add to index and storage
            self.index.add(np.array([chunk_embedding], dtype=np.float32))
            self.documents[doc.id] = doc
            
            # Add to knowledge graph
            self.knowledge_graph.add_document(doc)
            
        # Save state
        self._save_state()

    def _rerank_results(
        self,
        query: str,
        documents: List[Document],
        param: QueryParam
    ) -> List[Document]:
        """
        Rerank search results using multiple criteria to improve relevance.
        Considers semantic similarity, freshness, and document structure.
        """
        scored_docs = []
        query_embedding = self.embedding_model.encode([query])[0]
        
        for doc in documents:
            # Calculate semantic similarity score
            semantic_score = 1 - np.linalg.norm(
                np.array(doc.embedding) - query_embedding
            )
            
            # Calculate freshness score (if timestamp exists)
            freshness_score = 0.0
            if doc.timestamp:
                age = datetime.now() - datetime.fromisoformat(doc.timestamp)
                freshness_score = 1.0 / (1.0 + age.days)
            
            # Calculate structural relevance score
            structural_score = self._calculate_structural_score(doc, query)
            
            # Combine scores with weights
            final_score = (
                0.6 * semantic_score +
                0.2 * freshness_score +
                0.2 * structural_score
            )
            
            scored_docs.append((final_score, doc))
            
        # Sort by final score and return top results
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        return [doc for _, doc in scored_docs[:param.top_k]]

    def _calculate_structural_score(
        self,
        doc: Document,
        query: str
    ) -> float:
        """
        Calculate structural relevance score based on document hierarchy
        and query term positions.
        """
        score = 0.0
        
        # Check if query terms appear in headings
        if doc.metadata.get('heading'):
            heading_matches = sum(
                term.lower() in doc.metadata['heading'].lower()
                for term in query.split()
            )
            score += 0.5 * (heading_matches / len(query.split()))
            
        # Consider position in document
        if 'position' in doc.metadata:
            # Slight preference for earlier sections
            position_score = 1.0 / (1.0 + doc.metadata['position'])
            score += 0.2 * position_score
            
        # Add bonus for complete sentences
        if doc.metadata.get('complete_sentence'):
            score += 0.1
            
        return score
    
    def _get_search_function(self, mode: str):
        """
        Get the appropriate search function based on the specified mode.
        
        Args:
            mode (str): The search mode to use (naive, local, global, hybrid, or mix)
            
        Returns:
            function: The corresponding search function
        """
        search_methods = {
            SearchMode.NAIVE.value: self._naive_search,
            SearchMode.LOCAL.value: self._local_search,
            SearchMode.GLOBAL.value: self._global_search,
            SearchMode.HYBRID.value: self._hybrid_search,
            SearchMode.MIX.value: self._mix_search
        }
        
        # Default to hybrid search if mode not found
        return search_methods.get(mode, self._hybrid_search)

    def query(
        self,
        query: str,
        param: Optional[QueryParam] = None
    ) -> str:
        """
        Enhanced query processing with result reranking and context management.
        Provides more relevant and coherent responses.
        """
        param = param or QueryParam()
        
        # Select and execute search method
        search_func = self._get_search_function(param.mode)
        initial_results = search_func(query, param)
        
        # Rerank results
        reranked_results = self._rerank_results(query, initial_results, param)
        
        # Build context with document structure
        context = self._build_structured_context(reranked_results)
        
        # Generate response using LLM
        prompt = self._create_structured_prompt(query, context)
        return self.llm_model_func(prompt)

    def _build_structured_context(self, documents: List[Document]) -> str:
        """
        Create structured context from retrieved documents.
        Maintains document hierarchy and relationships.
        """
        context_parts = []
        
        # Group documents by section/heading
        sections = {}
        for doc in documents:
            heading = doc.metadata.get('heading', 'General')
            if heading not in sections:
                sections[heading] = []
            sections[heading].append(doc)
            
        # Build structured context
        for heading, docs in sections.items():
            context_parts.append(f"Section: {heading}")
            for doc in docs:
                context_parts.append(doc.content)
                
        return "\n\n".join(context_parts)

    def _create_structured_prompt(self, query: str, context: str) -> str:
        """
        Create a structured prompt that guides the LLM to provide
        more accurate and contextual responses.
        """
        return f"""
        Analyze the following content and answer the question.
        Pay attention to document structure and relationships between sections.
        
        Document Structure:
        {context}
        
        Question:
        {query}
        
        Provide a detailed answer that:
        1. Directly addresses the question
        2. Uses specific evidence from the provided context
        3. Maintains awareness of document structure and relationships
        4. Acknowledges any limitations in the available information
        
        Answer:
        """

    def get_analytics(self) -> Dict[str, Any]:
        """
        Provide analytics about system usage and performance.
        Helps in monitoring and optimizing the system.
        """
        return {
            'total_documents': len(self.documents),
            'total_tokens': sum(
                len(doc.content.split())
                for doc in self.documents.values()
            ),
            'average_chunk_size': np.mean([
                len(doc.content.split())
                for doc in self.documents.values()
            ]),
            'knowledge_graph_size': len(self.knowledge_graph.graph.nodes()),
            'index_size': self.index.ntotal,
            'metadata_coverage': sum(
                1 for doc in self.documents.values()
                if doc.metadata
            ) / len(self.documents) if self.documents else 0
        }

class CompanySearchAPI:
    def __init__(self, working_dir: str, llm_model_func: callable, host: str = "0.0.0.0", port: int = 8000):
        self.app = FastAPI()
        self.rag = LightRAG(working_dir=working_dir, llm_model_func=llm_model_func)
        self.scraper = GoogleNewsScraper()
        self.setup_routes()
        self.host = host
        self.port = port

    def setup_routes(self):
        @self.app.post("/search_company", response_model=CompanySearchResponse)
        async def search_company(request: CompanySearchRequest):
            try:
                print(f"SEARCHING FOR COMPANY: {request.company_name}")
                
                # Step 1: Scrape latest news
                print("Starting news article scraping...")
                articles = await self.scraper.get_news_articles(request.company_name)
                print(f"Scraped {len(articles)} articles for company: {request.company_name}")
                
                # Step 2: Organize insights by type
                print("Organizing insights from articles...")
                insights_by_type = {insight_type.value: [] for insight_type in BusinessInsightType}
                for article in articles:
                    print(f"Processing article insights: {article.title}")
                    for insight in article.insights:
                        insight_dict = {
                            "summary": insight.summary,
                            "date": insight.date,
                            "source_url": insight.source_url,
                            "sentiment": insight.sentiment
                        }
                        insights_by_type[insight.type.value].append(insight_dict)
                print("Insights organized successfully.")
                
                # Step 3: Build query and get summary
                print("Building company query and retrieving summary...")
                param = QueryParam(
                    mode=request.search_mode,
                    top_k=request.max_results,
                    context_window=request.search_depth,
                    use_knowledge_graph=request.include_related
                )
                query = self._build_company_query(request.company_name)
                summary = self.rag.query(query, param)
                print("Summary retrieved successfully.")
                
                # Step 4: Retrieve related entities (if applicable)
                related_entities = []
                if request.include_related:
                    print(f"Fetching related entities for: {request.company_name}")
                    related_entities = self.rag.knowledge_graph.find_related_concepts(
                        request.company_name,
                        max_depth=request.search_depth
                    )
                    print(f"Retrieved {len(related_entities)} related entities.")
                
                # Step 5: Get relevant documents
                print("Retrieving relevant documents...")
                relevant_docs = self._get_relevant_documents(request.company_name, param)
                print(f"Retrieved {len(relevant_docs)} relevant documents.")
                
                # Step 6: Prepare and send response
                print("Preparing response...")
                response = CompanySearchResponse(
                    company_name=request.company_name,
                    summary=summary,
                    related_entities=related_entities,
                    relevant_documents=[{
                        'content': doc.content,
                        'metadata': doc.metadata,
                        'timestamp': doc.timestamp
                    } for doc in relevant_docs],
                    business_insights=insights_by_type,
                    metadata=self._get_search_metadata(request),
                    timestamp=datetime.now().isoformat()
                )
                print("Response prepared successfully.")
                return response
            
            except Exception as e:
                print(f"Error processing request for company: {request.company_name}. Error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

        @self.app.post("/insert")
        async def insert_company_data(
            company_name: str,
            content: str,
            metadata: Optional[Dict[str, Any]] = None
        ):
            """
            Insert new company-related content into the system
            """
            try:
                # Add company name to metadata
                full_metadata = metadata or {}
                full_metadata['company_name'] = company_name
                
                # Insert content
                self.rag.insert(content, full_metadata)
                
                return {"status": "success", "message": "Content inserted successfully"}
                
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error inserting content: {str(e)}"
                )

        @self.app.get("/analytics")
        async def get_analytics():
            """
            Get system analytics and statistics
            """
            try:
                return self.rag.get_analytics()
                
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error retrieving analytics: {str(e)}"
                )

    def _build_company_query(self, company_name: str) -> str:
        """
        Build a comprehensive query for company information
        """
        return f"""
        Provide a detailed analysis of {company_name}, including:
        - Main business activities and industry
        - Key products or services
        - Market position and competitors
        - Recent developments or news
        - Any significant relationships or partnerships
        
        Use only the information available in the provided context.
        """

    def _get_relevant_documents(
        self,
        company_name: str,
        param: QueryParam
    ) -> List[Any]:
        """
        Get relevant documents for the company using different search modes
        """
        # Try different search modes to get comprehensive results
        docs_hybrid = set(
            self.rag._hybrid_search(company_name, param)
        )
        
        if param.use_knowledge_graph:
            docs_kg = set(
                self.rag._mix_search(company_name, param)
            )
            return list(docs_hybrid | docs_kg)
            
        return list(docs_hybrid)

    def _get_search_metadata(
        self,
        request: CompanySearchRequest
    ) -> Dict[str, Any]:
        """
        Collect metadata about the search process
        """
        return {
            'search_parameters': {
                'search_depth': request.search_depth,
                'max_results': request.max_results,
                'include_related': request.include_related,
                'search_mode': request.search_mode
            },
            'system_info': {
                'index_size': self.rag.index.ntotal,
                'knowledge_graph_nodes': len(self.rag.knowledge_graph.graph.nodes())
            }
        }

    def run(self):
        """
        Start the API server
        """
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port
        )

class BusinessInsightType(Enum):
    FUNDING = "Funding"
    HIRING = "Hiring"
    INNOVATION = "Innovation"
    INITIATIVE = "Initiative"
    INVESTMENT = "Investment"
    AWARD = "Award"
    LAUNCH = "Launch"
    ARR = "ARR"
    ACQUISITION = "Acquisition"
    STARTUP = "Startup"

@dataclass
class BusinessInsight:
    type: BusinessInsightType
    summary: str
    date: str
    source_url: str
    sentiment: float

class InsightExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        
        # Define keywords for each insight type
        self.insight_patterns = {
            BusinessInsightType.FUNDING: r"funding|raised|investment round|series [A-Z]|seed round",
            BusinessInsightType.HIRING: r"hiring|recruitment|new role|job|position|talent",
            BusinessInsightType.INNOVATION: r"innovation|breakthrough|patent|R&D|research|development",
            BusinessInsightType.INITIATIVE: r"initiative|program|campaign|strategy|partnership",
            BusinessInsightType.INVESTMENT: r"invest|stake|portfolio|acquisition|merger",
            BusinessInsightType.AWARD: r"award|recognition|prize|honor|achievement",
            BusinessInsightType.LAUNCH: r"launch|release|introduce|unveil|announce",
            BusinessInsightType.ARR: r"revenue|ARR|annual recurring revenue|sales|growth",
            BusinessInsightType.ACQUISITION: r"acquire|buyout|takeover|purchase|merge",
            BusinessInsightType.STARTUP: r"startup|founded|seed|early-stage|venture"
        }
        
        # Compile patterns
        self.compiled_patterns = {
            insight_type: re.compile(pattern, re.IGNORECASE)
            for insight_type, pattern in self.insight_patterns.items()
        }

    def extract_insights(self, article: NewsArticle) -> List[BusinessInsight]:
        insights = []
        
        # Analyze sentiment
        blob = TextBlob(article.content)
        sentiment = blob.sentiment.polarity
        
        # Process text with spaCy
        doc = self.nlp(article.content)
        
        # Extract insights for each type
        for insight_type, pattern in self.compiled_patterns.items():
            if pattern.search(article.content):
                # Find the relevant sentence(s)
                relevant_sentences = []
                for sent in doc.sents:
                    if pattern.search(sent.text):
                        relevant_sentences.append(sent.text)
                
                if relevant_sentences:
                    summary = " ".join(relevant_sentences)
                    insight = BusinessInsight(
                        type=insight_type,
                        summary=summary,
                        date=article.date,
                        source_url=article.url,
                        sentiment=sentiment
                    )
                    insights.append(insight)
        
        return insights
    
class GoogleNewsScraper:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.insight_extractor = InsightExtractor()

        # Set up Selenium WebDriver
        self.chrome_options = Options()
        # self.chrome_options.add_argument("--headless")  # Run headless browser
        self.driver = webdriver.Chrome(options=self.chrome_options)

    async def get_article_content(self, url: str) -> str:
        """Enhanced article content extraction with improved Google News handling."""
        try:
            print(f"Starting content extraction for URL: {url}")
            
            # Handle Google News URLs with Selenium to follow redirects
            real_url = url
            if 'news.google.com' in url:
                print(f"Detected Google News URL. Using Selenium to follow redirects: {url}")
                
                # Use Selenium to get the final URL
                self.driver.get(url)
                real_url = self.driver.current_url
                print(f"Final URL after redirects: {real_url}")
                
                # If still on Google News, try to extract the real URL
                if 'news.google.com' in real_url:
                    content = self.driver.page_source
                    soup = BeautifulSoup(content, 'html.parser')
                    # Look for redirect links in Google's format
                    article_link = soup.find('a', href=True)
                    if article_link and not article_link['href'].startswith('/'):
                        real_url = article_link['href']
                        print(f"Extracted actual article URL: {real_url}")

            # Enhanced headers for better site compatibility
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'DNT': '1',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'cross-site',
                'Cache-Control': 'no-cache',
            }

            # Fetch the actual article using aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(real_url, headers=headers, timeout=45) as response:
                    print(f"HTTP status from {real_url}: {response.status}")
                    if response.status != 200:
                        print(f"Failed to fetch content. Status: {response.status}")
                        return ""
                    
                    content = await response.text()
                    print(f"Retrieved content length: {len(content)} characters")

                    # Use html5lib parser for better handling of malformed HTML
                    soup = BeautifulSoup(content, 'html5lib')
                    
                    # More aggressive cleaning of unwanted elements
                    for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 
                                            'aside', 'iframe', 'form', 'noscript', 'meta',
                                            'time', 'button', 'svg', 'select']):
                        element.decompose()
                    
                    # Remove common ad and popup containers
                    for element in soup.find_all(class_=lambda x: x and any(term in x.lower() 
                        for term in ['social', 'newsletter', 'subscribe', 'ad', 'popup', 'modal',
                                'overlay', 'banner', 'promotion'])):
                        element.decompose()

                    # Try multiple strategies to find the content
                    
                    # 1. Look for structured article content
                    article_content = None
                    content_selectors = [
                        'article', 'main', '[role="main"]', '[role="article"]',
                        '.article-content', '.post-content', '.story-body',
                        '#article-body', '.article-body', '.story-content',
                        '.content-body', '.entry-content', '.post-body',
                        '[itemprop="articleBody"]', '[data-testid="article-body"]'
                    ]

                    for selector in content_selectors:
                        content_area = soup.select_one(selector)
                        if content_area:
                            article_content = self._extract_text_from_element(content_area)
                            if len(article_content) > 500:  # Minimum content threshold
                                return article_content

                    # 2. Find largest text container with improved scoring
                    containers = soup.find_all(['div', 'article', 'section'])
                    best_container = self._find_best_container(containers)
                    if best_container and len(best_container) > 500:
                        return best_container

                    # 3. Fallback: Get all paragraph text if other methods fail
                    paragraphs = soup.find_all('p')
                    if paragraphs:
                        text_blocks = []
                        for p in paragraphs:
                            text = p.get_text(strip=True)
                            if len(text) > 50:  # Filter out short paragraphs
                                text_blocks.append(text)
                        
                        if text_blocks:
                            return ' '.join(text_blocks)

                    print(f"No content extracted from {real_url}")
                    return ""

        except Exception as e:
            print(f"Error during content extraction for URL {url}: {str(e)}")
            print(f"Full traceback: {traceback.format_exc()}")
            return ""

    def _extract_text_from_element(self, element):
        """Extract meaningful text from an element while preserving some structure."""
        text_blocks = []
        for child in element.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
            text = child.get_text(strip=True)
            if len(text) > 30:  # Filter out short snippets
                text_blocks.append(text)
        return ' '.join(text_blocks)

    def _find_best_container(self, containers):
        """Find the container most likely to contain the article content."""
        scored_containers = []
        
        for container in containers:
            # Get text content
            text = ' '.join(p.get_text(strip=True) for p in container.find_all('p'))
            
            if len(text) > 200:
                # Scoring factors
                words = len(text.split())
                paragraphs = len(container.find_all('p'))
                links = len(container.find_all('a'))
                images = len(container.find_all('img'))
                
                # Improved scoring system
                content_score = (
                    (words * 1.0) +           # Base score from words
                    (paragraphs * 50) +       # Bonus for structured paragraphs
                    (images * 30) -           # Small bonus for images
                    (links * 25)              # Penalty for too many links
                )
                
                scored_containers.append((text, content_score))
        
        if scored_containers:
            # Return text from highest scoring container
            return max(scored_containers, key=lambda x: x[1])[0]
        
        return None

    async def get_news_articles(self, company_name: str, days: int = 7, max_articles: int = 50) -> List[NewsArticle]:
        """Get news articles with detailed logging and improved redirect handling."""
        print(f"Fetching news articles for company: {company_name} (max {max_articles} articles)...")
        articles = []
        encoded_company = quote(company_name)
        feeds = [
            f"https://news.google.com/news/rss/search?q={encoded_company}&hl=en-US&gl=US&ceid=US:en",
            f"https://news.google.com/rss/search?q={encoded_company}&hl=en-US&gl=US&ceid=US:en",
        ]

        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests

        async def process_article(item):
            async with semaphore:
                try:
                    # Log the raw item for debugging
                    print("Raw RSS item:", item)

                    title = item.title.text if item.title else None
                    link = item.link.string if item.link and item.link.string else None
                    pub_date = item.pubDate.text if item.pubDate else "Unknown date"

                    if not link:
                        # Attempt to extract from description
                        description = item.description.text if item.description else ""
                        match = re.search(r'href="(https?://.*?)"', description)
                        if match:
                            link = match.group(1)
                    
                    if not title or not link:
                        print(f"Skipping article due to missing data. Title: {title}, Link: {link}")
                        return None

                    print(f"Processing article: {title}, Link: {link}, Published: {pub_date}")
                    content = await self.get_article_content(link)

                    if not content:
                        print(f"Skipping article due to empty content. Title: {title}")
                        return None

                    article = NewsArticle(
                        title=title,
                        content=content,
                        date=pub_date,
                        url=link
                    )

                    # Extract insights
                    article.insights = self.insight_extractor.extract_insights(article)
                    print(f"Finished processing article: {title}")
                    return article

                except Exception as e:
                    print(f"Error processing article. Exception: {str(e)}")
                    return None

                finally:
                    await asyncio.sleep(1)  # Rate limiting

        for feed_url in feeds:
            if len(articles) >= max_articles:
                print(f"Reached max article limit: {max_articles}. Skipping remaining feeds.")
                break

            try:
                print(f"Fetching RSS feed: {feed_url}")
                async with aiohttp.ClientSession() as session:
                    async with session.get(feed_url) as response:
                        print(f"Response status for feed {feed_url}: {response.status}")
                        if response.status != 200:
                            print(f"Failed to fetch feed: {feed_url}. Status: {response.status}")
                            continue

                        content = await response.text()
                        print(f"RSS feed content fetched successfully from: {feed_url}")
                        soup = BeautifulSoup(content, 'html.parser')
                        items = soup.find_all('item')

                        if not items:
                            print(f"No items found in feed: {feed_url}")
                            continue

                        print(f"Found {len(items)} items in feed: {feed_url}")
                        tasks = [process_article(item) for item in items]
                        results = await asyncio.gather(*tasks)
                        valid_articles = [r for r in results if r is not None]
                        articles.extend(valid_articles)
                        print(f"Added {len(valid_articles)} articles from feed: {feed_url}")

            except Exception as e:
                print(f"Error processing feed {feed_url}: {str(e)}")
                continue

        print(f"Successfully processed {len(articles)} articles in total.")
        return articles[:max_articles]
        
    def _clean_content(self, text: str) -> str:
        """Clean and normalize article content with improved handling"""
        # Remove HTML tags and entities
        text = BeautifulSoup(text, 'html.parser').get_text()
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Remove control characters
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C')
        
        # Normalize quotes and dashes
        text = text.replace('"', '"').replace('"', '"').replace('—', '-')
        
        # Remove repetitive punctuation
        text = re.sub(r'([.!?])\1+', r'\1', text)
        
        # Ensure proper spacing around punctuation
        text = re.sub(r'\s*([.,!?])\s*', r'\1 ', text)
        
        return text.strip()

    async def insert_articles(self, company_name: str, articles: List[NewsArticle]):
        """Insert articles into the LightRAG system with improved error handling"""
        async with aiohttp.ClientSession() as session:
            for article in articles:
                metadata = {
                    'company_name': company_name,
                    'article_title': article.title,
                    'article_date': article.date,
                    'source_url': article.url,
                    'content_type': 'news_article'
                }
                
                try:
                    async with session.post(
                        f"{self.base_url}/insert",
                        params={"company_name": company_name},
                        json={
                            "content": article.content,
                            "metadata": metadata
                        },
                        timeout=30
                    ) as response:
                        if response.status == 200:
                            print(f"Successfully inserted article: {article.title}")
                        else:
                            print(f"Failed to insert article: {article.title}")
                            print(f"Error: {await response.text()}")
                    
                except Exception as e:
                    print(f"Error inserting article: {str(e)}")
                
                await asyncio.sleep(1)

# Example usage
if __name__ == "__main__":
    # Define a simple LLM function for testing
    def simple_llm(prompt: str) -> str:
        return "This is a placeholder LLM response"
    
    # Initialize and run the API
    api = CompanySearchAPI(
        working_dir="./data",
        llm_model_func=simple_llm
    )
    api.run()
