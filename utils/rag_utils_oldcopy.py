# utils/rag_utils.py
# This file handles all the "behind the scenes" document processing:
# loading PDFs, splitting them into chunks, and building a searchable index.

import os
import streamlit as st
#import anthropic
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# --- Custom Embeddings class using Anthropic ---
# LangChain expects an "embeddings" object to convert text into vectors.
# Anthropic doesn't have a dedicated embeddings API, so we use a lightweight
# workaround: we use a small local embedding model via a HuggingFace fallback.
# TRADEOFF: We could use OpenAI embeddings (better quality) but that adds a
# second API key/cost. For a demo, sentence-transformers works great and is free.

from langchain_community.embeddings import HuggingFaceEmbeddings

GEMINI_MODEL = "gemini-2.5-flash"

# Path where we'll save the FAISS indexes (one per union)
FAISS_INDEX_DIR = "faiss_indexes"

# Maps union short names to their PDF filenames in data/agreements/
CA_FILES = {
    "OSSTF (Secondary Teachers)": "TDSB-OSSTF-TTBU-2022-2026-Collective-Agreement-1.pdf",
    "ETFO (Elementary Teachers)": "TDSBETFO_Teachers_20222026_Collective_Agreement69.pdf",
    "ETFOOT (Elementary Occasional)": "TDSBETFOOT20222026CollectiveAgreement.pdf",
}

def get_embeddings_model():
    """
    Returns a sentence-transformer embeddings model.
    This converts text into vectors (lists of numbers) that capture meaning.
    We cache it so it only loads once per session.
    """
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2", # Small, fast, good quality for this use case
        model_kwargs={"device": "cpu"}
    )

def extract_article_headers(pages: list) -> dict:
    """
    Scans every page of the PDF and builds a map of:
    page number → most recent article header seen up to that page

    For example: {0: None, 1: "Article 3", 2: "Article 3", 3: "Article 4" ...}

    This lets us tag any chunk with the article it belongs to
    just by looking up its page number in this map.
    """
    import re

    # This will track the article number as we move through pages
    page_to_article = {}
    current_article = None

    for page in pages:
        page_num = page.metadata.get("page", 0)
        page_text = page.page_content

        # Look for patterns like "Article 7", "ARTICLE 12", "Article 12.3"
        # re.IGNORECASE handles both uppercase and mixed case headers
        matches = re.findall(
            r'Article\s+(\d+(?:\.\d+)?)',
            page_text,
            re.IGNORECASE
        )

        # If we found one or more article headers on this page,
        # update our tracker to the last one found on the page
        if matches:
            current_article = f"Article {matches[-1]}"

        # Tag this page with whatever article we're currently in
        page_to_article[page_num] = current_article

    return page_to_article

@st.cache_resource  # Streamlit caching: only builds the index once, then reuses it
def load_or_build_index(union_name: str):
    """
    For a given union, either:
    - Loads a pre-built FAISS index from disk (fast), or
    - Builds a new one from the PDF (slower, ~30 seconds first time)

    Returns a FAISS 'retriever' — an object you can ask questions to,
    and it returns the most relevant chunks from the CA.
    """
    index_path = os.path.join(FAISS_INDEX_DIR, union_name.replace(" ", "_"))

    embeddings = get_embeddings_model()

    # If we've already built and saved this index, just load it
    if os.path.exists(index_path):
        print(f"Loading existing index for {union_name}...")
        vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True  # Required flag for loading saved FAISS indexes
        )
    else:
        # Build the index from scratch
        print(f"Building new index for {union_name}...")
        pdf_path = os.path.join("data", "agreements", CA_FILES[union_name])

        # Step 1: Load the PDF — PyPDFLoader reads each page as a "Document" object
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        # Step 2: Split into chunks
        # chunk_size=800: each chunk is ~800 characters
        # chunk_overlap=100: chunks overlap by 100 chars so we don't cut a sentence
        #   right at the boundary and lose context this code has been commented out
        # and replaced with the subsequent code to incorporate article numbers in the results
        #code:
       #splitter = RecursiveCharacterTextSplitter(
       #    chunk_size=800,
       #    chunk_overlap=100
       #)
       #chunks = splitter.split_documents(pages)
       #print(f"  → {len(chunks)} chunks created from {len(pages)} pages")

        # Step 3: Convert chunks to vectors and store in FAISS
       #vectorstore = FAISS.from_documents(chunks, embeddings)
       #commented out code ends here

# Step 2: Build a map of page number → article number
        # We do this before chunking so we can tag each chunk
        page_to_article = extract_article_headers(pages)

        # Step 3: Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        chunks = splitter.split_documents(pages)
        print(f"  → {len(chunks)} chunks created from {len(pages)} pages")

        # Step 4: Tag each chunk with its article number
        # Each chunk already has a page number in its metadata from PyPDFLoader
        # We use that to look up which article it belongs to
        for chunk in chunks:
            page_num = chunk.metadata.get("page", 0)
            article = page_to_article.get(page_num)
            # Add article number to the chunk's metadata
            # If no article found for this page, it stays as None
            chunk.metadata["article"] = article

        print(f"  → Article headers tagged on all chunks")

        # Step 5: Convert chunks to vectors and store in FAISS
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Save to disk so next run is instant
        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        vectorstore.save_local(index_path)
        print(f"  → Index saved to {index_path}")

    # Return a retriever: given a query, returns top 5 most relevant chunks
    return vectorstore.as_retriever(search_kwargs={"k": 5})

def get_ca_answer(union_name: str, question: str, api_key: str) -> dict:
    """
    Main function called by the UI.
    Given a union name and a plain-language question:
    1. Retrieves the top 5 relevant CA chunks
    2. Sends them + the question to Claude
    3. Returns Claude's answer and the source chunks for citation

    Returns a dict with keys: "answer" and "sources"
    """
    # Get the retriever for this union's CA
    retriever = load_or_build_index(union_name)

    # Find the most relevant chunks for this question
    relevant_docs = retriever.invoke(question)

    # Build a context string from the retrieved chunks to send to Claude
    context_text = "\n\n---\n\n".join([
        f"[{doc.metadata.get('article', 'Article Unknown')} - Page {doc.metadata.get('page', '?') + 1}]\n{doc.page_content}"
        for doc in relevant_docs
    ])

    # Build the prompt — we tell Claude exactly what role to play and what to do
    prompt = f"""You are an expert Labour Relations advisor for the Toronto District School Board (TDSB).
Your job is to interpret collective agreement language clearly and accurately for HR managers and school administrators.

You are answering a question about the {union_name} Collective Agreement (2022-2026).

Use ONLY the collective agreement excerpts provided below to answer the question.
If the answer is not clearly addressed in the excerpts, say so — do not guess or draw on outside knowledge.

After your answer, list the article numbers or clause references you relied on.

--- COLLECTIVE AGREEMENT EXCERPTS ---
{context_text}

--- QUESTION ---
{question}

--- YOUR ANSWER ---
Provide a clear, plain-language answer (3-5 sentences). Then on a new line, cite the specific article/clause numbers you referenced."""

    # Call the Anthropic API directly (not through LangChain, simpler for this use case)
   #client = anthropic.Anthropic(api_key=api_key)
   #message = client.messages.create(
  #     model="claude-sonnet-4-20250514",
  #     max_tokens=1024,
  #     max_tokens=1024,
  #     messages=[{"role": "user", "content": prompt}]
  # )

  # return {
  #     "answer": message.content[0].text,
  #     "sources": relevant_docs # Raw chunks — we'll display these in the UI as expandable source citations
  # }
  # Configure Gemini with the API key and call the model
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)
    response = model.generate_content(prompt)

    return {
        "answer": response.text,
        "sources": relevant_docs
    }