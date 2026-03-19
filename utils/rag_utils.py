# utils/rag_utils.py
# This file handles all document processing and AI querying.
# It has been rebuilt with four reliability improvements:
# 1. Section-aware chunking — splits at CA section boundaries not character count
# 2. Hybrid search — combines semantic (FAISS) and keyword (BM25) retrieval
# 3. Query decomposition — breaks complex questions into sub-queries
# 4. Cross-reference pass — asks Gemini if it needs to look anywhere else

import os
import re
import streamlit as st
import google.genai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# --- Constants ---
GEMINI_MODEL = "gemini-2.5-flash"
FAISS_INDEX_DIR = "faiss_indexes"

# Maps union names to their PDF filenames
CA_FILES = {
    "OSSTF (Secondary Teachers)": "TDSB-OSSTF-TTBU-2022-2026-Collective-Agreement-1.pdf",
    "ETFO (Elementary Teachers)": "TDSBETFO_Teachers_20222026_Collective_Agreement69.pdf",
    "ETFOOT (Elementary Occasional)": "TDSBETFOOT20222026CollectiveAgreement.pdf",
}

# Section header patterns for all three CAs
# Each pattern matches the start of a new section in the document
# Order matters — more specific patterns first
SECTION_PATTERNS = [
    # OSSTF Central: C8.00, C8.1, C10.00
    r'^C\d+\.\d+\s+[A-Z]',
    # OSSTF/ETFO Local with letter-dot: L36.0, L13.1
    r'^L\d+\.\d+\s+[A-Z]',
    # ETFO Local with letter-part: L-A.1.0., L-C.2.0.
    r'^L-[A-Z]\.\d+\.\d+\.',
    # ETFOOT Local pure numbers: 1.0.0., 12.0.0.
    r'^\d+\.\d+\.\d+\.\s+[A-Z]',
    # Letters of Agreement/Understanding
    r'^LETTER OF (AGREEMENT|UNDERSTANDING)',
    r'^LETTER OF INTENT',
    # Local Appendices
    r'^LOCAL APPENDIX',
    # Part headers
    r'^PART [IVX]+',
]

# Compile all patterns into one combined regex for efficiency
COMBINED_SECTION_PATTERN = re.compile(
    '|'.join(SECTION_PATTERNS),
    re.IGNORECASE
)


def get_embeddings_model():
    """
    Returns a sentence-transformer embeddings model.
    Converts text into vectors that capture semantic meaning.
    Cached so it only loads once per session.
    """
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )


def is_section_header(line: str) -> bool:
    """
    Returns True if a line looks like a CA section header.
    Used by the section-aware splitter to find split points.
    """
    return bool(COMBINED_SECTION_PATTERN.match(line.strip()))


def extract_section_header(text: str) -> str:
    """
    Extracts the first section header found in a chunk of text.
    Used to label each chunk with its section reference.
    Returns 'Unknown Section' if no header is found.
    """
    for line in text.split('\n'):
        if is_section_header(line):
            return line.strip()[:80]  # Cap at 80 chars for display
    return "Unknown Section"


def section_aware_split(pages: list, max_chunk_size: int = 1500) -> list:
    """
    Splits CA pages into chunks at section boundaries instead of
    arbitrary character counts.

    Key improvement over RecursiveCharacterTextSplitter:
    - Every chunk starts at a section boundary
    - Section header is always preserved at the top of each sub-chunk
    - Long sections get split into sub-chunks but keep their header

    Returns a list of LangChain Document objects.
    """
    chunks = []
    current_section_header = "Preamble"
    current_text = ""
    current_page = 0

    for page in pages:
        page_num = page.metadata.get("page", 0)
        page_text = page.page_content

        if not page_text:
            continue

        lines = page_text.split('\n')

        for line in lines:
            stripped = line.strip()

            if is_section_header(stripped):
                # We've hit a new section — save what we have so far
                if current_text.strip():
                    # If the accumulated text is too long, split it into
                    # sub-chunks but keep the section header on each one
                    sub_chunks = split_long_section(
                        current_section_header,
                        current_text,
                        current_page,
                        max_chunk_size
                    )
                    chunks.extend(sub_chunks)

                # Start a new section
                current_section_header = stripped
                current_text = stripped + '\n'
                current_page = page_num
            else:
                current_text += line + '\n'

    # Don't forget the last section
    if current_text.strip():
        sub_chunks = split_long_section(
            current_section_header,
            current_text,
            current_page,
            max_chunk_size
        )
        chunks.extend(sub_chunks)

    print(f"    → {len(chunks)} section-aware chunks created")
    return chunks


def split_long_section(header: str, text: str, page_num: int,
                         max_size: int) -> list:
    """
    If a section is longer than max_size characters, splits it into
    sub-chunks. Each sub-chunk keeps the section header at the top
    so the reference is never lost.

    Returns a list of LangChain Document objects.
    """
    chunks = []

    if len(text) <= max_size:
        # Section fits in one chunk — no splitting needed
        chunks.append(Document(
            page_content=text.strip(),
            metadata={
                "page": page_num,
                "section": header,
                "source": f"Section: {header} | Page: {page_num + 1}"
            }
        ))
    else:
        # Split into sub-chunks with 200 char overlap
        # Always prepend the section header so context is preserved
        words = text.split()
        current_chunk = header + '\n'  # Start every sub-chunk with the header
        chunk_count = 0

        for word in words:
            if len(current_chunk) + len(word) + 1 > max_size:
                # Save current sub-chunk
                chunks.append(Document(
                    page_content=current_chunk.strip(),
                    metadata={
                        "page": page_num,
                        "section": header,
                        "source": f"Section: {header} | Page: {page_num + 1} | Part {chunk_count + 1}"
                    }
                ))
                chunk_count += 1
                # Start next sub-chunk with header for context continuity
                current_chunk = header + ' (continued)\n' + word + ' '
            else:
                current_chunk += word + ' '

        # Save the last sub-chunk
        if current_chunk.strip():
            chunks.append(Document(
                page_content=current_chunk.strip(),
                metadata={
                    "page": page_num,
                    "section": header,
                    "source": f"Section: {header} | Page: {page_num + 1} | Part {chunk_count + 1}"
                }
            ))

    return chunks


@st.cache_resource
def load_or_build_index(union_name: str):
    """
    Builds or loads the FAISS index for a given union's CA.
    Now uses section-aware chunking for reliable section references.

    Returns a tuple: (faiss_retriever, all_chunks)
    We return all_chunks so we can build the BM25 retriever on demand.
    """
    index_path = os.path.join(FAISS_INDEX_DIR, union_name.replace(" ", "_"))
    embeddings = get_embeddings_model()

    if os.path.exists(index_path):
        print(f"Loading existing index for {union_name}...")
        vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        # We need to reload chunks for BM25 — load from a saved chunks file
        chunks_path = index_path + "_chunks.txt"
        chunks = load_chunks_from_disk(chunks_path)
    else:
        print(f"Building new index for {union_name}...")
        pdf_path = os.path.join("data", "agreements", CA_FILES[union_name])

        # Step 1: Load PDF
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        print(f"    → {len(pages)} pages loaded")

        # Step 2: Section-aware chunking
        chunks = section_aware_split(pages)

        # Step 3: Build FAISS vector index
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Step 4: Save FAISS index to disk
        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        vectorstore.save_local(index_path)

        # Step 5: Save chunks to disk for BM25 reloading
        chunks_path = index_path + "_chunks.txt"
        save_chunks_to_disk(chunks, chunks_path)

        print(f"    → Index saved to {index_path}")

    return vectorstore, chunks


def save_chunks_to_disk(chunks: list, path: str):
    """
    Saves chunk text to disk so BM25 can be rebuilt on reload
    without re-processing the PDF.
    Uses a simple separator-based format.
    """
    with open(path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            # Write metadata and content separated by markers
            f.write(f"<<<SECTION>>>{chunk.metadata.get('section', '')}\n")
            f.write(f"<<<PAGE>>>{chunk.metadata.get('page', 0)}\n")
            f.write(f"<<<SOURCE>>>{chunk.metadata.get('source', '')}\n")
            f.write(f"<<<CONTENT>>>\n{chunk.page_content}\n")
            f.write("<<<END>>>\n")


def load_chunks_from_disk(path: str) -> list:
    """
    Reloads chunks from disk to rebuild BM25 retriever.
    Returns a list of Document objects.
    """
    chunks = []
    if not os.path.exists(path):
        return chunks

    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    raw_chunks = content.split("<<<END>>>\n")
    for raw in raw_chunks:
        if not raw.strip():
            continue
        try:
            section = raw.split("<<<SECTION>>>")[1].split("\n")[0]
            page = int(raw.split("<<<PAGE>>>")[1].split("\n")[0])
            source = raw.split("<<<SOURCE>>>")[1].split("\n")[0]
            text = raw.split("<<<CONTENT>>>\n")[1]
            chunks.append(Document(
                page_content=text.strip(),
                metadata={"section": section, "page": page, "source": source}
            ))
        except (IndexError, ValueError):
            continue

    return chunks


def build_hybrid_retriever(vectorstore, chunks: list, k: int = 6):
    """
    Manual hybrid retrieval — combines FAISS semantic search
    and BM25 keyword search without needing EnsembleRetriever.

    Returns a callable function that takes a query string
    and returns a deduplicated list of relevant documents.
    """
    from langchain_community.retrievers import BM25Retriever

    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = k

    def retrieve(query: str) -> list:
        faiss_docs = faiss_retriever.invoke(query)
        bm25_docs = bm25_retriever.invoke(query)
        # Combine both result sets and deduplicate by content fingerprint
        seen = set()
        combined = []
        for doc in faiss_docs + bm25_docs:
            fingerprint = doc.page_content[:100]
            if fingerprint not in seen:
                seen.add(fingerprint)
                combined.append(doc)
        return combined

    return retrieve


def decompose_query(question: str, api_key: str) -> list:
    """
    Query decomposition — breaks a complex question into sub-queries.

    For simple questions this returns just the original question.
    For complex multi-topic questions it returns 2-4 focused sub-queries.

    Example:
    Input: "teacher on sick leave who is also surplus"
    Output: [
        "sick leave entitlements and conditions",
        "surplus procedures and teacher rights",
        "interaction between sick leave and surplus status"
    ]
    """
    prompt = f"""You are helping search a collective agreement document.
Break the following question into 1-4 focused search queries that would help find all relevant sections.
Each query should target a specific topic or provision.
If the question is simple and targets one topic, return just 1 query.

Return ONLY a Python list of strings, nothing else. Example format:
["query one", "query two"]

Question: {question}"""

    #genai.configure(api_key=api_key)
    #model = genai.GenerativeModel(GEMINI_MODEL)
    #response = model.generate_content(prompt)
    client= genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    try:
        # Parse the list from Gemini's response
        import ast
        queries = ast.literal_eval(response.text.strip())
        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            print(f"    → Query decomposed into {len(queries)} sub-queries: {queries}")
            return queries
    except (ValueError, SyntaxError):
        pass

    # If parsing fails, fall back to original question
    print("    → Query decomposition failed, using original question")
    return [question]

def is_substantive_chunk(text: str) -> bool:
    """
    Filters out low quality chunks that add noise to results.
    Returns True if the chunk is worth including, False if it should be dropped.

    Filters out:
    - TOC entries (short lines with lots of dots and page numbers)
    - Section headers with no content (under 100 characters)
    - Salary tables and pure numeric data
    - Part headers with no substance
    """
    # Too short to be substantive — likely a header or TOC entry
    if len(text.strip()) < 120:
        return False

    words = [w for w in text.split() if len(w) > 2]
    if len(words) < 20:
        return False

    # TOC entries have lots of dots followed by a page number
    # e.g. "C8.1 Family Medical Leave .......................... 11"
    dot_count = text.count('.')
    if dot_count > 10 and any(
        line.strip().endswith(tuple(str(i) for i in range(100)))
        for line in text.split('\n')
    ):
        return False

    # Pure salary tables — lots of numbers, little text
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) > 3:
        numeric_lines = sum(
            1 for line in lines
            if sum(c.isdigit() for c in line) > len(line) * 0.5
        )
        if numeric_lines > len(lines) * 0.6:
            return False

    return True

def get_ca_answer(union_name: str, question: str, api_key: str) -> dict:
    """
    Main function called by Tab 1 UI.
    Uses all four reliability improvements:
    1. Section-aware chunks (built at index time)
    2. Hybrid retrieval (FAISS + BM25)
    3. Query decomposition (multiple sub-queries)
    4. Cross-reference pass (second Gemini call for gaps)
    """
    # Load index and chunks
    vectorstore, chunks = load_or_build_index(union_name)

    # Build hybrid retriever
    retriever = build_hybrid_retriever(vectorstore, chunks)

    # Step 1: Decompose question into sub-queries
    sub_queries = decompose_query(question, api_key)

    # Step 2: Retrieve chunks for each sub-query, deduplicate
    seen_contents = set()
    all_relevant_docs = []

    for query in sub_queries:
        docs = retriever(query)
        for doc in docs:
            #filter out low quality chunks before adding to results
            if not is_substantive_chunk(doc.page_content):
                continue
            # Use first 100 chars as a fingerprint to avoid duplicates
            fingerprint = doc.page_content[:100]
            if fingerprint not in seen_contents:
                seen_contents.add(fingerprint)
                all_relevant_docs.append(doc)

    print(f"    → {len(all_relevant_docs)} unique chunks retrieved across all sub-queries")

    # Step 3: Build context string with section references
    context_text = "\n\n---\n\n".join([
        f"[{doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
        for doc in all_relevant_docs
    ])

    # Step 4: First answer pass
    prompt = f"""You are an expert Labour Relations advisor for the Toronto District School Board (TDSB).
Your job is to interpret collective agreement language clearly and accurately.

YouAre answering a question about the {union_name} Collective Agreement (2022-2026).

Use ONLY the collective agreement excerpts provided below.
If the answer is not clearly addressed in the excerpts, say so explicitly.

Provide your response in exactly these three sections:

1. ANSWER
A clear plain-language answer in 3-5 sentences.

2. RELEVANT SECTIONS
List every section reference (e.g. C8.1, L36.0, L-C.2.0.) that is relevant to this question,
with a one-line description of what each covers. Include ALL relevant sections you can
identify from the excerpts — do not omit any.

3. GAPS
If you believe there are provisions in the CA that would be relevant but were NOT included
in the excerpts provided, flag them here. Otherwise write "None identified."

--- COLLECTIVE AGREEMENT EXCERPTS ---
{context_text}

--- QUESTION ---
{question}"""

    client= genai.Client(api_key=api_key)
    first_response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )
    first_answer = first_response.text

    # Step 5: Cross-reference pass
    # Ask Gemini if its GAPS section suggests we should search for anything else
    cross_ref_prompt = f"""Based on this initial answer to a collective agreement question,
identify if any additional search terms should be used to find missing provisions.

Initial Answer:
{first_answer}

If the GAPS section mentions specific topics or provisions to search for,
return them as a Python list of search strings.
If no gaps were identified, return an empty list: []

Return ONLY the Python list, nothing else."""

    #cross_ref_response = model.generate_content(cross_ref_prompt)
    cross_ref_response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=cross_ref_prompt
    )

    try:
        import ast
        additional_queries = ast.literal_eval(cross_ref_response.text.strip())
        if isinstance(additional_queries, list) and additional_queries:
            print(f"    → Cross-reference pass found {len(additional_queries)} additional queries")

            # Retrieve additional chunks
            for query in additional_queries:
                docs = retriever(query)
                for doc in docs:
                    fingerprint = doc.page_content[:100]
                    if fingerprint not in seen_contents:
                        seen_contents.add(fingerprint)
                        all_relevant_docs.append(doc)

            # Build updated context and get refined answer
            context_text = "\n\n---\n\n".join([
                f"[{doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}"
                for doc in all_relevant_docs
            ])

            refined_prompt = f"""You are an expert Labour Relations advisor for the TDSB.
Based on additional CA excerpts retrieved, provide a refined answer.

Use ONLY the excerpts below. Structure your response with:

1. ANSWER
2. RELEVANT SECTIONS
3. GAPS

--- ALL COLLECTIVE AGREEMENT EXCERPTS ---
{context_text}

--- QUESTION ---
{question}"""

            refined_response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=refined_prompt
            )
            final_answer = refined_response.text
        else:
            final_answer = first_answer

    except (ValueError, SyntaxError):
        final_answer = first_answer

    return {
        "answer": final_answer,
        "sources": all_relevant_docs
    }