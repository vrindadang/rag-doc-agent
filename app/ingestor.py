import fitz
import base64
import httpx
from openai import AzureOpenAI
from supabase import create_client
from app.config import Config

# AzureOpenAI requires: api_key, azure_endpoint, api_version
# These come from your Azure resource's "Keys and Endpoint" page
azure_client = AzureOpenAI(
    api_key=Config.AZURE_OPENAI_API_KEY,
    azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
    api_version=Config.AZURE_OPENAI_API_VERSION,
)
supabase_write = create_client(Config.SUPABASE_URL, Config.SUPABASE_SERVICE_KEY)


def _get_chat_deployment_name() -> str | None:
    """Return the chat deployment to use for vision analysis."""
    return Config.CHAT_DEPLOYMENT or Config.AZURE_CHAT_DEPLOYMENT


def _render_page_to_data_uri(page: fitz.Page, zoom: float = 1.7) -> str:
    """
    Render a PDF page to PNG and return a data URI.

    Azure OpenAI vision accepts image URLs; a data URI lets us keep everything in memory
    without writing temp files.
    """
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    png_bytes = pix.tobytes("png")
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _extract_table_text_with_pymupdf(page: fitz.Page) -> list[str]:
    """
    Try extracting table rows directly from PDF structure.

    This is fast and accurate on digitally-generated PDFs. If API support differs by
    PyMuPDF build or table parsing fails, we gracefully return an empty list.
    """
    if not hasattr(page, "find_tables"):
        return []

    extracted_tables = []
    try:
        tables = page.find_tables()
        if not tables:
            return []

        for table in tables:
            rows = table.extract()
            if not rows:
                continue
            table_lines = []
            for row in rows:
                cells = [str(cell).strip() if cell is not None else "" for cell in row]
                if any(cells):
                    table_lines.append(" | ".join(cells))
            if table_lines:
                extracted_tables.append("\n".join(table_lines))
    except Exception:
        # Non-fatal: keep ingestion running even if table extraction fails.
        return []

    return extracted_tables


def _extract_visual_summary_with_vision(page: fitz.Page, page_num: int) -> str:
    """
    Ask a vision-capable chat model to summarize image-heavy content.

    Output format is plain text with sections for tables, charts/graphs, and other
    notable visuals so it can be embedded and retrieved later.
    """
    deployment = _get_chat_deployment_name()
    if not deployment:
        return ""

    page_image_uri = _render_page_to_data_uri(page)

    vision_prompt = (
        "Analyze this PDF page image for retrieval indexing. "
        "Return concise plain text with these headers exactly:\n"
        "Table Data:\n"
        "Graphs and Charts:\n"
        "Other Visual Content:\n"
        "Rules:\n"
        "- Extract table values and headers when visible.\n"
        "- Describe graph/chart trends and approximate values if readable.\n"
        "- Include labels, legends, and units if present.\n"
        "- If a section has no content, write 'None'."
    )

    try:
        response = azure_client.chat.completions.create(
            model=deployment,
            temperature=0,
            max_tokens=1200,
            messages=[
                {
                    "role": "system",
                    "content": "You convert document visuals into searchable factual text.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Page number: {page_num}\n{vision_prompt}"},
                        {"type": "image_url", "image_url": {"url": page_image_uri}},
                    ],
                },
            ],
        )

        if not response.choices:
            return ""

        message = response.choices[0].message
        return (message.content or "").strip()
    except Exception:
        # Vision should enhance extraction, not block text ingestion.
        return ""


def extract_text_from_pdf(pdf_bytes: bytes) -> list[dict]:
    """
    Open PDF from bytes and extract text page by page.

    Returns:
    [{"page_num": 1, "text": "..."}, {"page_num": 2, "text": "..."}, ...]

    We use fitz.open() with stream= to open from bytes (not a file path).
    We skip blank pages using .strip() check.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page_index, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append({
                "page_num": page_index + 1,
                "text": text
            })
    return pages


def extract_multimodal_content_from_pdf(pdf_bytes: bytes) -> list[dict]:
    """
    Extract text + tables + image/graph descriptions for each page.

    Flow per page:
    1) Get native text (good for normal/selectable PDFs)
    2) Try structured table extraction via PyMuPDF
    3) If visuals exist (or text is sparse), ask vision model to summarize visuals

    Returns a list like:
    [{"page_num": 1, "text": "...combined searchable content..."}, ...]
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []

    for page_index, page in enumerate(doc):
        page_num = page_index + 1
        native_text = page.get_text().strip()

        # Structured table extraction from digital PDFs.
        extracted_tables = _extract_table_text_with_pymupdf(page)
        table_text = "\n\n".join(extracted_tables).strip()

        # Image presence helps decide whether to run vision analysis.
        image_blocks = page.get_images(full=True)
        has_images = len(image_blocks) > 0

        # Run vision when visuals likely matter or text is missing.
        visual_summary = ""
        if has_images or table_text or len(native_text) < 120:
            visual_summary = _extract_visual_summary_with_vision(page, page_num)

        combined_sections = []
        if native_text:
            combined_sections.append("Page Text:\n" + native_text)
        if table_text:
            combined_sections.append("Detected Tables:\n" + table_text)
        if visual_summary:
            combined_sections.append("Vision Summary:\n" + visual_summary)

        combined_text = "\n\n".join(combined_sections).strip()
        if combined_text:
            pages.append({"page_num": page_num, "text": combined_text})

    return pages


def split_into_chunks(pages: list[dict], chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """
    Split each page's text into smaller overlapping chunks.
    This helps with retrieval relevance and fits within embedding limits.
    """
    all_chunks = []
    for page in pages:
        text = page["text"]
        chunk_index = 0
        start = 0
        while start < len(text):
            chunk_text = text[start : start + chunk_size].strip()
            if chunk_text:
                all_chunks.append({
                    "page_num":    page["page_num"],
                    "chunk_index": chunk_index,
                    "text":        chunk_text,
                })
                chunk_index += 1
            start += chunk_size - overlap
    return all_chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Send a list of texts to Azure OpenAI and get back embeddings.

    Each embedding is a list of 1536 floats.
    We send all texts in one API call (batching) for efficiency.
    """
    response = azure_client.embeddings.create(
        model=Config.EMBEDDING_DEPLOYMENT,
        input=texts,
    )
    return [item.embedding for item in response.data]


def store_chunks_in_supabase(filename: str, chunks: list[dict], embeddings: list[list[float]]):
    """
    Insert chunks + embeddings into document_chunks table.
    Also upsert a row in ingested_files to track what's been processed.
    """
    rows = []
    for chunk, embedding in zip(chunks, embeddings):
        rows.append({
            "filename":    filename,
            "page_num":    chunk["page_num"],
            "chunk_index": chunk["chunk_index"],
            "content":     chunk["text"],
            "embedding":   embedding,   # Supabase accepts Python lists for vector columns
        })
    try:
        supabase_write.table("document_chunks").insert(rows).execute()
        supabase_write.table("ingested_files").upsert({
            "filename":    filename,
            "chunk_count": len(chunks),
        }).execute()
    except httpx.HTTPError as exc:
        raise RuntimeError(
            "Unable to reach Supabase while storing document chunks. "
            "Check SUPABASE_URL, internet access, and DNS resolution."
        ) from exc


def ingest_pdf(pdf_bytes: bytes, filename: str) -> dict:
    """
    Full ingestion pipeline. Call this from Streamlit when a PDF is uploaded.

    Steps:
    1. Extract text from PDF
    2. Split into chunks
    3. Embed all chunks via Azure OpenAI
    4. Store in Supabase

    Returns dict with chunk count for display in UI.
    """
    # Prefer multimodal extraction so tables/graphs/images can be represented in text
    # and embedded for retrieval. This still preserves native text extraction.
    pages = extract_multimodal_content_from_pdf(pdf_bytes)

    # Safety fallback: if multimodal extraction returns nothing, try plain text only.
    if not pages:
        pages = extract_text_from_pdf(pdf_bytes)

    if not pages:
        raise ValueError(f"No text found in {filename}. Is it a scanned PDF?")

    chunks = split_into_chunks(pages)
    embeddings = embed_texts([c["text"] for c in chunks])
    store_chunks_in_supabase(filename, chunks, embeddings)
    return {"chunks_indexed": len(chunks)}


def list_ingested_files() -> list[dict]:
    """Return list of files already ingested. Used in Streamlit sidebar."""
    client = create_client(Config.SUPABASE_URL, Config.SUPABASE_ANON_KEY)
    try:
        return client.table("ingested_files").select("filename, chunk_count, created_at").order("created_at", desc=True).execute().data
    except httpx.HTTPError as exc:
        raise RuntimeError(
            "Unable to reach Supabase to load ingested files. "
            "Check SUPABASE_URL, internet access, and DNS resolution."
        ) from exc