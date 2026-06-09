# app/retriever.py
import httpx
from openai import AzureOpenAI
from supabase import create_client
from app.config import Config

azure_client = AzureOpenAI(
    api_key=Config.AZURE_OPENAI_API_KEY,
    azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
    api_version=Config.AZURE_OPENAI_API_VERSION,
)
supabase_read = create_client(Config.SUPABASE_URL, Config.SUPABASE_ANON_KEY)


def embed_question(question: str) -> list[float]:
    """
    Embed the question using Azure OpenAI embeddings endpoint.
    """
    response = azure_client.embeddings.create(
        model=Config.EMBEDDING_DEPLOYMENT,
        input=[question],
    )
    return response.data[0].embedding

# this function is used to retrieve similar chunks from Supabase based on the question embedding.
def retrieve_similar_chunks(question: str, top_k: int = 3) -> list[dict]:
    """
    Find the top_k most similar chunks to the question.

    Steps:
    1. Embed the question
    2. Call our match_chunks SQL function via supabase.rpc()
    3. Return the matching chunks with similarity scores

    Returns:
    [{"filename": "doc.pdf", "page_num": 3, "content": "...", "similarity": 0.87}, ...]
    """
    question_embedding = embed_question(question)

    try:
        response = supabase_read.rpc(
            "match_chunks",
            {
                "query_embedding": question_embedding,
                "match_count":     top_k,
            }
        ).execute()
    except httpx.HTTPError as exc:
        raise RuntimeError(
            "Unable to reach Supabase to retrieve matching chunks. "
            "Check SUPABASE_URL, internet access, and DNS resolution."
        ) from exc

    return response.data
