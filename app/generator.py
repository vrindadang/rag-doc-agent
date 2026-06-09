from openai import AzureOpenAI
from app.config import Config
from app.prompt_builder import build_messages

azure_client = AzureOpenAI(
    api_key=Config.AZURE_OPENAI_API_KEY,
    azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
    api_version=Config.AZURE_OPENAI_API_VERSION,
)

# this function is used to stream the answer from Azure OpenAI token by token, which allows for a more responsive UI in Streamlit. It builds the messages with context and then yields each token as it arrives from the stream.
def stream_answer(question: str, chunks: list[dict]):
    """
    Generator that streams the LLM answer token by token.

    How it works:
    1. Build the messages (system + user with context)
    2. Call Azure OpenAI with stream=True
    3. Yield each token as it arrives
    The caller (Streamlit) loops over this generator and updates the UI.
    """
    messages = build_messages(question, chunks)

    stream = azure_client.chat.completions.create(
        model=Config.CHAT_DEPLOYMENT,   # your deployment name
        messages=messages,
        stream=True,
        temperature=0.2,
        max_tokens=1000,
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def get_sources_display(chunks: list[dict]) -> list[str]:
    """Format source chunks for display in Streamlit."""
    return [
        f"Excerpt {i}: {c['filename']} — page {c['page_num']} (similarity: {c.get('similarity', 0):.2f})"
        for i, c in enumerate(chunks, start=1)
    ]