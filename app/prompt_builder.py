def build_system_prompt() -> str:
    return """You are a helpful document assistant.
Answer questions based ONLY on the context provided in the user message.

Follow these rules strictly:
1. Only use information from the provided context. Do not use your own knowledge or make up random information.
2. If the user requests a table, return the answer as a markdown table whenever the context contains information that can be organized into rows and columns.
3. If the user requests bullet points, numbered lists, or another format, follow that format.
4. If no format is specified, answer in 4-5 sentences.
5. At the end of the response, list which sources you used in this format: Sources used: [filename, page X], [filename, page Y]
6. If the question cannot be answered from the context, say: "I'm sorry, I could not find this information in the provided context." Do not try to guess or use outside knowledge.

Do not make up information. Do not guess. Only state what the context says."""


def build_user_message(question: str, chunks: list[dict]) -> str:
    """
    The user message contains:
    - The retrieved chunks (injected as context)
    - The actual question

    Each chunk is labeled with filename and page number so the LLM can cite them.
    Context comes BEFORE the question — LLMs pay more attention to content at the end.
    """
    formatted_excerpts = []
    for i, chunk in enumerate(chunks, start=1):
        excerpt = f"""--- Excerpt {i} ---
File: {chunk['filename']}
Page: {chunk['page_num']}

{chunk['content']}"""
        formatted_excerpts.append(excerpt)

    context = "\n\n".join(formatted_excerpts)

    return f"""Here are the relevant excerpts from the context:

{context}

---

Based ONLY on the context above, please answer this question:
{question}"""


def build_messages(question: str, chunks: list[dict]) -> list[dict]:
    """
    Combine system prompt + user message into the messages list
    that the chat API expects.

    Format:
    [
        {"role": "system", "content": "..."},
        {"role": "user",   "content": "..."}
    ]
    """
    return [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user",   "content": build_user_message(question, chunks)},
    ]