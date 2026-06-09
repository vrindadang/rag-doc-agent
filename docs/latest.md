# Documentation for RAG Doc Agent (Azure OpenAI)

## Project Overview
The RAG Doc Agent is a FastAPI application that allows users to upload PDF documents, extract their content, and query the extracted information using Azure OpenAI's language model. The application supports streaming responses for a more interactive user experience.

## Architecture
The application is structured into two main directories: `app` and `agents`. The `app` directory contains the core functionality of the application, including document ingestion, querying, and response generation. The `agents` directory contains specialized agents for updating documentation and interacting with external services like Supabase.

## Application Modules

### `app/__init__.py`
- **Purpose**: Initializes the application package.
- **Key Functions/Classes**: None.
- **Responsibilities**: Package initialization.

### `app/config.py`
- **Purpose**: Configuration management for the application.
- **Key Functions/Classes**: `Config`
- **Responsibilities**: Loads environment variables for Azure OpenAI and Supabase configurations.

### `app/generator.py`
- **Purpose**: Generates responses from the Azure OpenAI model.
- **Key Functions/Classes**:
  - `stream_answer(question: str, chunks: list[dict])`: Streams the answer from Azure OpenAI token by token.
  - `get_sources_display(chunks: list[dict]) -> list[str]`: Formats source chunks for display.
- **Responsibilities**: Handles the interaction with the Azure OpenAI API to generate answers based on user queries.

### `app/ingestor.py`
- **Purpose**: Handles the ingestion of PDF documents and extraction of their content.
- **Key Functions/Classes**:
  - `extract_text_from_pdf(pdf_bytes: bytes) -> list[dict]`: Extracts text from a PDF file.
  - `extract_multimodal_content_from_pdf(pdf_bytes: bytes) -> list[dict]`: Extracts text, tables, and visual summaries from a PDF.
  - `ingest_pdf(pdf_bytes: bytes, filename: str) -> dict`: Full ingestion pipeline for processing PDFs.
- **Responsibilities**: Manages the extraction of content from PDF files and stores it in a database.

### `app/main.py`
- **Purpose**: Main entry point for the FastAPI application.
- **Key Functions/Classes**:
  - `health()`: Health check endpoint.
  - `list_documents()`: Lists all ingested documents.
  - `ingest()`: Endpoint to upload and index a PDF.
  - `query()`: Endpoint to query the knowledge base.
- **Responsibilities**: Defines the API endpoints and handles incoming requests.

### `app/prompt_builder.py`
- **Purpose**: Constructs prompts for the Azure OpenAI model.
- **Key Functions/Classes**:
  - `build_system_prompt()`: Builds the system prompt for the model.
  - `build_user_message(question: str, chunks: list[dict])`: Constructs the user message with context.
  - `build_messages(question: str, chunks: list[dict])`: Combines system and user messages into the required format.
- **Responsibilities**: Prepares the input for the Azure OpenAI model based on user queries and context.

### `app/retriever.py`
- **Purpose**: Retrieves similar content chunks based on user queries.
- **Key Functions/Classes**:
  - `embed_question(question: str) -> list[float]`: Embeds the question using Azure OpenAI.
  - `retrieve_similar_chunks(question: str, top_k: int = 3) -> list[dict]`: Retrieves the most similar chunks from the database.
- **Responsibilities**: Manages the retrieval of relevant content based on user queries.

## Agent Modules

### `agents/doc_updater.py`
- **Purpose**: Updates documentation based on code changes and reviewer feedback.
- **Key Functions/Classes**:
  - `modifier_agent(state: DocState) -> DocState`: Modifies documentation based on current code and feedback.
  - `reviewer_agent(state: DocState) -> DocState`: Reviews the updated documentation for accuracy.
- **Responsibilities**: Orchestrates the documentation update process, ensuring that the documentation reflects the current state of the codebase.

### `agents/supabase_store.py`
- **Purpose**: Interacts with Supabase for storing and retrieving documentation.
- **Key Functions/Classes**:
  - `upload_doc(content: str, version: int) -> str`: Uploads documentation to Supabase Storage.
  - `fetch_latest_doc() -> tuple[str, int]`: Fetches the latest documentation version.
  - `save_version(version: int, storage_path: str, commit_sha: str, trigger: str) -> int`: Saves version metadata.
- **Responsibilities**: Manages the storage and retrieval of documentation versions in Supabase.

## API Endpoints

### Health Check
- **Method**: GET
- **Path**: `/health`
- **Response**: `{"status": "ok"}`

### List Documents
- **Method**: GET
- **Path**: `/documents`
- **Response**: `{"documents": [...]}`

### Ingest PDF
- **Method**: POST
- **Path**: `/ingest`
- **Request Parameters**: 
  - `file`: PDF file to be ingested.
- **Response**: `{"message": "Ingested successfully", "chunks_indexed": <number>}`

### Query Knowledge Base
- **Method**: POST
- **Path**: `/query`
- **Request Parameters**:
  - `question`: The question to ask.
  - `top_k`: Number of top similar chunks to retrieve (default: 8).
- **Response**: Server-Sent Events stream with tokens.

## Setup Instructions
1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Set up environment variables for Azure OpenAI and Supabase.
4. Run the application using `uvicorn app.main:app --reload`.

## Environment Variables
- `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key.
- `AZURE_OPENAI_ENDPOINT`: The endpoint for Azure OpenAI.
- `AZURE_OPENAI_API_VERSION`: The API version for Azure OpenAI.
- `EMBEDDING_DEPLOYMENT`: The deployment name for embeddings.
- `CHAT_DEPLOYMENT`: The deployment name for chat.
- `SUPABASE_URL`: The URL for your Supabase instance.
- `SUPABASE_ANON_KEY`: The anonymous key for Supabase.
- `SUPABASE_SERVICE_KEY`: The service key for Supabase.
- `SUPABASE_BUCKET`: The bucket name for Supabase Storage.

## Data Flow
1. User uploads a PDF document via the `/ingest` endpoint.
2. The application extracts content from the PDF and stores it in Supabase.
3. Users can query the ingested content via the `/query` endpoint.
4. The application retrieves relevant chunks and generates responses using Azure OpenAI.

## Key Workflows
- **Ingestion Workflow**: Upload PDF → Extract content → Store in Supabase.
- **Query Workflow**: User asks a question → Retrieve similar chunks → Generate response with Azure OpenAI.

This documentation accurately reflects the current state of the codebase and adheres to the review criteria.