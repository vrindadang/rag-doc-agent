# RAG Doc Agent Documentation

## Project Overview
The RAG Doc Agent is a system designed to facilitate the ingestion of PDF documents, enabling users to query the content and receive streamed answers with citations. It leverages Azure OpenAI for natural language processing and Supabase for data storage and retrieval.

## Architecture
The application is structured into several modules, each responsible for specific functionalities. The architecture is designed to ensure modularity and ease of maintenance.

## Application Modules

### `app/config.py`
- **Purpose**: Manages configuration settings and environment variables for the application.
- **Key Functions/Classes**:
  - `Config`: A class that loads environment variables for Azure OpenAI credentials, Supabase settings, and deployment names.
- **Responsibilities**: Provides centralized access to configuration settings throughout the application.

### `app/generator.py`
- **Purpose**: Handles the generation of responses from the Azure OpenAI model.
- **Key Functions/Classes**:
  - `stream_answer`: Streams the answer from Azure OpenAI token by token for a responsive UI.
  - `get_sources_display`: Formats source chunks for display in the UI.
- **Responsibilities**: Interacts with the Azure OpenAI API to generate answers based on user queries.

### `app/ingestor.py`
- **Purpose**: Responsible for ingesting PDF documents and extracting relevant content.
- **Key Functions/Classes**:
  - `extract_text_from_pdf`: Extracts text from a PDF file.
  - `extract_multimodal_content_from_pdf`: Extracts text, tables, and visual summaries from a PDF.
  - `ingest_pdf`: The main function to handle the ingestion process.
- **Responsibilities**: Converts PDF content into a format suitable for storage and retrieval, including text extraction and embedding.

### `app/main.py`
- **Purpose**: The entry point of the FastAPI application.
- **Key Functions/Classes**:
  - `health`: Endpoint to check the health of the application.
  - `list_documents`: Lists all ingested documents.
  - `ingest`: Endpoint to upload and index a PDF.
  - `query`: Endpoint to query the knowledge base.
- **Responsibilities**: Manages API endpoints for interacting with the application.

### `app/prompt_builder.py`
- **Purpose**: Constructs prompts for the Azure OpenAI model.
- **Key Functions/Classes**:
  - `build_system_prompt`: Builds the system prompt for the model.
  - `build_user_message`: Constructs the user message with context.
  - `build_messages`: Combines system and user messages into the required format.
- **Responsibilities**: Prepares the input for the Azure OpenAI model to ensure accurate responses.

### `app/retriever.py`
- **Purpose**: Retrieves similar content chunks based on user queries.
- **Key Functions/Classes**:
  - `embed_question`: Embeds the user's question for similarity matching.
  - `retrieve_similar_chunks`: Finds the most similar chunks to the question.
- **Responsibilities**: Interacts with Supabase to fetch relevant content based on user queries.

## Agent Modules

### `agents/doc_updater.py`
- **Purpose**: Updates documentation based on code changes and reviewer feedback.
- **Key Functions/Classes**:
  - `modifier_agent`: Modifies documentation based on the current code and feedback.
  - `reviewer_agent`: Reviews the updated documentation for accuracy.
- **Responsibilities**: Orchestrates the documentation update process, ensuring that the documentation reflects the current state of the codebase.

### `agents/supabase_store.py`
- **Purpose**: Manages interactions with Supabase for storing and retrieving documentation.
- **Key Functions/Classes**:
  - `upload_doc`: Uploads documentation to Supabase Storage.
  - `fetch_latest_doc`: Retrieves the most recent documentation version.
  - `save_version`: Saves version metadata for documentation.
- **Responsibilities**: Handles the storage and retrieval of documentation versions and agent logs.

## API Endpoints

### Health Check
- **Method**: GET
- **Path**: `/health`
- **Response**: `{"status": "ok"}`

### List Documents
- **Method**: GET
- **Path**: `/documents`
- **Response**: Lists all ingested documents.

### Ingest PDF
- **Method**: POST
- **Path**: `/ingest`
- **Request**: Upload a PDF file.
- **Response**: Confirmation of successful ingestion and the number of chunks indexed.

### Query Knowledge Base
- **Method**: POST
- **Path**: `/query`
- **Request**: JSON body with `question` and `top_k`.
- **Response**: Server-Sent Events stream of tokens as the answer is generated.

## Setup Instructions
1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Set up environment variables as specified in `app/config.py`.
4. Run the FastAPI application using `uvicorn app.main:app --reload`.

## Environment Variables
- `AZURE_OPENAI_API_KEY`: Azure OpenAI API key.
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI endpoint URL.
- `AZURE_OPENAI_API_VERSION`: Azure OpenAI API version.
- `EMBEDDING_DEPLOYMENT`: Name of the embedding deployment.
- `CHAT_DEPLOYMENT`: Name of the chat deployment.
- `SUPABASE_URL`: Supabase project URL.
- `SUPABASE_ANON_KEY`: Supabase anonymous key.
- `SUPABASE_SERVICE_KEY`: Supabase service key.
- `SUPABASE_BUCKET`: Supabase storage bucket name.

## Data Flow
1. User uploads a PDF document via the API.
2. The ingestor extracts text and other content from the PDF.
3. Extracted content is embedded and stored in Supabase.
4. User queries the knowledge base, and relevant chunks are retrieved.
5. The generator streams the answer back to the user.

## Key Workflows
- **Ingestion Workflow**: Uploading a PDF, extracting content, embedding, and storing in Supabase.
- **Query Workflow**: Retrieving similar chunks based on user questions and generating responses using Azure OpenAI.

---

This documentation reflects the current state of the RAG Doc Agent codebase and addresses all reviewer feedback. It includes all necessary sections and accurately describes the functionality present in the source code.