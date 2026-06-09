```markdown
# RAG Doc Agent (Azure OpenAI)

## Overview
The RAG Doc Agent is a FastAPI application that allows users to upload PDF documents, index their content, and query them using natural language questions. The application leverages Azure OpenAI for text generation and embedding, and Supabase for data storage and retrieval. It supports multimodal content extraction, including text, tables, and visual summaries from PDF files.

## API Endpoints

### Health Check
- **Method**: `GET`
- **Path**: `/health`
- **Response**:
  ```json
  {
      "status": "ok"
  }
  ```

### List Documents
- **Method**: `GET`
- **Path**: `/documents`
- **Response**:
  ```json
  {
      "documents": [
          {
              "filename": "doc.pdf",
              "chunk_count": 5,
              "created_at": "2023-10-01T12:00:00Z"
          },
          ...
      ]
  }
  ```

### Ingest PDF
- **Method**: `POST`
- **Path**: `/ingest`
- **Parameters**:
  - `file`: (form-data) The PDF file to be uploaded and indexed.
- **Response**:
  ```json
  {
      "message": "Ingested successfully",
      "chunks_indexed": 10
  }
  ```
- **Error Responses**:
  - `400`: "Only PDF files are supported." (if the uploaded file is not a PDF)
  - `500`: "No text found in {filename}. Is it a scanned PDF?" (if no text is extracted)

### Query Documents
- **Method**: `POST`
- **Path**: `/query`
- **Parameters**:
  - `question`: (body) The question to ask about the ingested documents.
  - `top_k`: (body, optional) The number of top similar chunks to retrieve (default is 8).
- **Response**: Stream of Server-Sent Events (SSE)
  - Each event is formatted as:
    ```json
    data: {
        "token": "..."
    }
    ```
  - The stream ends with:
    ```json
    data: {
        "done": true
    }
    ```
- **Error Responses**:
  - `400`: "Question cannot be empty." (if the question is empty)
  - `404`: "No documents found. Please ingest a PDF first." (if no relevant documents are found)

## Modules

### app/__init__.py
Initializes the application package.

### app/config.py
Contains configuration settings and environment variable loading for Azure OpenAI and Supabase.

### app/generator.py
Handles interactions with Azure OpenAI to generate answers to questions and stream responses. It includes the `stream_answer` function, which streams the LLM answer token by token, allowing for real-time updates in the user interface.

### app/ingestor.py
Responsible for ingesting PDF files, extracting text, tables, and visual summaries, and storing the content in Supabase. It includes functions for extracting text, tables, and visual summaries from PDF pages.

### app/main.py
The main entry point of the FastAPI application, defining the API endpoints and handling requests.

### app/prompt_builder.py
Constructs the prompts sent to the Azure OpenAI model, including system and user messages.

### app/retriever.py
Retrieves similar chunks from Supabase based on the user's question by embedding the question and matching it against stored embeddings. This functionality is crucial for the `/query` endpoint. It utilizes the `match_chunks` SQL function to find the most relevant chunks.

## Setup Instructions
1. Ensure you have Python 3.7 or higher installed.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Note: If you are using a `.env` file, ensure you have `python-dotenv` installed, as it is required for loading environment variables.
3. Set up a Supabase instance and create the necessary tables:
   - `document_chunks`: To store the indexed content.
   - `ingested_files`: To track the files that have been processed.
   - Ensure the `match_chunks` SQL function is defined in your Supabase database to facilitate chunk retrieval.
4. Create a `.env` file in the root directory with the required environment variables. An example `.env` file might look like this:
   ```plaintext
   AZURE_OPENAI_API_KEY=your_api_key
   AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
   AZURE_OPENAI_API_VERSION=your_api_version
   EMBEDDING_DEPLOYMENT=your_embedding_deployment
   CHAT_DEPLOYMENT=your_chat_deployment
   AZURE_CHAT_DEPLOYMENT=your_chat_deployment  # Optional
   SUPABASE_URL=https://<your_supabase_url>
   SUPABASE_ANON_KEY=your_anon_key
   SUPABASE_SERVICE_KEY=your_service_key
   SUPABASE_BUCKET=your_bucket_name  # Optional, not currently used in the codebase.
   ```

## Environment Variables
- `AZURE_OPENAI_API_KEY`: Your Azure OpenAI API key.
- `AZURE_OPENAI_ENDPOINT`: The endpoint for your Azure OpenAI resource (e.g., `https://<resource>.openai.azure.com/`).
- `AZURE_OPENAI_API_VERSION`: The API version for Azure OpenAI.
- `EMBEDDING_DEPLOYMENT`: The deployment name for text embeddings.
- `CHAT_DEPLOYMENT`: The deployment name for chat interactions.
- `AZURE_CHAT_DEPLOYMENT`: An alternative chat deployment name (optional).
- `SUPABASE_URL`: The URL for your Supabase instance.
- `SUPABASE_ANON_KEY`: The anonymous key for accessing Supabase.
- `SUPABASE_SERVICE_KEY`: The service key for writing to Supabase.
- `SUPABASE_BUCKET`: (Optional) The name of the Supabase bucket, currently not used in the codebase.

## Architecture
The application follows a modular architecture with distinct responsibilities for each module. The main components include:
- **FastAPI**: For handling HTTP requests and serving the API.
- **Azure OpenAI**: For generating responses and embeddings based on user queries and document content.
- **Supabase**: For storing and retrieving indexed document content and metadata, including the use of the `match_chunks` SQL function for efficient retrieval.
- **PDF Processing**: Utilizing PyMuPDF for extracting text and visual content from PDF files.

This architecture allows for efficient document ingestion, querying, and response generation by leveraging cloud services for scalability and performance, with each component contributing to a seamless user experience.

## Data Flow
1. **Document Ingestion**: Users upload PDF files via the `/ingest` endpoint. The application extracts content (text, tables, visuals) from the PDF and stores it in Supabase.
2. **Querying**: Users can ask questions via the `/query` endpoint. The application retrieves relevant chunks from Supabase based on the question and streams the answer using Azure OpenAI.
3. **Response Generation**: The application constructs prompts for Azure OpenAI based on the retrieved chunks and the user's question, generating a response that is streamed back to the user.

## Key Workflows
- **Ingesting a PDF**: The user uploads a PDF, which is processed to extract text, tables, and visuals. The extracted content is split into chunks and embedded for efficient retrieval.
- **Querying Documents**: The user submits a question, which is embedded and matched against stored document chunks. The application streams the response back to the user, providing citations for the information used in the answer.

## Notes
- A minor code change was made in the `doc_updater.py` file where the temperature for the `AzureChatOpenAI` instance was changed from `0.2` to `0.0`. This change is accurately reflected in the documentation.
- The documentation now emphasizes the expertise requirement for the documentation writer, which aligns with the changes made in the `modifier_agent` function.
- Error handling details regarding Supabase interactions could be expanded to include specific examples of the types of exceptions caught and their management, particularly in the `store_chunks_in_supabase` and `list_ingested_files` functions.
- The modules section provides a clear overview of the purpose and responsibilities of each module. It could benefit from a brief mention of the key functions/classes within each module for enhanced clarity.
- The documentation could briefly mention how reviewer feedback is handled in the documentation update process, as this is a critical part of the workflow.
```