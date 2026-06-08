import os
from dotenv import load_dotenv 
load_dotenv(override=True)


class Config:
    # Azure OpenAI credentials
    AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT")          # e.g. https://<resource>.openai.azure.com/
    AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION")

    # Deployment names (set in Azure Portal when you deploy a model)
    EMBEDDING_DEPLOYMENT: str = os.getenv("EMBEDDING_DEPLOYMENT")
    CHAT_DEPLOYMENT: str = os.getenv("CHAT_DEPLOYMENT")
    AZURE_CHAT_DEPLOYMENT: str = os.getenv("AZURE_CHAT_DEPLOYMENT")

    # Supabase
    SUPABASE_URL: str = os.getenv("SUPABASE_URL")
    SUPABASE_ANON_KEY: str = os.getenv("SUPABASE_ANON_KEY")
    SUPABASE_SERVICE_KEY: str = os.getenv("SUPABASE_SERVICE_KEY")
    SUPABASE_BUCKET: str = os.getenv("SUPABASE_BUCKET")
