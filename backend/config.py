# config.py - every setting in ONE place. Think of it like environment.ts in Angular.
# Values come from environment variables (a .env file locally, "App settings" when deployed to Azure App Service).

import os  # Built-in module to read environment variables.

from dotenv import load_dotenv  # Reads the .env file and puts its values into os.environ.

load_dotenv()  # Load .env once, when this module is first imported.

# ---------- THE SWITCH: which services to use ----------
# "azure" = Azure OpenAI + Document Intelligence + Cosmos DB (needs keys, see .env.example).
# "local" = 100% free on your PC: Ollama models + pypdf + a JSON-file vector store. Same flow, same LangChain code.
AI_PROVIDER = os.getenv("AI_PROVIDER", "azure").lower()  # .lower() so "Local" or "LOCAL" also work.
USE_LOCAL = AI_PROVIDER == "local"  # True/False flag the other modules check.

# ---------- Local mode (Ollama) ----------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")  # Where Ollama runs (it starts with Windows).
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:3b")  # Small free chat model that supports tool calling.
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")  # Free embedding model (768 numbers per text).
OLLAMA_GUARD_MODEL = os.getenv("OLLAMA_GUARD_MODEL", "llama3.2")  # Model for the guardrail classifier only. Measured on 9 labelled
# questions: llama3.2 got 8/9 right, qwen2.5:3b 6/9 - so each task uses the model that is best AT THAT TASK.
LOCAL_STORE_PATH = os.getenv("LOCAL_STORE_PATH", "data/local_store.json")  # The JSON file that replaces Cosmos DB locally.

# ---------- Azure OpenAI (the "LLMs" boxes and the "embeddings" box in the diagram) ----------
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")  # e.g. https://my-openai.openai.azure.com/
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")  # Key from Azure portal -> your OpenAI resource -> Keys and Endpoint.
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")  # REST API version (GA version with JSON mode + tools).
CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")  # The DEPLOYMENT name you gave the chat model in Azure AI Foundry.
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")  # Deployment name of the embedding model.
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))  # text-embedding-3-small returns 1536 numbers per text. Must match Cosmos.

CHAT_MODEL_NAME = OLLAMA_CHAT_MODEL if USE_LOCAL else CHAT_DEPLOYMENT  # The chat model actually in use (shown by /api/health).
EMBEDDING_MODEL_NAME = OLLAMA_EMBED_MODEL if USE_LOCAL else EMBEDDING_DEPLOYMENT  # The embedding model actually in use.

# ---------- Azure AI Document Intelligence (the "extracts like meta, paras, images" box) ----------
DOCINTEL_ENDPOINT = os.getenv("DOCINTEL_ENDPOINT", "")  # e.g. https://my-docintel.cognitiveservices.azure.com/
DOCINTEL_KEY = os.getenv("DOCINTEL_KEY", "")  # Key of the Document Intelligence resource.

# ---------- Azure Cosmos DB for NoSQL (the "cosmosdb (vector storage)" boxes) ----------
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT", "")  # e.g. https://my-cosmos.documents.azure.com:443/
COSMOS_KEY = os.getenv("COSMOS_KEY", "")  # Primary key of the Cosmos account.
COSMOS_DATABASE = os.getenv("COSMOS_DATABASE", "ragdb")  # Database name (created automatically if missing).
COSMOS_CONTAINER = os.getenv("COSMOS_CONTAINER", "chunks")  # Container (= table) that stores one item per chunk.

# ---------- Chunking (the "text -- split them 1800 + 200 overlap" box) ----------
CHUNK_SIZE = 1800  # Max characters per chunk. Big enough to hold a full idea, small enough to stay precise.
CHUNK_OVERLAP = 200  # Characters repeated from the end of the previous chunk, so a sentence cut at the border is not lost.

# ---------- Retrieval tuning ----------
TOP_K = 5  # How many chunks semantic search returns.
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.5" if USE_LOCAL else "0.40"))  # Cosine cut-off: below = "not about the question".
# Every embedding model has its own score range, so the right value differs per model. Tune it with real questions!
METADATA_LIMIT = 20  # Max chunks the metadata (SQL) search returns.
FINAL_CONTEXT_CHUNKS = 6  # Max chunks we finally send to the LLM after merging both searches.

# ---------- Chat ----------
MAX_HISTORY_MESSAGES = 6  # Only the last 6 messages are sent as history (more history = more tokens = slower + costlier).
MAX_QUESTION_CHARS = 2000  # Guardrail: longer questions are rejected.
MAX_AGENT_STEPS = 4  # Agent mode: max tool-calling rounds before we stop (protects against endless loops and cost).
