# clients.py - creates the clients ONCE and shares them (like a singleton service with providedIn: 'root').
# LLM + embeddings come from LANGCHAIN: AzureChatOpenAI/AzureOpenAIEmbeddings, or ChatOllama/OllamaEmbeddings in local mode.
# Because every LangChain chat model has the SAME interface, the rest of the code never knows which one it got.
# Document Intelligence + Cosmos use the Azure SDKs directly (only in azure mode).

from functools import lru_cache  # @lru_cache = "run once per set of arguments, then return the cached result".

from azure.ai.documentintelligence import DocumentIntelligenceClient  # SDK for Azure AI Document Intelligence.
from azure.core.credentials import AzureKeyCredential  # Wraps an API key so Azure SDKs can use it.
from azure.cosmos import ContainerProxy, CosmosClient, PartitionKey  # SDK for Azure Cosmos DB (NoSQL API).
from langchain_core.embeddings import Embeddings  # Base type of every LangChain embeddings class.
from langchain_core.language_models import BaseChatModel  # Base type of every LangChain chat model.
from langchain_ollama import ChatOllama, OllamaEmbeddings  # LangChain wrappers around local Ollama models.
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings  # LangChain wrappers around Azure OpenAI.

from backend import config  # Our settings.

# How with_structured_output() forces a schema. Azure: tool calling. Ollama: "json_schema" = Ollama's native structured
# output, which constrains the small model's output to the schema (more reliable than tool calling for a 3B model).
STRUCTURED_OUTPUT_METHOD = "json_schema" if config.USE_LOCAL else "function_calling"

# How Cosmos should store and index the vector field. Created together with the container (cannot be changed later!).
VECTOR_EMBEDDING_POLICY = {
    "vectorEmbeddings": [  # A container can have several vector fields; we have one.
        {
            "path": "/embedding",  # The JSON field that holds the vector in every item.
            "dataType": "float32",  # Each number is a 32-bit float.
            "distanceFunction": "cosine",  # How "closeness" is measured. Cosine = compare the DIRECTION of vectors (= meaning).
            "dimensions": config.EMBEDDING_DIMENSIONS,  # 1536 for text-embedding-3-small. Must match the model exactly.
        }
    ]
}

INDEXING_POLICY = {
    "indexingMode": "consistent",  # The index is updated in the same write as the item (queries see new items immediately).
    "includedPaths": [{"path": "/*"}],  # Index every field -> fast WHERE filters on metadata (the SQL branch of the diagram).
    "excludedPaths": [  # ...except these:
        {"path": '/"_etag"/?'},  # A system field we never filter on.
        {"path": "/embedding/*"},  # The vector must NOT be in the normal index - it gets its own vector index below.
    ],
    "vectorIndexes": [  # A special index that makes vector search fast.
        {"path": "/embedding", "type": "diskANN"},  # DiskANN = Microsoft's approximate-nearest-neighbour index, scales to millions.
    ],  # ("flat" is exact but only allows up to 505 dimensions, so it can't hold 1536.)
}


@lru_cache  # One chat model per (temperature, local_model) combination.
def get_chat_model(temperature: float = 0.0, local_model: str | None = None) -> BaseChatModel:  # Something you can .invoke() / .stream().
    if config.USE_LOCAL:  # Free local mode.
        return ChatOllama(
            model=local_model or config.OLLAMA_CHAT_MODEL,  # A specific model if asked (e.g. the guard model), else qwen2.5:3b.
            base_url=config.OLLAMA_BASE_URL,  # http://localhost:11434
            temperature=temperature,  # Same meaning as on Azure.
            num_ctx=8192,  # Context window in tokens. Ollama's default is small and would silently CUT our RAG prompt.
        )
    return AzureChatOpenAI(  # Azure mode.
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,  # Your resource URL.
        api_key=config.AZURE_OPENAI_API_KEY,  # Production tip: use Managed Identity (azure_ad_token_provider) instead of keys.
        api_version=config.AZURE_OPENAI_API_VERSION,  # Azure needs an explicit API version.
        azure_deployment=config.CHAT_DEPLOYMENT,  # Your DEPLOYMENT name (e.g. gpt-4o-mini).
        temperature=temperature,  # 0 = predictable (routing, guardrails), a bit higher = more natural wording.
    )


@lru_cache  # One embeddings client.
def get_embeddings() -> Embeddings:  # LangChain "Embeddings" interface: .embed_documents([...]) and .embed_query("...").
    if config.USE_LOCAL:  # Free local mode.
        return OllamaEmbeddings(model=config.OLLAMA_EMBED_MODEL, base_url=config.OLLAMA_BASE_URL)  # nomic-embed-text.
    return AzureOpenAIEmbeddings(  # Azure mode.
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,  # Same Azure OpenAI resource.
        api_key=config.AZURE_OPENAI_API_KEY,  # Same key.
        api_version=config.AZURE_OPENAI_API_VERSION,  # Same API version.
        azure_deployment=config.EMBEDDING_DEPLOYMENT,  # Embedding deployment (e.g. text-embedding-3-small).
    )


@lru_cache  # One Document Intelligence client for the whole process (azure mode only).
def get_docintel() -> DocumentIntelligenceClient:
    return DocumentIntelligenceClient(  # Create the client.
        endpoint=config.DOCINTEL_ENDPOINT,  # Resource URL.
        credential=AzureKeyCredential(config.DOCINTEL_KEY),  # Key-based auth.
    )


@lru_cache  # Connect + create database/container only once (azure mode only).
def get_container() -> ContainerProxy:  # ContainerProxy = the object you query/insert with.
    client = CosmosClient(config.COSMOS_ENDPOINT, credential=config.COSMOS_KEY)  # Connect to the Cosmos account.
    database = client.create_database_if_not_exists(id=config.COSMOS_DATABASE)  # Like CREATE DATABASE IF NOT EXISTS.
    return database.create_container_if_not_exists(  # Like CREATE TABLE IF NOT EXISTS, plus the vector settings.
        id=config.COSMOS_CONTAINER,  # Container name.
        partition_key=PartitionKey(path="/documentId"),  # Items of the same document live together -> cheap "delete all chunks of a document".
        indexing_policy=INDEXING_POLICY,  # Normal index + vector index (above).
        vector_embedding_policy=VECTOR_EMBEDDING_POLICY,  # Tells Cosmos /embedding is a 1536-dim cosine vector.
    )
