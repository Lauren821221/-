import os
from threading import Lock

from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings

from server.rag.document_processor import load_seed_documents

_LOCK = Lock()
_STORE = None


def _required(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"{name} 환경변수가 필요합니다.")
    return value


def embeddings():
    return AzureOpenAIEmbeddings(
        azure_endpoint=_required("AOAI_ENDPOINT"),
        api_key=_required("AOAI_API_KEY"),
        azure_deployment=os.getenv("AOAI_DEPLOY_EMBED_3_LARGE", "text-embedding-3-large"),
        api_version=os.getenv("AOAI_API_VERSION", "2024-10-21"),
    )


def get_vector_store():
    global _STORE
    with _LOCK:
        if _STORE is None:
            _STORE = FAISS.from_documents(load_seed_documents(), embeddings())
        return _STORE


def index_documents(documents):
    if documents:
        get_vector_store().add_documents(documents)
    return len(documents)
