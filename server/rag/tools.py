import json
from langchain_core.tools import tool
from server.rag.retriever import retrieve_learning_context

@tool
def search_learning_knowledge(query: str, learner_level: str = "") -> str:
    """Search vocabulary knowledge using hybrid FAISS+BM25 retrieval, reranking and retry.
    Use this when grounded meanings, examples, synonyms, antonyms, collocations or usage
    can improve a recommendation, quiz, grading explanation or remediation plan.
    """
    result = retrieve_learning_context(query, learner_level)
    return json.dumps({
        "context": result["context"],
        "matches": result["matches"],
        "strategy": result["strategy"],
    }, ensure_ascii=False)
