from __future__ import annotations
import re
from langchain_community.retrievers import BM25Retriever
from server.rag.vector_store import get_vector_store
from server.rag.document_processor import load_seed_documents

def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z가-힣]+", (text or "").lower()))

def _expand_query(query: str) -> list[str]:
    """Rule-based multi-query expansion for vocabulary-learning retrieval."""
    base = (query or "").strip()
    return [
        base,
        f"{base} meaning definition example usage collocation",
        f"{base} synonym antonym related words context",
    ]

def _lexical_overlap(query: str, text: str) -> float:
    q, d = _tokens(query), _tokens(text)
    return 0.0 if not q else len(q & d) / len(q)

def _rerank(query: str, rows: list[dict], final_k: int) -> list[dict]:
    """Fuse vector/BM25 evidence and rerank with lexical + metadata signals."""
    for row in rows:
        vector_sim = row.get("vector_similarity", 0.0)
        bm25 = row.get("bm25_signal", 0.0)
        lexical = _lexical_overlap(query, row["text"])
        metadata_bonus = 0.08 if row["metadata"].get("word","").lower() in query.lower() else 0.0
        row["rerank_score"] = round(0.55*vector_sim + 0.20*bm25 + 0.20*lexical + metadata_bonus, 6)
    return sorted(rows, key=lambda x: x["rerank_score"], reverse=True)[:final_k]

def retrieve_learning_context(query, learner_level="", top_k=10, final_k=5, score_threshold=1.35):
    store = get_vector_store()
    pool: dict[str, dict] = {}

    # 1. Multi-query dense retrieval
    for expanded in _expand_query(query):
        for d, distance in store.similarity_search_with_score(expanded, k=top_k):
            level = str(d.metadata.get("level", ""))
            if learner_level and level and level != learner_level:
                continue
            # FAISS distance: lower is better. Convert to bounded similarity.
            similarity = 1.0 / (1.0 + max(float(distance), 0.0))
            key = d.page_content
            row = pool.setdefault(key, {
                "text": d.page_content, "metadata": d.metadata,
                "vector_similarity": 0.0, "bm25_signal": 0.0,
                "retrieval_sources": []
            })
            row["vector_similarity"] = max(row["vector_similarity"], similarity)
            row["retrieval_sources"].append("dense")

    # 2. Sparse BM25 retrieval for hybrid search
    docs = load_seed_documents()
    if docs:
        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = min(top_k, len(docs))
        for rank, d in enumerate(bm25.invoke(query), start=1):
            level = str(d.metadata.get("level", ""))
            if learner_level and level and level != learner_level:
                continue
            key = d.page_content
            row = pool.setdefault(key, {
                "text": d.page_content, "metadata": d.metadata,
                "vector_similarity": 0.0, "bm25_signal": 0.0,
                "retrieval_sources": []
            })
            row["bm25_signal"] = max(row["bm25_signal"], 1.0 / rank)
            row["retrieval_sources"].append("bm25")

    candidates = list(pool.values())
    selected = _rerank(query, candidates, final_k)

    # 3. Quality gate + fallback/retrieval retry
    accepted = [x for x in selected if x["vector_similarity"] >= 1/(1+score_threshold) or x["bm25_signal"] > 0]
    if not accepted and query:
        retry = store.similarity_search_with_score(f"English vocabulary {query} example sentence", k=final_k)
        accepted = [{
            "text": d.page_content, "metadata": d.metadata,
            "vector_similarity": 1/(1+max(float(s),0.0)),
            "bm25_signal": 0.0, "rerank_score": 0.0,
            "retrieval_sources": ["dense_retry"]
        } for d,s in retry]

    context = "\n\n---\n\n".join(
        f"[source={x['metadata'].get('source','knowledge')} word={x['metadata'].get('word','')} "
        f"rerank={x.get('rerank_score',0):.4f} via={','.join(sorted(set(x['retrieval_sources'])))}]\n{x['text']}"
        for x in accepted[:final_k]
    )
    return {
        "context": context,
        "matches": accepted[:final_k],
        "strategy": {
            "query_expansion": True,
            "hybrid_search": "FAISS+dense + BM25+sparse",
            "metadata_filter": learner_level or "none",
            "reranker": "weighted fusion",
            "quality_gate_and_retry": True,
            "candidate_count": len(candidates),
            "final_k": final_k,
        }
    }
