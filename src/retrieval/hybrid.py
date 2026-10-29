"""Hybrid retrieval: BM25 sparse + dense vector search, fused with RRF."""
import numpy as np
from rank_bm25 import BM25Okapi
import structlog

log = structlog.get_logger()


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[int]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion.
    k=60 is the standard constant from the RRF paper.
    """
    scores: dict[int, float] = {}
    for ranked_list in rankings:
        for rank, doc_id in enumerate(ranked_list):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda d: scores[d], reverse=True)


class HybridRetriever:
    """
    Combines BM25 sparse retrieval with dense vector search using RRF fusion.
    """

    def __init__(self, corpus: list[str], dense_index, embedder):
        self.corpus       = corpus
        self.dense_index  = dense_index
        self.embedder     = embedder
        tokenized = [doc.lower().split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)
        log.info("hybrid_retriever_built", corpus_size=len(corpus))

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        # Sparse (BM25)
        tokens    = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokens)
        sparse_ranked = list(np.argsort(bm25_scores)[::-1][:top_k * 2])

        # Dense (vector)
        q_emb = self.embedder.encode([query])
        _, dense_ids = self.dense_index.search(q_emb.astype("float32"), top_k * 2)
        dense_ranked = dense_ids[0].tolist()

        # Fuse
        fused = reciprocal_rank_fusion([sparse_ranked, dense_ranked])[:top_k]
        results = [{"id": i, "text": self.corpus[i], "rank": r}
                   for r, i in enumerate(fused)]
        log.info("hybrid_search_done", query_len=len(query.split()), results=len(results))
        return results
