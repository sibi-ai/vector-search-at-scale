# Vector Search at Scale

High-performance vector similarity search at billion-scale.
FAISS, Pinecone, approximate nearest neighbour (ANN) indexing,
and hybrid BM25 + dense retrieval with re-ranking.

## Features

- FAISS IVF-PQ indexing for billion-scale approximate nearest neighbour search
- Pinecone managed vector store integration
- Hybrid retrieval: BM25 sparse + dense embedding, fused with RRF
- Re-ranking with cross-encoder models
- Async FastAPI serving with batched inference
- Benchmarking suite: recall@k, latency, throughput

## Architecture

```
Query
  ↓
Embedding (dense)     BM25 (sparse)
  ↓                        ↓
FAISS / Pinecone      Elasticsearch
  ↓                        ↓
        RRF Fusion
            ↓
       Cross-encoder re-ranking
            ↓
       Top-k results
```

## Stack

| Component | Technology |
|-----------|-----------|
| ANN index | FAISS (IVF-PQ), Pinecone |
| Sparse | BM25 (rank-bm25) |
| Embeddings | sentence-transformers |
| Re-ranking | cross-encoder/ms-marco |
| Serving | FastAPI |
| Benchmarking | Custom recall@k suite |

## Project Structure

```
├── src/
│   ├── indexing/       # FAISS and Pinecone index builders
│   ├── retrieval/      # Dense, sparse, hybrid retrieval
│   ├── reranking/      # Cross-encoder re-ranking
│   ├── serving/        # FastAPI search API
│   └── benchmark/      # Recall@k, latency, throughput tests
├── configs/
├── tests/
├── notebooks/
├── Dockerfile
└── requirements.txt
```

## Quickstart

```bash
git clone https://github.com/sibi-ai/vector-search-at-scale
cd vector-search-at-scale
pip install -r requirements.txt
python src/indexing/build_faiss.py --config configs/index.yaml
uvicorn src.serving.api:app --reload
```
