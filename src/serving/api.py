"""FastAPI vector search API."""
from fastapi import FastAPI
from pydantic import BaseModel
import structlog

app = FastAPI(title="Vector Search API")
log = structlog.get_logger()


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    use_hybrid: bool = True


class SearchResult(BaseModel):
    id: int
    text: str
    rank: int


class SearchResponse(BaseModel):
    results: list[SearchResult]
    query: str
    total: int


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    # Retriever injected at startup (see lifespan)
    from src.serving.state import retriever
    results = retriever.search(req.query, top_k=req.top_k)
    log.info("search_request", query=req.query, results=len(results))
    return SearchResponse(
        results=[SearchResult(**r) for r in results],
        query=req.query,
        total=len(results),
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
