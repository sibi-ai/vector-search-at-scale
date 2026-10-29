"""Build a FAISS IVF-PQ index from a corpus of documents."""
import argparse, yaml, os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import structlog

log = structlog.get_logger()


def build_index(cfg: dict) -> faiss.Index:
    model  = SentenceTransformer(cfg["model_name"])
    dim    = model.get_sentence_embedding_dimension()
    nlist  = cfg.get("nlist", 256)
    m      = cfg.get("pq_m", 8)   # sub-quantisers (dim must be divisible by m)
    nbits  = cfg.get("pq_nbits", 8)

    quantiser = faiss.IndexFlatL2(dim)
    index     = faiss.IndexIVFPQ(quantiser, dim, nlist, m, nbits)

    log.info("building_index", dim=dim, nlist=nlist)

    # Encode corpus in batches
    corpus = _load_corpus(cfg["corpus_path"])
    embeddings = []
    batch_size = cfg.get("batch_size", 256)
    for i in tqdm(range(0, len(corpus), batch_size), desc="Encoding"):
        batch = corpus[i:i + batch_size]
        embeddings.append(model.encode(batch, show_progress_bar=False))
    embeddings = np.vstack(embeddings).astype("float32")

    log.info("training_index", n_vectors=len(embeddings))
    index.train(embeddings)
    index.add(embeddings)
    index.nprobe = cfg.get("nprobe", 32)

    out = cfg.get("index_path", "models/faiss.index")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    faiss.write_index(index, out)
    log.info("index_saved", path=out, total_vectors=index.ntotal)
    return index


def _load_corpus(path: str) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/index.yaml")
    with open(parser.parse_args().config) as f:
        cfg = yaml.safe_load(f)
    build_index(cfg)
