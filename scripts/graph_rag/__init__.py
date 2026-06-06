"""GraphRAG-style retrieval for DecompVerify.

Modules:
    bm25_index  — tantivy-based BM25 over the HOVER (HotpotQA-style) wiki corpus.
    doc_store   — load wiki docs by title and split into sentences.
    encoder     — pooled sentence embeddings (MiniLM via transformers).
    graph_build — build a per-claim heterogeneous S/N/D graph with HNSW S-S edges.
    ppr        — personalized PageRank over the heterogeneous graph.
    retriever   — orchestrate per-fact retrieval with dependency-aware seeds.
    assembly    — fact-bundle / coverage-driven evidence assembly.
"""
