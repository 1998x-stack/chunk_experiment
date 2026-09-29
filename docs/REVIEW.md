# Deep review: findings and redesign

## P0 / correctness

### 1. "Cosine" was a raw dot product

The legacy `EmbeddingModel.similarity()` returned `np.dot(a, b)` without normalization. Thresholds in the range 0–1 are only interpretable as cosine thresholds when vectors are normalized.

**Fix:** legacy path now computes actual cosine; v2 normalizes provider output and validates vector shapes.

### 2. Whitespace counting breaks Chinese chunk budgets

`len(text.split())` can report an entire Chinese paragraph as one token.

**Fix:** v2 exposes a `LengthMetric` protocol and ships a deterministic CJK-aware approximation. Production experiments can inject the tokenizer of the retrieval/generation model.

### 3. Adaptive threshold leaked across documents

The legacy semantic chunker assigned a computed document threshold back to `self.similarity_threshold`. Later documents could therefore reuse the first document's threshold.

**Fix:** legacy compatibility code restores configured state after each call; v2 never mutates configured thresholds.

### 4. Mock embeddings did not benchmark semantic quality

The original ablation runner returned synthetic repeated vectors, so "semantic" algorithm comparisons could not measure semantic coherence.

**Fix:** new ablation runs use a deterministic lexical-hash baseline and are explicitly labelled. Real semantic claims require a real documented embedding backend.

## P1 / reproducibility and safety

### 5. Enhanced default used random vectors

Running the same document twice could produce different chunks.

**Fix:** default fallback is deterministic lexical hashing.

### 6. Embedding HTTP client contained a hard-coded Cookie

Credentials/session material should never live in source code, and requests had no timeout.

**Fix:** headers are injectable, no Cookie is embedded, and requests use an explicit timeout.

### 7. Embedding response length could silently truncate sentences

Legacy code zipped sentences and embeddings. Too few vectors silently dropped trailing sentences.

**Fix:** response cardinality is validated before chunk construction.

## P1 / architecture

### 8. Experiments, notebooks, library code and outputs shared one flat namespace

This makes it difficult to know which APIs are stable and which files are research artifacts.

**Fix:** the new importable package lives under `chunk_experiment/`; experiments live under `experiments/`; architecture and review notes live under `docs/`; historical files remain untouched where possible.

### 9. No shared chunk contract

Algorithms returned different object shapes and made coverage difficult to audit.

**Fix:** v2 returns one immutable `Chunk(text, start, end, metadata)` model with exact source offsets.

### 10. Evaluation focused on chunk count / average size / wall time

Those are useful operational metrics but insufficient for RAG quality.

**Fix:** v2 adds source coverage, duplicated span ratio, size compliance and dispersion. The next benchmark layer should add retrieval labels and Recall@K / MRR / nDCG.

## Migration strategy

1. Keep notebooks and historical JSON for traceability.
2. Use `chunk_experiment.RecursiveChunker` and `chunk_experiment.SemanticChunker` for new work.
3. Use `semantic-hash` only for deterministic CI/plumbing tests.
4. Use `semantic-http` (or implement another `EmbeddingProvider`) for semantic experiments.
5. Record dataset hashes, backend/model identity and parameters with every benchmark.
