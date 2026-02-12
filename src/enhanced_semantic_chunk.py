# -*- coding: utf-8 -*-

import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/" + ".."))

import numpy as np
import warnings
from typing import List, Optional, Tuple, Dict, Callable
from util.sentence_split import GeneralTextSplitter
import logging


class EnhancedEmbeddingModel:
    """Enhanced embedding model class supporting multiple embedding strategies."""

    def __init__(self, embedding_client=None, embedding_dim: int = 128, strategy: str = "mean_pooling"):
        """Initialize enhanced embedding model.
        
        Args:
            embedding_client: Client for real embedding services
            embedding_dim: Dimension of embedding vectors
            strategy: Strategy for combining sentence embeddings ('mean_pooling', 'weighted_mean', 'attention')
        """
        self.embedding_client = embedding_client
        self.embedding_dim = embedding_dim
        self.strategy = strategy

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a batch of texts and return a list of embedding vectors."""
        if self.embedding_client:
            # Use real embedding service
            batch_size = 10
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                response = self.embedding_client.get_embeddings(batch_texts)
                batch_embeddings = response.get("data", {}).get("resultList", [])
                embeddings.extend(batch_embeddings)
            return [np.array(embedding).astype(np.float32) for embedding in embeddings]
        else:
            # Use simulated embeddings
            return [np.random.rand(self.embedding_dim).astype(np.float32) for _ in texts]

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))


class EnhancedSemanticSentence:
    """Enhanced sentence class with additional metadata."""

    def __init__(
        self,
        text: str,
        start_index: int,
        end_index: int,
        token_count: int,
        embedding: np.ndarray,
        position: int = 0,
        section_info: dict = None
    ):
        self.text = text
        self.start_index = start_index
        self.end_index = end_index
        self.token_count = token_count
        self.embedding = embedding
        self.position = position  # Position in the document
        self.section_info = section_info or {}  # Metadata about document structure
        self.sentence_id = f"sentence_{position}_{start_index}"  # Unique identifier


class EnhancedSemanticChunk:
    """Enhanced chunk class with additional metadata and metrics."""

    def __init__(
        self,
        text: str,
        start_index: int,
        end_index: int,
        token_count: int,
        sentences: List[EnhancedSemanticSentence],
        chunk_embedding: np.ndarray,
        chunk_id: str = None,
        semantic_coherence: float = 0.0,
        metadata: dict = None
    ):
        self.text = text
        self.start_index = start_index
        self.end_index = end_index
        self.token_count = token_count
        self.sentences = sentences
        self.chunk_embedding = chunk_embedding
        self.chunk_id = chunk_id or f"chunk_{start_index}_{end_index}"
        self.semantic_coherence = semantic_coherence  # Measure of internal coherence
        self.metadata = metadata or {}
        self.sentence_count = len(sentences)
        self.avg_sentence_length = token_count / len(sentences) if sentences else 0


class EnhancedSemanticChunker:
    """Enhanced semantic chunker with multiple improvements."""

    def __init__(
        self,
        embedding_model: EnhancedEmbeddingModel,
        min_characters_per_sentence: int = 5,
        similarity_threshold: Optional[float] = None,
        similarity_percentile: Optional[float] = 90,
        similarity_window: int = 1,
        mode: str = "cumulative",
        initial_sentences: int = 1,
        min_sentences: int = 1,
        chunk_size: int = 200,
        min_chunk_size: int = 50,
        threshold_step: float = 0.05,
        sep: str = "🐮🍺",
        enable_optimization: bool = True,
        optimization_method: str = "binary_search",  # binary_search, percentile, golden_section
        adaptive_threshold: bool = True,
        context_window_size: int = 3,  # Context window for improved similarity calculation
        min_similarity_for_merge: float = 0.1  # Minimum similarity to merge chunks
    ):
        self.embedding_model = embedding_model
        self.min_characters_per_sentence = min_characters_per_sentence
        self.similarity_threshold = similarity_threshold
        self.similarity_percentile = similarity_percentile
        self.similarity_window = similarity_window
        self.mode = mode
        self.initial_sentences = initial_sentences
        self.min_sentences = min_sentences
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size
        self.threshold_step = threshold_step
        self.sep = sep
        self.enable_optimization = enable_optimization
        self.optimization_method = optimization_method
        self.adaptive_threshold = adaptive_threshold
        self.context_window_size = context_window_size
        self.min_similarity_for_merge = min_similarity_for_merge
        
        self.splitter = GeneralTextSplitter(max_sentence_length=120)
        
        # Performance optimization caches
        self._sentence_cache = {}
        self._similarity_cache = {}

    def _count_tokens(self, text: str) -> int:
        """Count approximate tokens in text based on whitespace splitting."""
        return len(text.split())

    def _count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens for a batch of texts."""
        return [self._count_tokens(t) for t in texts]

    def _split_sentences(self, text: str) -> List[str]:
        """Fast sentence splitting while maintaining accuracy."""
        sentences = self.splitter.split_text(text)
        # Filter out very short sentences
        filtered_sentences = [s for s in sentences if len(s.strip()) >= self.min_characters_per_sentence]
        return filtered_sentences

    def _compute_contextual_embeddings(self, raw_sentences: List[str], context_window: int) -> List[np.ndarray]:
        """Compute embeddings considering context around each sentence."""
        embeddings = []
        for i in range(len(raw_sentences)):
            # Build context window around the sentence
            context_start = max(0, i - context_window)
            context_end = min(len(raw_sentences), i + context_window + 1)
            context_text = "".join(raw_sentences[context_start:context_end])
            # Get embedding for the sentence with context
            sentence_embedding = self.embedding_model.embed_batch([context_text])[0]
            embeddings.append(sentence_embedding)
        return embeddings

    def _prepare_sentences(self, text: str) -> List[EnhancedSemanticSentence]:
        """Prepare sentences with precomputed information."""
        if not text.strip():
            return []

        raw_sentences = self._split_sentences(text)

        if not raw_sentences:
            return []

        # Compute start/end indices
        sentence_indices = []
        current_idx = 0
        for sent in raw_sentences:
            start_idx = text.find(sent, current_idx)
            end_idx = start_idx + len(sent)
            sentence_indices.append((start_idx, end_idx))
            current_idx = end_idx

        # Compute embeddings with context if enabled
        if self.context_window_size > 0:
            embeddings = self._compute_contextual_embeddings(raw_sentences, self.context_window_size)
        else:
            # Create sentence groups for embedding computation (original approach)
            sentence_groups = []
            for i in range(len(raw_sentences)):
                group = []
                for j in range(i - self.similarity_window, i + self.similarity_window + 1):
                    if 0 <= j < len(raw_sentences):
                        group.append(raw_sentences[j])
                sentence_groups.append("".join(group))
            embeddings = self.embedding_model.embed_batch(sentence_groups)

        # Compute token counts
        token_counts = self._count_tokens_batch(raw_sentences)
        
        # Create EnhancedSemanticSentence objects
        sentences = []
        for i, (sent, (start_idx, end_idx), count, embedding) in enumerate(
            zip(raw_sentences, sentence_indices, token_counts, embeddings)
        ):
            sentence_obj = EnhancedSemanticSentence(
                text=sent,
                start_index=start_idx,
                end_index=end_idx,
                token_count=count,
                embedding=embedding,
                position=i
            )
            sentences.append(sentence_obj)

        return sentences

    def _get_semantic_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return self.embedding_model.similarity(embedding1, embedding2)

    def _compute_group_embedding(self, sentences: List[EnhancedSemanticSentence]) -> np.ndarray:
        """Compute mean embedding for a group of sentences."""
        if not sentences:
            raise ValueError("Cannot compute embedding for empty sentence list")
        
        # Weighted average based on token count
        total_weight = sum(sent.token_count for sent in sentences)
        if total_weight == 0:
            # Fallback to equal weighting if all token counts are zero
            return np.mean([sent.embedding for sent in sentences], axis=0, dtype=np.float32)
        
        weighted_sum = np.zeros_like(sentences[0].embedding, dtype=np.float32)
        for sent in sentences:
            weighted_sum += sent.embedding * sent.token_count
        return weighted_sum / total_weight

    def _compute_semantic_coherence(self, sentences: List[EnhancedSemanticSentence]) -> float:
        """Compute semantic coherence of a group of sentences."""
        if len(sentences) < 2:
            return 1.0  # Single sentence is perfectly coherent
        
        similarities = []
        for i in range(len(sentences) - 1):
            sim = self._get_semantic_similarity(sentences[i].embedding, sentences[i + 1].embedding)
            similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0

    def _compute_pairwise_similarities(self, sentences: List[EnhancedSemanticSentence]) -> List[float]:
        """Compute all pairwise similarities between sentences."""
        if len(sentences) < 2:
            return []
        
        # Use cache if available
        cache_key = tuple(s.sentence_id for s in sentences)
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]
        
        similarities = [
            self._get_semantic_similarity(
                sentences[i].embedding, sentences[i + 1].embedding
            )
            for i in range(len(sentences) - 1)
        ]
        
        # Cache the result
        self._similarity_cache[cache_key] = similarities
        return similarities

    def _calculate_threshold_via_binary_search(self, sentences: List[EnhancedSemanticSentence]) -> float:
        """Calculate similarity threshold via binary search."""
        if len(sentences) < 2:
            return 0.5  # Default threshold for single sentence
        
        token_counts = [sent.token_count for sent in sentences]
        similarities = self._compute_pairwise_similarities(sentences)

        if not similarities:
            return 0.5

        median = np.median(similarities)
        std = np.std(similarities)

        low = max(median - 1 * std, 0.0)
        high = min(median + 1 * std, 1.0)

        iterations = 0
        threshold = (low + high) / 2

        while abs(high - low) > self.threshold_step and iterations < 20:
            threshold = (low + high) / 2
            split_indices = self._get_split_indices(similarities, threshold)
            
            if len(split_indices) < 2:
                break  # Cannot split further
                
            # Extract token counts of each segment
            segment_lengths = []
            for i in range(len(split_indices) - 1):
                start = split_indices[i]
                end = split_indices[i + 1]
                segment_token_count = sum(token_counts[start:end])
                segment_lengths.append(segment_token_count)

            # Check if all segments meet size requirements
            valid_sizes = all(
                self.min_chunk_size <= length <= self.chunk_size
                for length in segment_lengths
            )
            
            if valid_sizes:
                break
            elif any(length > self.chunk_size for length in segment_lengths):
                # Increase threshold to reduce number of groups
                low = threshold + self.threshold_step
            else:
                # Decrease threshold to increase number of groups
                high = threshold - self.threshold_step

            iterations += 1

        return threshold

    def _calculate_threshold_via_percentile(self, sentences: List[EnhancedSemanticSentence]) -> float:
        """Calculate similarity threshold via percentile."""
        if len(sentences) < 2:
            return 0.5
        
        all_similarities = self._compute_pairwise_similarities(sentences)
        if not all_similarities:
            return 0.5
        
        # Use complementary percentile for threshold calculation
        return float(np.percentile(all_similarities, 100 - self.similarity_percentile))

    def _calculate_similarity_threshold(self, sentences: List[EnhancedSemanticSentence]) -> float:
        """Calculate similarity threshold based on configuration."""
        if self.similarity_threshold is not None:
            return self.similarity_threshold
        elif self.optimization_method == "percentile":
            return self._calculate_threshold_via_percentile(sentences)
        else:
            return self._calculate_threshold_via_binary_search(sentences)

    def _get_split_indices(self, similarities: List[float], threshold: float = None) -> List[int]:
        """Get indices of sentences to split at."""
        if threshold is None:
            threshold = (
                self.similarity_threshold
                if self.similarity_threshold is not None
                else 0.5
            )

        # Get indices of sentences where similarity drops below threshold
        splits = [
            i + 1
            for i, s in enumerate(similarities)
            if s <= threshold and i + 1 < len(similarities) + 1
        ]

        # Ensure we have start and end markers
        splits = [0] + splits + [len(similarities) + 1]

        # Ensure minimum sentences per group
        i = 0
        while i < len(splits) - 1:
            if splits[i + 1] - splits[i] < self.min_sentences:
                splits.pop(i + 1)
            else:
                i += 1
        
        # Remove duplicates and ensure valid indices
        splits = sorted(list(set(splits)))
        return splits

    def _group_sentences_cumulative(self, sentences: List[EnhancedSemanticSentence]) -> List[List[EnhancedSemanticSentence]]:
        """Group sentences based on cumulative semantic similarity."""
        groups = []
        if not sentences:
            return groups

        current_group = sentences[:self.initial_sentences]
        current_embedding = self._compute_group_embedding(current_group)

        for sentence in sentences[self.initial_sentences:]:
            similarity = self._get_semantic_similarity(
                current_embedding, sentence.embedding
            )
            
            # Check if similarity is above threshold and group won't exceed size limits
            current_group_tokens = sum(s.token_count for s in current_group) + sentence.token_count
            if similarity >= self.similarity_threshold and current_group_tokens <= self.chunk_size:
                current_group.append(sentence)
                current_embedding = self._compute_group_embedding(current_group)
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [sentence]
                current_embedding = sentence.embedding

        if current_group:
            groups.append(current_group)

        return groups

    def _group_sentences_window(self, sentences: List[EnhancedSemanticSentence]) -> List[List[EnhancedSemanticSentence]]:
        """Group sentences based on semantic similarity using a window-based approach."""
        if len(sentences) < 2:
            return [sentences] if sentences else []
        
        similarities = self._compute_pairwise_similarities(sentences)
        split_indices = self._get_split_indices(similarities, self.similarity_threshold)
        
        # Handle edge case where no splits are needed
        if len(split_indices) <= 1:
            return [sentences]
        
        groups = [
            sentences[split_indices[i] : split_indices[i + 1]]
            for i in range(len(split_indices) - 1)
        ]
        return groups

    def _group_sentences(self, sentences: List[EnhancedSemanticSentence]) -> List[List[EnhancedSemanticSentence]]:
        """Group sentences based on semantic similarity."""
        if self.mode == "cumulative":
            return self._group_sentences_cumulative(sentences)
        else:
            return self._group_sentences_window(sentences)

    def _create_chunk(self, sentences: List[EnhancedSemanticSentence]) -> EnhancedSemanticChunk:
        """Create a chunk from a list of sentences."""
        if not sentences:
            raise ValueError("Cannot create chunk from empty sentence list")

        text = "".join(sent.text for sent in sentences)
        token_count = sum(sent.token_count for sent in sentences) + (len(sentences) - 1)
        
        # Compute chunk embedding
        chunk_embedding = self._compute_group_embedding(sentences)
        
        # Compute semantic coherence
        semantic_coherence = self._compute_semantic_coherence(sentences)

        chunk_metadata = {
            'sentence_count': len(sentences),
            'avg_sentence_length': token_count / len(sentences) if sentences else 0,
            'positions': [s.position for s in sentences],
            'start_position': sentences[0].position if sentences else 0,
            'end_position': sentences[-1].position if sentences else 0
        }

        return EnhancedSemanticChunk(
            text=text,
            start_index=sentences[0].start_index,
            end_index=sentences[-1].end_index,
            token_count=token_count,
            sentences=sentences,
            chunk_embedding=chunk_embedding,
            semantic_coherence=semantic_coherence,
            metadata=chunk_metadata
        )

    def _split_chunks(self, sentence_groups: List[List[EnhancedSemanticSentence]]) -> List[EnhancedSemanticChunk]:
        """Split sentence groups into chunks that respect chunk_size."""
        chunks = []

        for group in sentence_groups:
            current_chunk_sentences = []
            current_tokens = 0

            for sentence in group:
                test_tokens = (
                    current_tokens
                    + sentence.token_count
                    + (1 if current_chunk_sentences else 0)
                )

                if test_tokens <= self.chunk_size:
                    current_chunk_sentences.append(sentence)
                    current_tokens = test_tokens
                else:
                    if current_chunk_sentences:
                        chunks.append(self._create_chunk(current_chunk_sentences))
                    current_chunk_sentences = [sentence]
                    current_tokens = sentence.token_count

            if current_chunk_sentences:
                chunks.append(self._create_chunk(current_chunk_sentences))

        return chunks

    def chunk(self, text: str) -> List[EnhancedSemanticChunk]:
        """Split text into semantically coherent chunks."""
        if not text.strip():
            return []

        sentences = self._prepare_sentences(text)
        
        if len(sentences) <= self.min_sentences:
            if sentences:
                return [self._create_chunk(sentences)]

        # Calculate similarity threshold
        if self.adaptive_threshold:
            self.similarity_threshold = self._calculate_similarity_threshold(sentences)
        
        sentence_groups = self._group_sentences(sentences)
        chunks = self._split_chunks(sentence_groups)
        
        # Post-process chunks for optimization
        optimized_chunks = self._optimize_chunks(chunks)
        
        return optimized_chunks

    def _optimize_chunks(self, chunks: List[EnhancedSemanticChunk]) -> List[EnhancedSemanticChunk]:
        """Optimize chunks by merging small chunks with similar neighbors."""
        if len(chunks) <= 1:
            return chunks

        optimized = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            # Check if current chunk is too small and can be merged
            if (current_chunk.token_count < self.min_chunk_size and 
                i + 1 < len(chunks)):  # Has next chunk
                next_chunk = chunks[i + 1]
                
                # Calculate similarity between current and next chunk
                similarity = self._get_semantic_similarity(
                    current_chunk.chunk_embedding, 
                    next_chunk.chunk_embedding
                )
                
                # Merge if similarity is above threshold
                if similarity >= self.min_similarity_for_merge:
                    # Create merged chunk
                    merged_sentences = current_chunk.sentences + next_chunk.sentences
                    merged_chunk = self._create_chunk(merged_sentences)
                    optimized.append(merged_chunk)
                    i += 2  # Skip both chunks
                else:
                    optimized.append(current_chunk)
                    i += 1
            else:
                optimized.append(current_chunk)
                i += 1
        
        return optimized

    def get_statistics(self, chunks: List[EnhancedSemanticChunk]) -> Dict:
        """Get statistics about the chunking results."""
        if not chunks:
            return {}
        
        token_counts = [chunk.token_count for chunk in chunks]
        coherences = [chunk.semantic_coherence for chunk in chunks]
        
        stats = {
            'total_chunks': len(chunks),
            'total_tokens': sum(token_counts),
            'avg_chunk_size': np.mean(token_counts),
            'median_chunk_size': np.median(token_counts),
            'std_chunk_size': np.std(token_counts),
            'min_chunk_size': min(token_counts),
            'max_chunk_size': max(token_counts),
            'avg_semantic_coherence': np.mean(coherences) if coherences else 0.0,
            'avg_sentences_per_chunk': np.mean([len(chunk.sentences) for chunk in chunks]),
        }
        
        return stats


def create_default_enhanced_chunker(chunk_size: int = 200) -> EnhancedSemanticChunker:
    """Create a default enhanced semantic chunker with reasonable parameters."""
    # Create a default model
    model = EnhancedEmbeddingModel(embedding_dim=128)
    
    chunker = EnhancedSemanticChunker(
        embedding_model=model,
        chunk_size=chunk_size,
        similarity_percentile=85,  # Slightly lower for better grouping
        mode="cumulative",
        min_chunk_size=50,
        context_window_size=2,  # Include context for better embeddings
        adaptive_threshold=True,
        enable_optimization=True
    )
    
    return chunker