# -*- coding: utf-8 -*-

import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/" + ".."))


from typing import List, Union, Any
import numpy as np
from src.semantic_chunk import SemanticChunker, SemanticChunk

# 假设 SemanticChunker 和 SemanticChunk 已经被实现并导入
# from semantic_chunker_module import SemanticChunker, SemanticChunk

class SDPMChunker(SemanticChunker):
    """SDPMChunker implementation using the Semantic Document Partitioning Method (SDPM).

    This class extends SemanticChunker by adding the ability to merge similar groups
    with a skip window and split the merged groups into size-appropriate chunks.

    Args:
        embedding_model (Union[str, Any]): Sentence embedding model to use.
        similarity_threshold (Union[str, float, int]): Minimum similarity score or percentile.
        chunk_size (int): Maximum token count for a chunk.
        similarity_window (int): Number of sentences to consider for similarity threshold calculation.
        min_sentences (int): Minimum number of sentences per chunk.
        skip_window (int): Number of chunks to skip when looking for similarities.
        min_chunk_size (int): Minimum number of tokens per chunk.
        min_characters_per_sentence (int): Minimum number of characters per sentence.
        threshold_step (float): Step size for similarity threshold calculation.
        delim (Union[str, List[str]]): Delimiters to split sentences on.
    """

    def __init__(
        self,
        embedding_model: Union[str, Any] = "minishlab/potion-base-8M",
        similarity_threshold: Union[str, float, int] = "auto",
        chunk_size: int = 512,
        similarity_window: int = 1,
        min_sentences: int = 1,
        skip_window: int = 1,
        min_chunk_size: int = 2,
        min_characters_per_sentence: int = 12,
        threshold_step: float = 0.01,
        delim: Union[str, List[str]] = [".", "!", "?", "\n"],
    ):
        super().__init__(
            embedding_model=embedding_model,
            min_characters_per_sentence=min_characters_per_sentence,
            similarity_threshold=similarity_threshold,
            similarity_percentile=None,  # Assuming percentile is handled via threshold
            similarity_window=similarity_window,
            mode="cumulative",  # Assuming SDPM uses cumulative mode
            initial_sentences=1,  # Initial sentences for grouping
            min_sentences=min_sentences,
            chunk_size=chunk_size,
            min_chunk_size=min_chunk_size,
            threshold_step=threshold_step,
            sep=delim,
        )
        self.skip_window = skip_window

    def chunk(self, text: str) -> List[SemanticChunk]:
        """Split text into chunks using the SDPM approach.

        Args:
            text (str): Input text to be chunked.

        Returns:
            List[SemanticChunk]: List of SemanticChunk objects.
        """
        if not text.strip():
            return []

        # Step 1: Initial semantic chunking using the base class
        initial_chunks = super().chunk(text)

        if len(initial_chunks) <= 1:
            return initial_chunks

        # Step 2: Merge similar chunks with skip window
        merged_chunks = self._skip_and_merge(initial_chunks, self.similarity_threshold)

        # Step 3: Split merged chunks into size-appropriate chunks
        final_chunks = self._split_chunks(merged_chunks)

        return final_chunks

    def _skip_and_merge(self, chunks: List[SemanticChunk], similarity_threshold: float) -> List[SemanticChunk]:
        """Merge similar chunks considering skip window.

        Args:
            chunks (List[SemanticChunk]): Initial list of SemanticChunk objects.
            similarity_threshold (float): Threshold for semantic similarity.

        Returns:
            List[SemanticChunk]: Merged list of SemanticChunk objects.
        """
        if len(chunks) <= 1:
            return chunks

        merged_chunks = []
        embeddings = [chunk.chunk_embedding for chunk in chunks]

        i = 0
        while i < len(chunks):
            current_chunk = chunks[i]
            current_embedding = embeddings[i]

            # Determine the index to skip to
            skip_index = i + self.skip_window + 1
            if skip_index >= len(chunks):
                skip_index = len(chunks) - 1

            if skip_index <= i:
                # Prevent infinite loop if skip_index is not moving forward
                skip_index = i + 1
                if skip_index >= len(chunks):
                    merged_chunks.append(current_chunk)
                    break

            skipped_chunk = chunks[skip_index]
            skipped_embedding = embeddings[skip_index]

            similarity = self._compute_similarity(current_embedding, skipped_embedding)

            if similarity >= similarity_threshold:
                # Merge current_chunk and skipped_chunk
                merged_text = current_chunk.text + " " + skipped_chunk.text
                merged_start = current_chunk.start_index
                merged_end = skipped_chunk.end_index
                merged_token_count = current_chunk.token_count + skipped_chunk.token_count
                merged_sentences = current_chunk.sentences + skipped_chunk.sentences
                merged_embedding = self._compute_group_embedding(merged_sentences)

                merged_chunk = SemanticChunk(
                    text=merged_text,
                    start_index=merged_start,
                    end_index=merged_end,
                    token_count=merged_token_count,
                    sentences=merged_sentences,
                    chunk_embedding=merged_embedding,
                )

                # Replace the two chunks with the merged chunk
                chunks[i] = merged_chunk
                embeddings[i] = merged_embedding
                del chunks[skip_index]
                del embeddings[skip_index]
            else:
                # No merge, add the current chunk to merged_chunks
                merged_chunks.append(current_chunk)
                i += 1

        return merged_chunks

    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute the semantic similarity between two embeddings.

        Args:
            embedding1 (np.ndarray): Embedding of the first chunk.
            embedding2 (np.ndarray): Embedding of the second chunk.

        Returns:
            float: Cosine similarity between the two embeddings.
        """
        if not isinstance(embedding1, np.ndarray) or not isinstance(embedding2, np.ndarray):
            raise ValueError("Embeddings must be numpy arrays.")

        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def _split_chunks(self, chunks: List[SemanticChunk]) -> List[SemanticChunk]:
        """Split merged chunks into size-appropriate chunks based on chunk_size.

        Args:
            chunks (List[SemanticChunk]): List of merged SemanticChunk objects.

        Returns:
            List[SemanticChunk]: Final list of SemanticChunk objects.
        """
        final_chunks = []
        for chunk in chunks:
            if chunk.token_count <= self.chunk_size:
                final_chunks.append(chunk)
            else:
                # Split the chunk into smaller chunks
                sentences = chunk.sentences
                current_text = ""
                current_tokens = 0
                start_index = chunk.start_index
                for sentence in sentences:
                    sentence_tokens = self._count_tokens(sentence.text)
                    if current_tokens + sentence_tokens > self.chunk_size:
                        if current_text:
                            end_index = sentence.start_index
                            chunk_embedding = self._compute_group_embedding(current_text)
                            new_chunk = SemanticChunk(
                                text=current_text.strip(),
                                start_index=start_index,
                                end_index=end_index,
                                token_count=current_tokens,
                                sentences=sentences[:len(current_text)],
                                chunk_embedding=chunk_embedding,
                            )
                            final_chunks.append(new_chunk)
                            current_text = sentence.text + " "
                            current_tokens = sentence_tokens
                            start_index = sentence.start_index
                        else:
                            # Sentence itself exceeds chunk_size, force to add
                            final_chunks.append(chunk)
                            break
                    else:
                        current_text += sentence.text + " "
                        current_tokens += sentence_tokens
                if current_text:
                    end_index = sentences[-1].end_index
                    chunk_embedding = self._compute_group_embedding(current_text)
                    new_chunk = SemanticChunk(
                        text=current_text.strip(),
                        start_index=start_index,
                        end_index=end_index,
                        token_count=current_tokens,
                        sentences=sentences[:len(current_text)],
                        chunk_embedding=chunk_embedding,
                    )
                    final_chunks.append(new_chunk)
        return final_chunks

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a given text.

        Args:
            text (str): The text to count tokens for.

        Returns:
            int: The number of tokens.
        """
        # 这里假设每个单词或符号为一个token，具体实现可根据实际分词器调整
        return len(text.split())

    def __repr__(self) -> str:
        """Return a string representation of the SDPMChunker."""
        return (
            f"SDPMChunker(embedding_model={self.embedding_model}, "
            f"similarity_threshold={self.similarity_threshold}, "
            f"chunk_size={self.chunk_size}, "
            f"similarity_window={self.similarity_window}, "
            f"min_sentences={self.min_sentences}, "
            f"skip_window={self.skip_window}, "
            f"min_chunk_size={self.min_chunk_size}, "
            f"min_characters_per_sentence={self.min_characters_per_sentence}, "
            f"threshold_step={self.threshold_step}, "
            f"delim={self.sep})"
        )
        
if __name__ == '__main__':
    # Example usage
    chunker = SDPMChunker()
    text = "This is a sample text. It will be chunked using the SDPM method. The method is based on semantic similarity."
    chunks = chunker.chunk(text)
    for idx, chunk in enumerate(chunks):
        print(f"Chunk {idx + 1}: {chunk.text}")
        print(f"Start Index: {chunk.start_index}, End Index: {chunk.end_index}")
        print(f"Token Count: {chunk.token_count}")
        print(f"Sentences: {len(chunk.sentences)}")
        print(f"Chunk Embedding: {chunk.chunk_embedding}\n")
    print(chunker)