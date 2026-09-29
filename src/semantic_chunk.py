# -*- coding: utf-8 -*-

import sys
import os
import re

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/" + ".."))

import numpy as np
import warnings
from typing import List, Optional

from util.sentence_split import GeneralTextSplitter

# 下方定义的类与方法用于处理文本的语义分块操作，包含分句、语义相似性计算、
# 分组与最终chunk生成的流程。还提供对外接口函数实现根据query选出最匹配文本块的功能。


class EmbeddingModel:
    """A placeholder embedding model class.

    This class simulates an embedding model with:
    - A `embed_batch` method to embed a batch of texts.
    - A `similarity` method to compute similarity between two embeddings.

    Note:
        In a real-world scenario, this would load a specific model
        (e.g., a SentenceTransformer model or a proprietary large language model)
        based on the model_name. Here, we just simulate embeddings with random vectors.
    """

    def __init__(self, embedding_client, embedding_dim: int = 128):
        # 中文注释: 初始化嵌入模型，实际实现中可加载相应的模型文件。
        self.embedding_client = embedding_client
        self.embedding_dim = embedding_dim

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Embed a batch of texts and return a list of embedding vectors.

        Args:
            texts (List[str]): List of texts to embed.

        Returns:
            List[np.ndarray]: List of embedding vectors.
        """
        # 中文注释: 返回模拟的随机嵌入向量，实际情况需调用真实模型方法。
        # return [np.random.rand(self.embedding_dim).astype(np.float32) for _ in texts]
        # embeddings = embedding_api(texts)
        # 每十个文本一组，避免一次请求过多文本
        batch_size = 10
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            response = self.embedding_client.get_embeddings(batch_texts)
            batch_embeddings = response.get("data", {}).get("resultList", [])
            embeddings.extend(batch_embeddings)
        return [np.array(embedding).astype(np.float32) for embedding in embeddings]

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1 (np.ndarray): Embedding vector 1.
            embedding2 (np.ndarray): Embedding vector 2.

        Returns:
            float: Cosine similarity between the two embeddings.
        """
        norm1 = float(np.linalg.norm(embedding1))
        norm2 = float(np.linalg.norm(embedding2))
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))


class Sentence:
    """A simple Sentence class to store raw sentence text and metadata."""

    def __init__(self, text: str, start_index: int, end_index: int):
        # 中文注释: 基础句子类，用于存储句子文本和起始结束位置。
        self.text = text
        self.start_index = start_index
        self.end_index = end_index


class SemanticSentence(Sentence):
    """A SemanticSentence class that includes embedding and token count information."""

    def __init__(
        self,
        text: str,
        start_index: int,
        end_index: int,
        token_count: int,
        embedding: np.ndarray,
    ):
        # 中文注释: 扩展的句子类，包含嵌入向量和token计数信息。
        super().__init__(text, start_index, end_index)
        self.token_count = token_count
        self.embedding = embedding


class SemanticChunk:
    """A SemanticChunk class representing a coherent text chunk."""

    def __init__(
        self,
        text: str,
        start_index: int,
        end_index: int,
        token_count: int,
        sentences: List[SemanticSentence],
        chunk_embedding: np.ndarray,
    ):
        # 中文注释: 语义块类，包含块文本、起止位置、token计数及所属句子列表、以及预先计算好的chunk embedding。
        self.text = text
        self.start_index = start_index
        self.end_index = end_index
        self.token_count = token_count
        self.sentences = sentences
        self.chunk_embedding = chunk_embedding


class SemanticChunker:
    """A SemanticChunker class to split text into semantically coherent chunks.

    This class:
    - Splits text into sentences.
    - Computes embeddings for sentences.
    - Groups sentences based on semantic similarity.
    - Splits groups into chunks that respect a specified maximum size.

    Attributes:
        embedding_model (EmbeddingModel): The embedding model.
        min_characters_per_sentence (int): Minimum char length to consider a valid sentence.
        similarity_threshold (Optional[float]): Fixed similarity threshold or None.
        similarity_percentile (Optional[float]): Percentile to compute dynamic threshold.
        similarity_window (int): Window size for window-based grouping mode.
        mode (str): "cumulative" or "window" grouping mode.
        initial_sentences (int): Initial number of sentences to start the cumulative grouping.
        min_sentences (int): Minimum number of sentences per chunk/group.
        chunk_size (int): Maximum token count allowed per chunk.
        min_chunk_size (int): Minimum token count per chunk.
        threshold_step (float): Step size for binary search threshold adjustments.
        sep (str): Separator for sentence splitting.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
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
    ):
        # 中文注释: 初始化分块器参数，支持两种模式并设置各种限制参数和分句分隔符。
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
        self.splitter = GeneralTextSplitter(max_sentence_length=120)

    def _count_tokens(self, text: str) -> int:
        """Count approximate tokens in text based on whitespace splitting.

        Args:
            text (str): Input text.

        Returns:
            int: Approximate token count.
        """
        # Model-agnostic approximation only. CJK characters are counted explicitly
        # so Chinese text is not collapsed into one whitespace-delimited token.
        return len(
            re.findall(
                r"[\u4e00-\u9fff]|[A-Za-z0-9]+(?:['’_-][A-Za-z0-9]+)?|[^\s]",
                text,
            )
        )

    def _count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens for a batch of texts.

        Args:
            texts (List[str]): Input texts.

        Returns:
            List[int]: List of token counts.
        """
        # 中文注释: 批量计算token数。
        return [self._count_tokens(t) for t in texts]

    def _split_sentences(self, text: str) -> List[str]:
        """Fast sentence splitting while maintaining accuracy.

        This method is faster than using regex for sentence splitting and
        more accurate than using spaCy sentence tokenizer.

        Args:
            text (str): Input text to be split into sentences.

        Returns:
            List[str]: List of sentences.
        """
        sentences = self.splitter.split_text(text)
        return sentences

    def _compute_similarity_threshold(self, all_similarities: List[float]) -> float:
        """Compute similarity threshold based on percentile if specified."""
        # 中文注释: 若设置了固定threshold则返回，否则按percentile计算。
        if self.similarity_threshold is not None:
            return self.similarity_threshold
        else:
            return float(np.percentile(all_similarities, self.similarity_percentile))

    def _prepare_sentences(self, text: str) -> List[SemanticSentence]:
        """Prepare sentences with precomputed information.

        Args:
            text (str): Input text to be processed.

        Returns:
            List[SemanticSentence]: List of Sentence objects with precomputed embeddings.
        """
        # 中文注释: 将文本分句，计算句子embedding和token数，并构建SemanticSentence对象列表。
        if not text.strip():
            return []

        raw_sentences = self._split_sentences(text)

        # Compute start/end indices
        sentence_indices = []
        current_idx = 0
        for sent in raw_sentences:
            start_idx = text.find(sent, current_idx)
            end_idx = start_idx + len(sent)
            sentence_indices.append((start_idx, end_idx))
            current_idx = end_idx

        # Create sentence groups for embedding computation
        sentence_groups = []
        for i in range(len(raw_sentences)):
            group = []
            for j in range(i - self.similarity_window, i + self.similarity_window + 1):
                if 0 <= j < len(raw_sentences):
                    group.append(raw_sentences[j])
            sentence_groups.append("".join(group))

        # Compute embeddings
        embeddings = self.embedding_model.embed_batch(sentence_groups)
        # Compute token counts
        token_counts = self._count_tokens_batch(raw_sentences)
        sentences = [
            SemanticSentence(
                text=sent,
                start_index=start_idx,
                end_index=end_idx,
                token_count=count,
                embedding=embedding,
            )
            for sent, (start_idx, end_idx), count, embedding in zip(
                raw_sentences, sentence_indices, token_counts, embeddings
            )
        ]

        return sentences

    def _get_semantic_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        # 中文注释: 简化调用embedding_model的similarity方法。
        return self.embedding_model.similarity(embedding1, embedding2)

    def _compute_group_embedding(self, sentences: List[SemanticSentence]) -> np.ndarray:
        """Compute mean embedding for a group of sentences.

        Args:
            sentences (List[SemanticSentence]): Input sentences.

        Returns:
            np.ndarray: Mean embedding vector.
        """
        # 中文注释: 加权平均，考虑token数对平均值的影响。
        return np.divide(
            np.sum([(sent.embedding * sent.token_count) for sent in sentences], axis=0),
            np.sum([sent.token_count for sent in sentences]),
            dtype=np.float32,
        )

    def _compute_pairwise_similarities(
        self, sentences: List[SemanticSentence]
    ) -> List[float]:
        """Compute all pairwise similarities between sentences.

        Args:
            sentences (List[SemanticSentence]): Input sentences.

        Returns:
            List[float]: List of similarities between consecutive sentences.
        """
        # 中文注释: 计算相邻句子间相似度。
        return [
            self._get_semantic_similarity(
                sentences[i].embedding, sentences[i + 1].embedding
            )
            for i in range(len(sentences) - 1)
        ]

    def _get_split_indices(
        self, similarities: List[float], threshold: float = None
    ) -> List[int]:
        """Get indices of sentences to split at.

        Args:
            similarities (List[float]): List of pairwise similarities.
            threshold (float): Similarity threshold.

        Returns:
            List[int]: Indices indicating the split points.
        """
        # 中文注释: 根据相似度阈值确定分组的切分点。
        if threshold is None:
            threshold = (
                self.similarity_threshold
                if self.similarity_threshold is not None
                else 0.5
            )

        # Get indices of sentences that are below the threshold
        splits = [
            i + 1
            for i, s in enumerate(similarities)
            if s <= threshold and i + 1 < len(similarities) + 1
        ]

        # Add the start and end of the text
        splits = [0] + splits + [len(similarities) + 1]

        # Ensure minimum sentences per group
        i = 0
        while i < len(splits) - 1:
            if splits[i + 1] - splits[i] < self.min_sentences:
                splits.pop(i + 1)
            else:
                i += 1
        return splits

    def _calculate_threshold_via_binary_search(
        self, sentences: List[SemanticSentence]
    ) -> float:
        """Calculate similarity threshold via binary search.

        Args:
            sentences (List[SemanticSentence]): Input sentences.

        Returns:
            float: Computed threshold.
        """
        # 中文注释: 通过binary search来寻找合适的相似度阈值，使分块满足大小要求。
        token_counts = [sent.token_count for sent in sentences]

        similarities = self._compute_pairwise_similarities(sentences)

        median = np.median(similarities)
        std = np.std(similarities)

        low = max(median - 1 * std, 0.0)
        high = min(median + 1 * std, 1.0)

        iterations = 0
        threshold = (low + high) / 2

        while abs(high - low) > self.threshold_step:
            threshold = (low + high) / 2
            split_indices = self._get_split_indices(similarities, threshold)
            # Extract token counts of each segment
            segment_lengths = []
            for i in range(len(split_indices) - 1):
                start = split_indices[i]
                end = split_indices[i + 1]
                segment_token_count = sum(token_counts[start:end])
                segment_lengths.append(segment_token_count)

            # Check condition: ideally segment lengths between min_chunk_size and chunk_size
            if all(
                self.min_chunk_size <= length <= self.chunk_size
                for length in segment_lengths
            ):
                break
            elif any(length > self.chunk_size for length in segment_lengths):
                # 若有超过chunk_size的分段，增加threshold降低分组数量
                low = threshold + self.threshold_step
            else:
                # 若有分段小于min_chunk_size，减少threshold增加分组数量
                high = threshold - self.threshold_step

            iterations += 1
            if iterations > 10:
                warnings.warn(
                    "Too many iterations in threshold calculation, stopping...",
                    stacklevel=2,
                )
                break

        return threshold

    def _calculate_threshold_via_percentile(
        self, sentences: List[SemanticSentence]
    ) -> float:
        """Calculate similarity threshold via percentile.

        Args:
            sentences (List[SemanticSentence]): Input sentences.

        Returns:
            float: Computed threshold.
        """
        # 中文注释: 根据percentile计算相似度阈值。
        all_similarities = self._compute_pairwise_similarities(sentences)
        return float(np.percentile(all_similarities, 100 - self.similarity_percentile))

    def _calculate_similarity_threshold(
        self, sentences: List[SemanticSentence]
    ) -> float:
        """Calculate similarity threshold either through binary search or percentile.

        Args:
            sentences (List[SemanticSentence]): Input sentences.

        Returns:
            float: Computed similarity threshold.
        """
        # 中文注释: 根据配置决定使用固定值、percentile还是binary search。
        if self.similarity_threshold is not None:
            return self.similarity_threshold
        elif self.similarity_percentile is not None:
            return self._calculate_threshold_via_percentile(sentences)
        else:
            return self._calculate_threshold_via_binary_search(sentences)

    def _group_sentences_cumulative(
        self, sentences: List[SemanticSentence]
    ) -> List[List[SemanticSentence]]:
        """Group sentences based on cumulative semantic similarity.

        Args:
            sentences (List[SemanticSentence]): Input sentences.

        Returns:
            List[List[SemanticSentence]]: Grouped sentences.
        """
        # 中文注释: 累计方式分组，根据当前组平均embedding与新句子的相似度决定是否加入同组。
        groups = []
        if not sentences:
            return groups

        current_group = sentences[: self.initial_sentences]
        current_embedding = self._compute_group_embedding(current_group)

        for sentence in sentences[self.initial_sentences :]:
            similarity = self._get_semantic_similarity(
                current_embedding, sentence.embedding
            )
            if similarity >= self.similarity_threshold:
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

    def _group_sentences_window(
        self, sentences: List[SemanticSentence]
    ) -> List[List[SemanticSentence]]:
        """Group sentences based on semantic similarity using a window-based approach.

        Args:
            sentences (List[SemanticSentence]): Input sentences.

        Returns:
            List[List[SemanticSentence]]: Grouped sentences.
        """
        # 中文注释: 窗口模式下根据相邻句子相似度和阈值分割。
        similarities = self._compute_pairwise_similarities(sentences)
        split_indices = self._get_split_indices(similarities, self.similarity_threshold)
        groups = [
            sentences[split_indices[i] : split_indices[i + 1]]
            for i in range(len(split_indices) - 1)
        ]
        return groups

    def _group_sentences(
        self, sentences: List[SemanticSentence]
    ) -> List[List[SemanticSentence]]:
        """Group sentences based on semantic similarity, either cumulatively or by window.

        Args:
            sentences (List[SemanticSentence]): Input sentences.

        Returns:
            List[List[SemanticSentence]]: Grouped sentences.
        """
        # 中文注释: 根据mode选择分组方式。
        if self.mode == "cumulative":
            return self._group_sentences_cumulative(sentences)
        else:
            return self._group_sentences_window(sentences)

    def _create_chunk(self, sentences: List[SemanticSentence]) -> SemanticChunk:
        """Create a chunk from a list of sentences.

        Args:
            sentences (List[SemanticSentence]): Input sentences.

        Returns:
            SemanticChunk: Created chunk.
        """
        # 中文注释: 将句子列表合并为一个chunk，并计算chunk的embedding。
        if not sentences:
            raise ValueError("Cannot create chunk from empty sentence list")

        text = "".join(sent.text for sent in sentences)
        token_count = sum(sent.token_count for sent in sentences) + (len(sentences) - 1)
        # 在这里计算并存储chunk级别的embedding
        chunk_embedding = self._compute_group_embedding(sentences)

        return SemanticChunk(
            text=text,
            start_index=sentences[0].start_index,
            end_index=sentences[-1].end_index,
            token_count=token_count,
            sentences=sentences,
            chunk_embedding=chunk_embedding,
        )

    def _split_chunks(
        self, sentence_groups: List[List[SemanticSentence]]
    ) -> List[SemanticChunk]:
        """Split sentence groups into chunks that respect chunk_size.

        Args:
            sentence_groups (List[List[SemanticSentence]]): Semantically coherent sentence groups.

        Returns:
            List[SemanticChunk]: List of SemanticChunk objects.
        """
        # 中文注释: 将句子组按照chunk_size划分为更小的chunk。
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

    def chunk(self, text: str) -> List[SemanticChunk]:
        """Split text into semantically coherent chunks using a two-pass approach.

        Args:
            text (str): Input text to be chunked.

        Returns:
            List[SemanticChunk]: List of SemanticChunk objects containing chunked text and metadata.
        """
        # 中文注释: 主流程：先准备句子与embedding，再计算相似度阈值，然后分组和二次分块。
        if not text.strip():
            return []

        sentences = self._prepare_sentences(text)
        if len(sentences) <= self.min_sentences:
            # 所有句子过少，直接作为一个chunk返回
            if sentences:
                return [self._create_chunk(sentences)]

        self.similarity_threshold = self._calculate_similarity_threshold(sentences)
        sentence_groups = self._group_sentences(sentences)
        chunks = self._split_chunks(sentence_groups)
        return chunks
