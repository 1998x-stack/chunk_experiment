import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/" + ".."))

import re
from typing import List, Optional, Tuple, Dict, Union
from util.sentence_split import GeneralTextSplitter
import logging


class EnhancedRecursiveCharacterTextSplitter:
    """Enhanced recursive character text splitter with multiple improvements."""

    def __init__(
        self,
        chunk_size: int = 200,
        chunk_overlap: int = 50,
        sep_type: str = "chunk_size",
        separators: Optional[List[str]] = None,
        length_function: Optional[callable] = None,  # Custom function to measure text length
        keep_separator: bool = False,  # Whether to keep separators in the chunks
        strip_separators: bool = True,  # Whether to strip separators from chunks
        is_separator_regex: bool = False,  # Whether separators are regex patterns
        add_start_index: bool = False,  # Whether to add start index metadata to chunks
        chunk_header_prefix: str = "",  # Prefix to add to each chunk
        chunk_header_suffix: str = "",  # Suffix to add to each chunk
        min_chunk_size: int = 0,  # Minimum chunk size to keep
        merge_small_chunks: bool = True,  # Whether to merge small chunks
        chunk_strategy: str = "balanced",  # balanced, aggressive, conservative
    ):
        """Initialize enhanced splitter.
        
        Args:
            chunk_size: Maximum length of each chunk
            chunk_overlap: Overlap between adjacent chunks
            sep_type: Type of splitting ('chunk_size', 'sentence', 'paragraph')
            separators: List of separators in order of preference
            length_function: Custom function to measure text length (default: len)
            keep_separator: Whether to keep separators in the chunks
            strip_separators: Whether to strip separators from chunk boundaries
            is_separator_regex: Whether separators are regex patterns
            add_start_index: Whether to add start index metadata
            chunk_header_prefix: String to prepend to each chunk
            chunk_header_suffix: String to append to each chunk
            min_chunk_size: Minimum chunk size to keep
            merge_small_chunks: Whether to merge small chunks with neighbors
            chunk_strategy: Strategy for chunking ('balanced', 'aggressive', 'conservative')
        """
        if separators is None:
            separators = ["\n\n", "\n", "。", "？", "！", "，", " ", ""]
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.sep_type = sep_type
        self.separators = separators
        self.length_function = length_function or len
        self.keep_separator = keep_separator
        self.strip_separators = strip_separators
        self.is_separator_regex = is_separator_regex
        self.add_start_index = add_start_index
        self.chunk_header_prefix = chunk_header_prefix
        self.chunk_header_suffix = chunk_header_suffix
        self.min_chunk_size = min_chunk_size
        self.merge_small_chunks = merge_small_chunks
        self.chunk_strategy = chunk_strategy
        
        # Initialize general splitter
        self.general_splitter = GeneralTextSplitter()
        
        # Validate inputs
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    def split_text(self, text: str, sep_type: str = None) -> List[str]:
        """Split text into smaller chunks."""
        if not text:
            return []
        
        # Use passed sep_type or instance default
        effective_sep_type = sep_type or self.sep_type
        
        if effective_sep_type == "sentence":
            return self._split_by_sentence(text)
        elif effective_sep_type == "paragraph":
            return self._split_by_paragraph(text)
        else:
            return self._split_by_size(text)

    def _split_by_size(self, text: str) -> List[str]:
        """Split text based on chunk size."""
        # Start recursive splitting
        splits = self._recursive_split(text, self.separators, self.chunk_size)
        # Merge splits and handle overlap
        chunks = self._merge_splits(splits)
        
        # Apply post-processing
        chunks = self._post_process_chunks(chunks)
        
        return chunks

    def _split_by_sentence(self, text: str) -> List[str]:
        """Split text by sentences with size constraints."""
        # Use the general splitter to get sentence chunks
        sentence_chunks = self.general_splitter.batch_chunk(
            text, 
            max_length=self.chunk_size, 
            overlap_size=self.chunk_overlap, 
            return_counts=False
        )
        
        # Flatten the result (batch_chunk returns nested lists)
        if isinstance(sentence_chunks, list) and sentence_chunks and isinstance(sentence_chunks[0], list):
            result = []
            for chunk_list in sentence_chunks:
                result.extend(chunk_list)
            final_chunks = result
        else:
            final_chunks = sentence_chunks
            
        # Apply post-processing
        final_chunks = self._post_process_chunks(final_chunks)
        
        return final_chunks

    def _split_by_paragraph(self, text: str) -> List[str]:
        """Split text by paragraphs with size constraints."""
        # Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            # Check if adding this paragraph would exceed chunk size
            test_chunk = current_chunk + ("\n\n" if current_chunk else "") + para
            
            if self.length_function(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                # Save current chunk if it's substantial enough
                if self.length_function(current_chunk) >= self.min_chunk_size:
                    chunks.append(current_chunk)
                
                # If paragraph itself is too large, split it recursively
                if self.length_function(para) > self.chunk_size:
                    # Recursively split the large paragraph
                    para_chunks = self._recursive_split(para, self.separators[1:], self.chunk_size)
                    para_chunks = self._merge_splits(para_chunks)
                    chunks.extend(para_chunks)
                else:
                    current_chunk = para
        
        # Don't forget the last chunk
        if current_chunk and self.length_function(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk)
        
        # Apply post-processing
        chunks = self._post_process_chunks(chunks)
        
        return chunks

    def _recursive_split(self, text: str, separators: List[str], chunk_size: int) -> List[str]:
        """Recursively split text."""
        if self.length_function(text) <= chunk_size or not separators:
            return [text]

        separator = separators[0]
        splits = []
        
        if self.is_separator_regex:
            # Treat separators as regex patterns
            splits = re.split(separator, text)
        else:
            # Escape separator for literal matching
            pattern = re.escape(separator) if separator else '.'
            splits = re.split(pattern, text)
        
        # Process each split
        result_splits = []
        for split in splits:
            if self.length_function(split) > chunk_size:
                # Recursively process oversized splits with remaining separators
                result_splits.extend(
                    self._recursive_split(split, separators[1:], chunk_size)
                )
            elif self.length_function(split) >= self.min_chunk_size or split == "":
                # Only keep splits that meet min size requirement (except empty strings for spacing)
                result_splits.append(split)
        
        return result_splits

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """Merge splits and handle overlap."""
        if not splits:
            return []
        
        chunks = []
        current_chunk = ""
        
        for split in splits:
            # Determine if we should add a separator
            separator = self.separators[0] if self.separators else ""
            test_chunk = current_chunk + separator + split if current_chunk and separator else (current_chunk + split if current_chunk else split)
            
            if self.length_function(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                # Current chunk is full, save it
                if current_chunk:
                    chunks.append(current_chunk)
                
                # For the new chunk, decide strategy based on chunk_strategy
                if self.chunk_strategy == "aggressive":
                    # Try to fit as much as possible
                    current_chunk = split
                elif self.chunk_strategy == "conservative":
                    # Be more restrictive
                    if self.length_function(split) > self.chunk_size:
                        # If split is too big, we need to handle it differently
                        current_chunk = split[:self.chunk_size]
                        # Add remainder back to splits for processing
                        remainder = split[self.chunk_size:]
                        if remainder:
                            splits.insert(splits.index(split) + 1, remainder)
                    else:
                        current_chunk = split
                else:  # balanced
                    # Standard behavior
                    if self.length_function(split) > self.chunk_size:
                        # Split is too big, need to handle recursively
                        temp_splits = self._recursive_split(split, self.separators[1:], self.chunk_size)
                        if temp_splits and current_chunk:
                            # Add first part to current chunk
                            first_part = temp_splits[0]
                            if self.length_function(current_chunk + first_part) <= self.chunk_size:
                                current_chunk += first_part
                                temp_splits = temp_splits[1:]  # Remove first part
                        
                        # Add remaining parts as new chunks
                        for temp_split in temp_splits:
                            if self.length_function(temp_split) > 0:
                                chunks.append(temp_split)
                    else:
                        current_chunk = split
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        # Add overlap between chunks
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._add_overlap(chunks)
        
        return chunks

    def _add_overlap(self, chunks: List[str]) -> List[str]:
        """Add overlap between chunks."""
        if not chunks or self.chunk_overlap <= 0:
            return chunks
        
        result = [chunks[0]]
        
        for i in range(1, len(chunks)):
            # Get the overlap from the previous chunk
            prev_chunk = chunks[i-1]
            overlap = self._get_overlap_text(prev_chunk, self.chunk_overlap)
            
            # Add overlap to current chunk
            new_chunk = overlap + chunks[i]
            result.append(new_chunk)
        
        return result

    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get overlap text from the end of a chunk."""
        if overlap_size <= 0:
            return ""
        
        # Try to find a good break point within the overlap region
        # Prefer word boundaries or sentence endings
        text_len = self.length_function(text)
        start_pos = max(0, text_len - overlap_size)
        
        # Extract a portion to look for good breaks
        candidate = text[start_pos:]
        
        # Look for good break points in reverse order
        break_points = [' ', '.', '!', '?', ',', ';', '\n', '\t']
        for bp in break_points:
            idx = candidate.rfind(bp)
            if idx != -1:
                # Found a good break point
                return text[start_pos + idx + 1:]  # Return from after the break point
        
        # If no good break point found, return the full overlap
        return text[start_pos:]

    def _post_process_chunks(self, chunks: List[str]) -> List[str]:
        """Apply post-processing to chunks."""
        processed = []
        
        for chunk in chunks:
            # Strip separators if requested
            if self.strip_separators:
                chunk = chunk.strip()
            
            # Skip chunks that are too small
            if self.length_function(chunk) < self.min_chunk_size:
                continue
            
            # Add header and footer if requested
            processed_chunk = self.chunk_header_prefix + chunk + self.chunk_header_suffix
            
            processed.append(processed_chunk)
        
        # Merge small chunks if enabled
        if self.merge_small_chunks:
            processed = self._merge_small_chunks(processed)
        
        return processed

    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """Merge small chunks with their neighbors."""
        if len(chunks) <= 1 or self.min_chunk_size <= 0:
            return chunks
        
        result = []
        i = 0
        
        while i < len(chunks):
            current = chunks[i]
            
            # Check if current chunk is too small
            if self.length_function(current) < self.min_chunk_size and i + 1 < len(chunks):
                # Merge with next chunk
                next_chunk = chunks[i + 1]
                merged = current + self.separators[0] + next_chunk if self.separators else current + next_chunk
                result.append(merged)
                i += 2  # Skip both chunks
            else:
                result.append(current)
                i += 1
        
        return result

    def get_statistics(self, text: str, chunks: List[str]) -> Dict:
        """Get statistics about the chunking process."""
        original_length = self.length_function(text)
        num_chunks = len(chunks)
        
        if not chunks:
            return {
                'original_length': original_length,
                'num_chunks': 0,
                'avg_chunk_size': 0,
                'max_chunk_size': 0,
                'min_chunk_size': 0,
                'total_output_length': 0,
                'compression_ratio': 0,
                'overlap_characters': 0
            }
        
        chunk_lengths = [self.length_function(chunk) for chunk in chunks]
        total_output_length = sum(chunk_lengths)
        
        # Calculate overlap by checking repeated content
        overlap_chars = 0
        for i in range(1, len(chunks)):
            current = chunks[i]
            prev = chunks[i-1]
            # Simple heuristic: count overlap as common prefix/suffix
            # More sophisticated overlap detection could go here
            overlap_chars += min(self.length_function(prev), self.chunk_overlap)
        
        return {
            'original_length': original_length,
            'num_chunks': num_chunks,
            'avg_chunk_size': sum(chunk_lengths) / len(chunk_lengths),
            'max_chunk_size': max(chunk_lengths),
            'min_chunk_size': min(chunk_lengths),
            'total_output_length': total_output_length,
            'compression_ratio': total_output_length / original_length if original_length > 0 else 0,
            'overlap_characters': overlap_chars
        }


def create_default_enhanced_recursive_splitter(chunk_size: int = 200) -> EnhancedRecursiveCharacterTextSplitter:
    """Create a default enhanced recursive splitter with reasonable parameters."""
    return EnhancedRecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_size // 5,  # 20% overlap
        separators=["\n\n", "\n", "。", "？", "！", "，", ".", "?", "!", ",", " ", ""],
        min_chunk_size=10,
        chunk_strategy="balanced"
    )