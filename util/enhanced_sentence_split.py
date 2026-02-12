import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/" + ".."))

import re
from typing import List, Optional, Dict, Tuple, Pattern, Union
from util.utils import calculate_custom_length
import logging


class EnhancedGeneralTextSplitter:
    """An enhanced text splitter with improved heuristics and customization options."""

    def __init__(
        self,
        sentence_endings: Optional[str] = None,
        max_sentence_length: int = 100,
        is_pdf: bool = False,
        custom_sentence_patterns: Optional[List[str]] = None,  # Additional sentence boundary patterns
        min_sentence_length: int = 5,  # Minimum sentence length to keep
        split_long_sentences: bool = True,  # Whether to split long sentences
        preserve_structure: bool = False,  # Whether to preserve document structure info
        language: str = "mixed",  # Language: 'zh', 'en', 'mixed'
        enable_hybrid_splitting: bool = True,  # Enable hybrid splitting logic
        max_chunk_length: int = 512,  # Max length for chunks in batch_chunk
        overlap_strategy: str = "sentence_boundary",  # How to handle overlaps: 'sentence_boundary', 'character_based'
        sentence_boundary_penalty: float = 0.1,  # Penalty for splitting mid-sentence
        paragraph_marker: str = "\n\n",  # Paragraph separation marker
    ):
        """Initialize enhanced splitter.
        
        Args:
            sentence_endings: Custom sentence ending punctuation
            max_sentence_length: Maximum length for a sentence
            is_pdf: Whether text is from PDF
            custom_sentence_patterns: Additional regex patterns for sentence boundaries
            min_sentence_length: Minimum sentence length to keep
            split_long_sentences: Whether to split long sentences
            preserve_structure: Whether to preserve document structure
            language: Text language ('zh', 'en', 'mixed')
            enable_hybrid_splitting: Enable advanced splitting heuristics
            max_chunk_length: Maximum length for chunks in batch_chunk
            overlap_strategy: Strategy for handling overlap ('sentence_boundary', 'character_based')
            sentence_boundary_penalty: Penalty for splitting mid-sentence
            paragraph_marker: Marker for paragraphs
        """
        self.default_endings = "。?？!！;；…"  # Default Chinese sentence endings
        self.en_endings = ".?!;"  # English sentence endings
        self.sentence_endings = sentence_endings or self.default_endings
        
        # Adjust for language
        if language == "en":
            self.sentence_endings = self.en_endings
        elif language == "mixed":
            self.sentence_endings = self.default_endings + self.en_endings
            
        self.max_sentence_length = max_sentence_length
        self.min_sentence_length = min_sentence_length
        self.is_pdf = is_pdf
        self.custom_sentence_patterns = custom_sentence_patterns or []
        self.split_long_sentences = split_long_sentences
        self.preserve_structure = preserve_structure
        self.language = language
        self.enable_hybrid_splitting = enable_hybrid_splitting
        self.max_chunk_length = max_chunk_length
        self.overlap_strategy = overlap_strategy
        self.sentence_boundary_penalty = sentence_boundary_penalty
        self.paragraph_marker = paragraph_marker
        
        # Compile sentence splitting pattern
        self.sentence_split_pattern = self._compile_sentence_split_pattern()
        self.custom_patterns = [re.compile(p) for p in self.custom_sentence_patterns]
        self._initialize_regex_patterns()

    def _compile_sentence_split_pattern(self) -> Pattern:
        """Dynamic sentence splitting pattern with language awareness."""
        if self.language == "en":
            # English-specific pattern
            pattern = rf"(?<=[{re.escape(self.sentence_endings)}])\s+(?=[A-Z])"
        elif self.language == "zh":
            # Chinese-specific pattern
            pattern = rf"(?<=[{re.escape(self.sentence_endings)}])\s*(?=[\u4e00-\u9fffA-Z])"
        else:
            # Mixed language pattern
            pattern = rf"(?<=[{re.escape(self.sentence_endings)}]|(?<!\b[A-Za-z])\.(?![A-Za-z]\b))\s*(?=\s*[\u4e00-\u9fa5A-Za-z0-9\(\[\"'])"
        
        return re.compile(pattern)

    def _initialize_regex_patterns(self) -> None:
        """Initialize additional regex patterns."""
        self.newline_excess_re = re.compile(r"\n{3,}")  # Multiple newlines to one
        self.whitespace_re = re.compile(r"\s+")  # Multiple whitespace to one
        self.double_newline_re = re.compile(r"\n\n")  # Double newline pattern
        self.sentence_end_re = re.compile(r"([。！？!?])([^”’])")  # Sentence ends
        self.ellipsis_en_re = re.compile(r"(\.{6})([^”’])")  # Long ellipsis
        self.ellipsis_cn_re = re.compile(r"(…{2})([^”’])")  # Chinese ellipsis
        self.quote_sentence_end_re = re.compile(r"([。！？!?][”’])([^，。！？!?])")  # Quotes and sentence ends
        self.comma_semicolon_re = re.compile(r"([，；,;])")  # Comma and semicolon
        self.acronym_re = re.compile(r"\b(?:[A-Z]\.)*[A-Z]\.")  # Acronyms like U.S.A.

    def split_text(self, text: str) -> List[str]:
        """Split text with enhanced logic."""
        if not text:
            return []

        # Preprocess text
        if self.is_pdf:
            text = self._preprocess_pdf_text(text)

        # Split by sentences
        initial_segments = self._split_by_sentences(text)
        
        # Filter by minimum length and apply other filters
        filtered_segments = [
            seg for seg in initial_segments 
            if len(seg.strip()) >= self.min_sentence_length
        ]
        
        # Further split long segments
        if self.split_long_sentences:
            final_segments = self._split_long_segments(filtered_segments)
        else:
            final_segments = filtered_segments

        return final_segments

    def batch_chunk(
        self,
        text_list: Union[str, List[str]],
        max_length: int = None,
        overlap_size: int = None,
        return_counts: bool = True,
        chunk_strategy: str = "balanced",  # 'balanced', 'prefer_boundaries', 'strict_size'
    ) -> Union[Tuple[List[List[str]], List[int]], List[List[str]]]:
        """
        Enhanced batch chunking with multiple strategies.
        """
        # Use defaults if not provided
        max_length = max_length or self.max_chunk_length
        overlap_size = overlap_size or (max_length // 10)  # 10% overlap by default
        
        chunk_list = []
        cumulative_counts = [0]

        if isinstance(text_list, str):
            text_list = [text_list]
        elif not isinstance(text_list, list):
            raise TypeError(f"text_list must be str or List[str], received {type(text_list)}")

        for text in text_list:
            # Split into sentences
            sent_list = self.split_text(text)
            if not sent_list:
                chunk_list.append([])
                cumulative_counts.append(cumulative_counts[-1])
                continue

            # Precompute sentence lengths
            sent_lengths = [calculate_custom_length(sentence) for sentence in sent_list]

            current_chunk = []
            current_length = 0
            chunks = []
            overlap_elements = []  # Store overlap elements based on strategy
            overlap_length = 0

            for i, sentence in enumerate(sent_list):
                sentence_length = sent_lengths[i]

                # Handle very long sentences
                if sentence_length > max_length:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))
                        current_chunk = []
                        current_length = 0
                    
                    # Process long sentence based on strategy
                    if chunk_strategy == "prefer_boundaries":
                        # Split the long sentence intelligently
                        sentence_chunks = self._split_long_sentence(sentence, max_length)
                        for chunk in sentence_chunks[:-1]:
                            chunks.append(chunk)
                        
                        # Handle the last piece of the split sentence
                        last_piece = sentence_chunks[-1]
                        last_piece_len = calculate_custom_length(last_piece)
                        
                        if current_length + last_piece_len <= max_length:
                            current_chunk.append(last_piece)
                            current_length += last_piece_len
                        else:
                            if current_chunk:
                                chunks.append(" ".join(current_chunk))
                            chunks.append(last_piece)
                            current_chunk = []
                            current_length = 0
                    else:
                        # Add as standalone chunk
                        chunks.append(sentence)
                        # Update overlap for next chunk
                        overlap_elements = [sentence]
                        overlap_length = sentence_length
                    continue

                # Determine if we need a new chunk
                if current_length + sentence_length > max_length:
                    if current_chunk:
                        chunks.append(" ".join(current_chunk))

                    # Calculate overlap based on strategy
                    if self.overlap_strategy == "sentence_boundary":
                        overlap_elements, overlap_length = self._calculate_sentence_overlap(
                            current_chunk, sent_lengths[:i], overlap_size
                        )
                    else:  # character_based
                        overlap_elements, overlap_length = self._calculate_character_overlap(
                            chunks[-1] if chunks else "", overlap_size
                        )

                    # Start new chunk with overlap
                    current_chunk = overlap_elements[:]
                    current_length = overlap_length

                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_length += sentence_length

            # Add final chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))

            # Post-process chunks
            chunks = self._post_process_chunks(chunks, chunk_strategy)

            # Update counts
            chunk_list.append(chunks)
            cumulative_counts.append(cumulative_counts[-1] + len(chunks))

        if return_counts:
            return chunk_list, cumulative_counts[1:]
        else:
            return chunk_list

    def _split_long_sentence(self, sentence: str, max_length: int) -> List[str]:
        """Split a long sentence into smaller pieces while respecting boundaries."""
        if calculate_custom_length(sentence) <= max_length:
            return [sentence]
        
        # Try to split by clauses or phrases
        sub_sentences = re.split(r'[，,、]', sentence)
        result = []
        current_sub = ""
        
        for sub in sub_sentences:
            test_chunk = current_sub + ("，" if current_sub else "") + sub
            if calculate_custom_length(test_chunk) <= max_length:
                current_sub = test_chunk
            else:
                if current_sub:
                    result.append(current_sub)
                current_sub = sub
        
        if current_sub:
            result.append(current_sub)
        
        # If still too long, split by words
        if result and calculate_custom_length(result[-1]) > max_length:
            final_result = []
            for chunk in result:
                if calculate_custom_length(chunk) <= max_length:
                    final_result.append(chunk)
                else:
                    # Split by words
                    words = chunk.split()
                    temp_chunk = ""
                    for word in words:
                        test_temp = temp_chunk + (" " if temp_chunk else "") + word
                        if calculate_custom_length(test_temp) <= max_length:
                            temp_chunk = test_temp
                        else:
                            if temp_chunk:
                                final_result.append(temp_chunk)
                            temp_chunk = word
                    if temp_chunk:
                        final_result.append(temp_chunk)
            
            result = final_result
        
        return result

    def _calculate_sentence_overlap(self, chunk_sentences: List[str], sentence_lengths: List[int], overlap_size: int) -> Tuple[List[str], int]:
        """Calculate overlap based on sentences."""
        if not chunk_sentences:
            return [], 0
        
        overlap_sentences = []
        overlap_length = 0
        
        # Go backwards through sentences to accumulate overlap
        for sent, sent_len in zip(reversed(chunk_sentences), reversed(sentence_lengths[-len(chunk_sentences):])):
            if overlap_length + sent_len > overlap_size and overlap_sentences:
                break
            overlap_sentences.insert(0, sent)
            overlap_length += sent_len
        
        # Ensure at least one sentence if possible
        if not overlap_sentences and chunk_sentences:
            last_sentence = chunk_sentences[-1]
            overlap_sentences = [last_sentence]
            overlap_length = calculate_custom_length(last_sentence)
        
        return overlap_sentences, overlap_length

    def _calculate_character_overlap(self, chunk_text: str, overlap_size: int) -> Tuple[List[str], int]:
        """Calculate overlap based on character count."""
        if not chunk_text:
            return [], 0
        
        overlap_text = chunk_text[-overlap_size:]
        return [overlap_text], len(overlap_text)

    def _post_process_chunks(self, chunks: List[str], strategy: str) -> List[str]:
        """Post-process chunks based on strategy."""
        if strategy == "balanced":
            # Default behavior - keep as is
            return chunks
        elif strategy == "prefer_boundaries":
            # Ensure chunks end at meaningful boundaries
            processed = []
            for chunk in chunks:
                if len(chunk) < 16:  # Too short
                    if processed:
                        # Merge with previous chunk
                        processed[-1] += " " + chunk
                    else:
                        processed.append(chunk)
                else:
                    processed.append(chunk)
            return processed
        else:  # strict_size
            # Enforce strict size limits
            processed = []
            for chunk in chunks:
                if calculate_custom_length(chunk) > self.max_chunk_length:
                    # Split oversized chunks
                    sub_chunks = self._split_oversized_chunk(chunk)
                    processed.extend(sub_chunks)
                else:
                    processed.append(chunk)
            return processed

    def _split_oversized_chunk(self, chunk: str) -> List[str]:
        """Split an oversized chunk."""
        if calculate_custom_length(chunk) <= self.max_chunk_length:
            return [chunk]
        
        # Split by sentences first if possible
        sentences = self.split_text(chunk)
        if len(sentences) > 1:
            result = []
            current_sub = []
            current_len = 0
            
            for sent in sentences:
                sent_len = calculate_custom_length(sent)
                if current_len + sent_len > self.max_chunk_length and current_sub:
                    result.append(" ".join(current_sub))
                    current_sub = [sent]
                    current_len = sent_len
                else:
                    current_sub.append(sent)
                    current_len += sent_len + 1  # +1 for space
            
            if current_sub:
                result.append(" ".join(current_sub))
            
            return result
        else:
            # Split by words if only one sentence
            words = chunk.split()
            result = []
            current_sub = []
            current_len = 0
            
            for word in words:
                word_len = len(word)
                if current_len + word_len > self.max_chunk_length and current_sub:
                    result.append(" ".join(current_sub))
                    current_sub = [word]
                    current_len = word_len
                else:
                    current_sub.append(word)
                    current_len += word_len + 1  # +1 for space
            
            if current_sub:
                result.append(" ".join(current_sub))
            
            return result

    def _preprocess_pdf_text(self, text: str) -> str:
        """Preprocess PDF text."""
        text = self.newline_excess_re.sub("\n", text)
        text = self.whitespace_re.sub(" ", text)
        text = self.double_newline_re.sub("\n", text)
        return text

    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Apply custom patterns first if provided
        for pattern in self.custom_patterns:
            text = pattern.sub(r'\g<0>\n', text)  # Add newline after matches
        
        # Apply main sentence splitting
        sentences = self.sentence_split_pattern.split(text)
        
        # Clean up and filter
        result = []
        for sentence in sentences:
            clean_sent = sentence.strip()
            if clean_sent:
                result.append(clean_sent)
        
        return result

    def _split_long_segments(self, segments: List[str]) -> List[str]:
        """Further split segments that exceed max length."""
        final_segments = []
        for segment in segments:
            if len(segment) <= self.max_sentence_length:
                final_segments.append(segment)
            else:
                # Split by commas/semicolons first
                sub_segments = self._split_by_commas(segment)
                for sub_segment in sub_segments:
                    if len(sub_segment) <= self.max_sentence_length:
                        final_segments.append(sub_segment)
                    else:
                        # If still too long, split by spaces
                        final_segments.extend(self._split_by_spaces(sub_segment))
        
        return final_segments

    def _split_by_commas(self, text: str) -> List[str]:
        """Split by commas and semicolons."""
        text = self.comma_semicolon_re.sub(r'\1\n', text)
        return [segment.strip() for segment in text.split('\n') if segment.strip()]

    def _split_by_spaces(self, text: str) -> List[str]:
        """Split by spaces."""
        words = text.split()
        segments = []
        current_segment = ""
        
        for word in words:
            if len(current_segment) + len(word) + 1 <= self.max_sentence_length:
                current_segment += (' ' if current_segment else '') + word
            else:
                if current_segment:
                    segments.append(current_segment)
                current_segment = word
        
        if current_segment:
            segments.append(current_segment)
        
        return segments

    def get_statistics(self, text: str) -> Dict:
        """Get statistics about text structure."""
        sentences = self.split_text(text)
        lengths = [len(s) for s in sentences]
        
        if not sentences:
            return {
                'total_sentences': 0,
                'avg_sentence_length': 0,
                'max_sentence_length': 0,
                'min_sentence_length': 0,
                'total_text_length': len(text)
            }
        
        return {
            'total_sentences': len(sentences),
            'avg_sentence_length': sum(lengths) / len(lengths),
            'max_sentence_length': max(lengths),
            'min_sentence_length': min(lengths),
            'total_text_length': len(text),
            'length_std_dev': (sum((x - sum(lengths)/len(lengths))**2 for x in lengths) / len(lengths))**0.5 if lengths else 0
        }


def create_default_enhanced_splitter(language: str = "mixed") -> EnhancedGeneralTextSplitter:
    """Create a default enhanced splitter."""
    return EnhancedGeneralTextSplitter(
        language=language,
        max_sentence_length=150,
        min_sentence_length=5,
        split_long_sentences=True,
        enable_hybrid_splitting=True
    )