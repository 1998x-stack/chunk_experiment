from typing import List, Optional, Any
import re

class LatexTextSplitter:
    """Attempts to split the text along Latex-formatted layout elements."""
    
    def __init__(self, chunk_size: int = 1000,chunk_overlap=20, keep_separator: bool = True, is_separator_regex: bool = False, **kwargs: Any) -> None:
        """Initialize a LatexTextSplitter with LaTeX-specific separators."""
        self.chunk_size = chunk_size
        self.chunk_overlap=chunk_overlap
        self.keep_separator = keep_separator
        self.is_separator_regex = is_separator_regex
        self.separators = [
            "\n\\\\chapter{",
            "\n\\\\section{",
            "\n\\\\subsection{",
            "\n\\\\subsubsection{",
            "\n\\\\begin{enumerate}",
            "\n\\\\begin{itemize}",
            "\n\\\\begin{description}",
            "\n\\\\begin{list}",
            "\n\\\\begin{quote}",
            "\n\\\\begin{quotation}",
            "\n\\\\begin{verse}",
            "\n\\\\begin{verbatim}",
            "\n\\\begin{align}",
            "$$",
            "$",
            " ",
            "",
        ]
    
    def _split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        
        separator = self.separators[-1]  # Default to splitting by empty string
        new_separators = []
        
        # Search for an appropriate separator to use
        for i, _s in enumerate(self.separators):
            _separator = _s if self.is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = self.separators[i + 1:]
                break
        
        _separator = separator if self.is_separator_regex else re.escape(separator)
        splits = re.split(_separator, text)
        
        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = "" if self.keep_separator else separator
        
        for s in splits:
            if len(s) < self.chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                final_chunks.append(s)
        
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        
        return final_chunks

    def split_text(self, text: str) -> List[str]:
        """Public method to split text."""
        return self._split_text(text)

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge split chunks based on separator."""
        return [separator.join(splits)]
    
if __name__ == "__main__":
    text = """\\chapter{Introduction}
    This is an introduction to the paper.

    \\section{Methodology}
    This section covers the methods used.

    \\subsection{Data Collection}
    Here we describe how the data was collected.

    \\begin{enumerate}
    \\item First step
    \\item Second step
    \\end{enumerate}
    """

    splitter = LatexTextSplitter(chunk_size=20)
    chunks = splitter.split_text(text)

    for chunk in chunks:
        print(f"Content:\n{chunk}\n")