import re
from typing import List, Any, Tuple, Dict


class PythonCodeTextSplitter:
    """Attempts to split the text along Python-formatted layout elements."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 20,
        keep_separator: bool = True,
        is_separator_regex: bool = True,  # 设置为 True
        **kwargs: Any,
    ) -> None:
        """Initialize a PythonTextSplitter with Python-specific separators."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.keep_separator = keep_separator
        self.is_separator_regex = is_separator_regex
        self.separators = [
            # First, try to split along class definitions
            "\nclass ",
            "\ndef ",
            "\n\tdef ",
            # Now split by the normal type of lines
            "\n\n",
            "\n",
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
                new_separators = self.separators[i + 1 :]
                break

        _separator = separator if self.is_separator_regex else re.escape(separator)
        splits = re.split(_separator, text)

        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = "" if self.keep_separator else separator

        for s in splits:
            if not isinstance(s, str):
                # 如果 s 不是字符串，跳过或处理
                continue

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
        if not isinstance(text, str):
            raise ValueError("输入的文本必须是字符串类型。")
        return self._split_text(text)

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge split chunks based on separator."""
        # 过滤掉非字符串元素，确保 join 不会出错
        splits = [s for s in splits if isinstance(s, str)]
        return [separator.join(splits)]


if __name__ == "__main__":
    text = """
    def add_numbers(a, b):
        \"\"\"
        计算两个数的和。
        
        参数:
        a (int or float): 第一个数字
        b (int or float): 第二个数字
        
        返回:
        int or float: 两个数字的和
        \"\"\"
        return a + b

    def subtract_numbers(a, b):
        \"\"\"
        计算两个数的差。
        
        参数:
        a (int or float): 被减数
        b (int or float): 减数
        
        返回:
        int or float: 两个数字的差
        \"\"\"
        return a - b
    """
    splitter = PythonCodeTextSplitter(chunk_size=10)
    chunks = splitter.split_text(text)

    for chunk in chunks:
        print(f"Content:\n{chunk}\n")
