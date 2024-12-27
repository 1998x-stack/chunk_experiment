import re
from typing import List, Any, Tuple, Dict


class MarkdownTextSplitter:
    """Attempts to split the text along Markdown-formatted layout elements."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 20,
        keep_separator: bool = True,
        is_separator_regex: bool = True,  # 设置为 True
        **kwargs: Any,
    ) -> None:
        """Initialize a MarkdownTextSplitter with Markdown-specific separators."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.keep_separator = keep_separator
        self.is_separator_regex = is_separator_regex
        self.separators = [
            # 正则表达式模式，用于匹配 Markdown 标题
            r"\n#{1,6} ",
            # 代码块结束符
            "```\n",
            # 水平分隔线
            r"\n\*{3,}\n",
            r"\n-{3,}\n",
            r"\n_{3,}\n",
            # 两个换行符
            "\n\n",
            # 单个换行符
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


class MarkdownHeaderSplitter:
    """Splitting markdown files based on specified headers."""

    def __init__(
        self,
        headers_to_split_on: List[Tuple[str, str]],
        return_each_line: bool = False,
        strip_headers: bool = True,
    ):
        """Create a new MarkdownHeaderSplitter.

        Args:
            headers_to_split_on: Headers we want to track (e.g., "#, ##, etc")
            return_each_line: Return each line w/ associated headers
            strip_headers: Strip split headers from the content of the chunk
        """
        self.return_each_line = return_each_line
        # Sort headers by length (longer headers first)
        self.headers_to_split_on = sorted(
            headers_to_split_on, key=lambda split: len(split[0]), reverse=True
        )
        self.strip_headers = strip_headers

    def aggregate_lines_to_chunks(self, lines: List[Dict]) -> List[Dict]:
        """Combine lines with common metadata into chunks."""
        aggregated_chunks = []

        for line in lines:
            if (
                aggregated_chunks
                and aggregated_chunks[-1]["metadata"] == line["metadata"]
            ):
                # If metadata matches, append the current content to the last line
                aggregated_chunks[-1]["content"] += "\n" + line["content"]
            else:
                # Otherwise, start a new chunk
                aggregated_chunks.append(line)

        return aggregated_chunks

    def split_text(self, text: str) -> List[Dict]:
        """Split markdown file by headers.

        Args:
            text: Markdown file content as a string.
        """
        lines = text.split("\n")
        lines_with_metadata = []
        current_content = []
        current_metadata = {}
        header_stack = []
        initial_metadata = {}

        in_code_block = False
        opening_fence = ""

        for line in lines:
            stripped_line = line.strip()
            stripped_line = "".join(
                filter(str.isprintable, stripped_line)
            )  # Clean the line

            if not in_code_block:
                if stripped_line.startswith("```") or stripped_line.startswith("~~~"):
                    # Handle code blocks
                    in_code_block = True
                    opening_fence = stripped_line[:3]
                else:
                    # Check for headers to split on
                    for sep, name in self.headers_to_split_on:
                        if stripped_line.startswith(sep):
                            # Determine if it's a valid header
                            if (
                                len(stripped_line) == len(sep)
                                or stripped_line[len(sep)] == " "
                            ):
                                if name:
                                    current_header_level = sep.count("#")

                                    # Pop headers of equal or higher level from stack
                                    while (
                                        header_stack
                                        and header_stack[-1]["level"]
                                        >= current_header_level
                                    ):
                                        popped_header = header_stack.pop()
                                        if popped_header["name"] in initial_metadata:
                                            initial_metadata.pop(popped_header["name"])

                                    # Add the current header to stack
                                    header = {
                                        "level": current_header_level,
                                        "name": name,
                                        "data": stripped_line[len(sep) :].strip(),
                                    }
                                    header_stack.append(header)

                                    # Update initial metadata
                                    initial_metadata[name] = header["data"]

                                # If current_content is not empty, add it to the lines with metadata
                                if current_content:
                                    lines_with_metadata.append(
                                        {
                                            "content": "\n".join(current_content),
                                            "metadata": current_metadata.copy(),
                                        }
                                    )
                                    current_content.clear()

                                if not self.strip_headers:
                                    current_content.append(stripped_line)

                                break
                    else:
                        # If not a header, continue accumulating content
                        if stripped_line:
                            current_content.append(stripped_line)
                        elif current_content:
                            # If a blank line is encountered, add the content to the chunks
                            lines_with_metadata.append(
                                {
                                    "content": "\n".join(current_content),
                                    "metadata": current_metadata.copy(),
                                }
                            )
                            current_content.clear()

            else:
                if stripped_line.startswith(opening_fence):
                    # End of code block
                    in_code_block = False
                    opening_fence = ""
                current_content.append(stripped_line)

            current_metadata = initial_metadata.copy()

        # Add any remaining content
        if current_content:
            lines_with_metadata.append(
                {"content": "\n".join(current_content), "metadata": current_metadata}
            )

        # If we want to return each line with its metadata
        if self.return_each_line:
            return [
                {"content": chunk["content"], "metadata": chunk["metadata"]}
                for chunk in lines_with_metadata
            ]
        else:
            return self.aggregate_lines_to_chunks(lines_with_metadata)


if __name__ == "__main__":
    text = """
    # Title 1
    This is the first section of the document.

    ## Title 2
    This is a subsection.

    Some more text.

    ---

    ### Title 3
    This is another section with some content.

    ```python
    # Code block example
    print("Hello, world!")
    """
    headers_to_split_on = [
        ("#", "Main Title"),
        ("##", "Sub Title"),
        ("###", "Sub Sub Title"),
    ]
    splitter = MarkdownTextSplitter(chunk_size=10, is_separator_regex=True)
    chunks = splitter.split_text(text)

    for chunk in chunks:
        print(f"Content:\n{chunk}\n")
