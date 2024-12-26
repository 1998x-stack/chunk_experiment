import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/" + ".."))

import re
from typing import List, Optional

from util.sentence_split import GeneralTextSplitter

class RecursiveCharacterTextSplitter:
    """递归字符文本分割器。

    此类用于将长文本递归地分割为较小的块，以确保每个块的长度
    不超过指定的chunk_size，并且块之间可以有一定的重叠。
    """

    def __init__(
        self,
        chunk_size: int = 200,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None,
    ):
        """初始化分割器。

        Args:
            chunk_size: 每个块的最大字符数。
            chunk_overlap: 相邻块之间的重叠字符数。
            separators: 分隔符列表，按优先级从高到低排列。
        """
        if separators is None:
            separators = ["\n\n", "\n", "。", "？", "！", "，", " ", ""]
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators
        self.general_splitter = GeneralTextSplitter()

    def split_text(self, text: str, sep_type='sentence') -> List[str]:
        """分割文本为较小的块。

        Args:
            text: 待分割的长文本。

        Returns:
            分割后的文本块列表。
        """
        if sep_type == 'sentence':
            chunks = self.general_splitter.batch_chunk(text, max_length=self.chunk_size, overlap_size=self.chunk_overlap, return_counts=False)
        # 开始递归分割
        splits = self._recursive_split(text, self.separators, self.chunk_size)
        # 合并分割后的块，处理重叠
        chunks = self._merge_splits(splits)
        return chunks

    def _recursive_split(
        self, text: str, separators: List[str], chunk_size: int
    ) -> List[str]:
        """递归地分割文本。

        Args:
            text: 当前待分割的文本。
            separators: 当前使用的分隔符列表。
            chunk_size: 当前块的最大字符数。

        Returns:
            分割后的文本片段列表。
        """
        if len(text) <= chunk_size or not separators:
            # 无需进一步分割，或无更多分隔符
            return [text]

        separator = separators[0]
        if separator:
            # 使用当前分隔符进行分割
            pattern = re.escape(separator)
            splits = re.split(pattern, text)
        else:
            # 无分隔符，按字符切割
            splits = list(text)

        # 处理分割后的每个部分
        result_splits = []
        for split in splits:
            if len(split) > chunk_size:
                # 递归处理过大的片段
                result_splits.extend(
                    self._recursive_split(split, separators[1:], chunk_size)
                )
            else:
                result_splits.append(split)
        return result_splits

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """合并分割后的片段，处理重叠。

        Args:
            splits: 分割后的文本片段列表。

        Returns:
            处理重叠后的文本块列表。
        """
        chunks = []
        current_chunk = ""
        for split in splits:
            if (
                len(current_chunk) + len(split) + len(self.separators[0])
                <= self.chunk_size
            ):
                # 可以继续添加到当前块
                if current_chunk:
                    current_chunk += self.separators[0] + split
                else:
                    current_chunk = split
            else:
                # 保存当前块，开始新块
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = split
        if current_chunk:
            chunks.append(current_chunk)

        # 添加重叠部分
        if self.chunk_overlap > 0 and len(chunks) > 1:
            for i in range(1, len(chunks)):
                overlap = chunks[i - 1][-self.chunk_overlap :]
                chunks[i] = overlap + chunks[i]
        return chunks


if __name__ == "__main__":
    # 示例文本
    long_text = (
        "这是一个用于测试的长文本。它包含多个句子，用于演示递归文本分割器的功能。\n"
        "我们希望将这个文本分割成较小的块，每个块的长度不超过指定的字符数。\n\n"
        "分割器应能智能地处理中文标点和换行符。"
    )

    # 初始化分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=10,
        separators=["\n\n", "\n", "。", "？", "！", "，", " ", ""],
    )

    # 执行分割
    chunks = text_splitter.split_text(long_text)

    # 输出结果
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i + 1}:\n{chunk}\n")
