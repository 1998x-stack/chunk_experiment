import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))

import re
from typing import List, Optional

from util.utils import calculate_custom_length


class GeneralTextSplitter:
    """一个综合的中英混合文本句子分割器，支持动态配置和多层次分割。"""

    def __init__(self, sentence_endings: Optional[str] = None, max_sentence_length: int = 100, is_pdf: bool = False):
        """初始化分割器。

        Args:
            sentence_endings (Optional[str]): 自定义的句末标点符号。如果为空，则使用默认标点。
            max_sentence_length (int): 每个句子允许的最大长度。
            is_pdf (bool): 是否对 PDF 文本进行预处理。
        """
        self.default_endings = "。?？!！;；…"  # 默认句末标点符号
        self.sentence_endings = sentence_endings or self.default_endings
        self.max_sentence_length = max_sentence_length
        self.is_pdf = is_pdf
        self.sentence_split_pattern = self._compile_sentence_split_pattern()
        self._initialize_regex_patterns()

    def _compile_sentence_split_pattern(self) -> re.Pattern:
        """动态生成句子分割的正则表达式。

        Returns:
            re.Pattern: 编译后的正则表达式。
        """
        pattern = rf"(?<=[{re.escape(self.sentence_endings)}]|(?<!\b[A-Za-z])\.(?![A-Za-z]\b))\s*(?=\s*[\u4e00-\u9fa5A-Za-z0-9\(\[\"'])"
        return re.compile(pattern)

    def _initialize_regex_patterns(self) -> None:
        """初始化额外的正则表达式，用于长段落分割和文本预处理。"""
        self.newline_excess_re = re.compile(r"\n{3,}")  # 合并多余的换行符
        self.whitespace_re = re.compile(r"\s+")  # 替换多余空格
        self.double_newline_re = re.compile(r"\n\n")  # 去除双重换行符
        self.sentence_end_re = re.compile(r"([。！？!?])([^”’])")  # 普通句子结束符
        self.ellipsis_en_re = re.compile(r"(\.{6})([^”’])")  # 英文省略号
        self.ellipsis_cn_re = re.compile(r"(…{2})([^”’])")  # 中文省略号
        self.quote_sentence_end_re = re.compile(r"([。！？!?][”’])([^，。！？!?])")  # 带引号句子
        self.comma_semicolon_re = re.compile(r"([，；,;])")  # 逗号和分号

    def split_text(self, text: str) -> List[str]:
        """根据内容分割文本，包括 PDF 预处理和长句处理。

        Args:
            text (str): 输入文本。

        Returns:
            List[str]: 分割后的句子列表。
        """
        if not text:
            return []

        # PDF 文本预处理
        if self.is_pdf:
            text = self._preprocess_pdf_text(text)

        # 初步按句子边界分割
        segments = self._split_by_sentences(text)

        # 进一步分割长段落
        final_segments = self._split_long_segments(segments)

        return final_segments

    def batch_chunk(self, text_list, max_length: int = 512, overlap_size: int = 50):
        """
        将输入文本列表中的每个文本分割为句子，并将句子合并为最大长度为 max_length 的块。
        每个新块会引入前一个块的最后几个句子，重叠部分的总长度不超过 overlap_size。
        同时若最后的块长度不足16，则将其与前一个块合并。

        Args:
            text_list (List[str]): 输入的文本列表。
            max_length (int): 每个块的最大字符长度，默认为512。
            overlap_size (int): 新块中引入前一个块的重叠部分的最大字符长度，默认为50。

        Returns:
            Tuple[List[List[str]], List[int]]:
                - 分块后的文本列表（二维列表），其中每个元素是一个文本对应的分块列表。
                - 每个文本对应的累积分块数列表。例如，如果有三个文本，其分块数分别为3、2、4，
                则返回的累积计数可能为[3, 5, 9]。通过相邻差分即可还原每个文本的块数。
        """
        chunk_list = []
        cumulative_counts = [0]

        for text in text_list:
            # 分割为句子列表
            sent_list = self.split_text(text)
            if not sent_list:
                chunk_list.append([])
                cumulative_counts.append(cumulative_counts[-1])
                continue

            # 预计算所有句子的长度
            sent_lengths = [calculate_custom_length(sentence) for sentence in sent_list]

            current_chunk = []
            current_length = 0
            chunks = []
            overlap_sentences = []
            overlap_length = 0

            for i, sentence in enumerate(sent_list):
                sentence_length = sent_lengths[i]

                # 如果当前句子本身长度超过 max_length，需单独作为一个块
                if sentence_length > max_length:
                    if current_chunk:
                        # 保存当前块
                        chunks.append(" ".join(current_chunk))
                        # 重置当前块
                        current_chunk = []
                        current_length = 0

                    # 将长句子作为单独的块
                    chunks.append(sentence)
                    # 重叠部分为该长句
                    overlap_sentences = [sentence]
                    overlap_length = sentence_length
                    continue

                # 检查是否需要开始一个新块
                if current_length + sentence_length > max_length:
                    if current_chunk:
                        # 保存当前块
                        chunks.append(" ".join(current_chunk))

                        # 计算重叠部分
                        overlap_sentences = []
                        overlap_length = 0
                        # 从当前_chunk的末尾开始，逐句添加到重叠部分，直到达到 overlap_size
                        for sent, sent_len in zip(reversed(current_chunk), reversed(sent_lengths[:i])):
                            if overlap_length + sent_len > overlap_size and overlap_sentences:
                                break
                            overlap_sentences.insert(0, sent)
                            overlap_length += sent_len

                        # 确保至少有一个句子重叠
                        if not overlap_sentences:
                            overlap_sentences = [current_chunk[-1]]
                            overlap_length = calculate_custom_length(current_chunk[-1])

                    # 开始新块，包含重叠部分
                    current_chunk = overlap_sentences.copy()
                    current_length = overlap_length

                # 添加当前句子到当前块
                current_chunk.append(sentence)
                current_length += sentence_length

            # 将最后的块添加到列表（如果存在）
            if current_chunk:
                chunks.append(" ".join(current_chunk))

            # 检查最后一个块是否过短，如果过短，则与倒数第二个块合并
            if len(chunks) >= 2:
                last_chunk_length = calculate_custom_length(chunks[-1])
                if last_chunk_length < 16:
                    # 合并最后两个块
                    merged_chunk = chunks[-2] + " " + chunks[-1]
                    chunks[-2] = merged_chunk
                    chunks.pop()  # 删除最后一个块

            # 更新累积计数
            chunk_list.append(chunks)
            cumulative_counts.append(cumulative_counts[-1] + len(chunks))

        return chunk_list, cumulative_counts[1:]


    def _preprocess_pdf_text(self, text: str) -> str:
        """预处理 PDF 文本，去除多余换行和空格。

        Args:
            text (str): 输入 PDF 文本。

        Returns:
            str: 规范化后的文本。
        """
        text = self.newline_excess_re.sub("\n", text)
        text = self.whitespace_re.sub(" ", text)
        text = self.double_newline_re.sub("\n", text)
        return text

    def _split_by_sentences(self, text: str) -> List[str]:
        """使用正则表达式分割文本为句子。

        Args:
            text (str): 输入文本。

        Returns:
            List[str]: 分割后的句子列表。
        """
        text = self.sentence_split_pattern.split(text)
        return [sentence.strip() for sentence in text if sentence.strip()]

    def _split_long_segments(self, segments: List[str]) -> List[str]:
        """进一步分割超过最大长度的段落。

        Args:
            segments (List[str]): 初步分割后的文本段落列表。

        Returns:
            List[str]: 进一步分割后的文本段落列表。
        """
        final_segments = []
        for segment in segments:
            if len(segment) <= self.max_sentence_length:
                final_segments.append(segment)
            else:
                sub_segments = self._split_by_commas(segment)
                for sub_segment in sub_segments:
                    if len(sub_segment) <= self.max_sentence_length:
                        final_segments.append(sub_segment)
                    else:
                        final_segments.extend(self._split_by_spaces(sub_segment))
        return final_segments

    def _split_by_commas(self, text: str) -> List[str]:
        """按逗号和分号进一步分割文本。

        Args:
            text (str): 输入段落。

        Returns:
            List[str]: 分割后的段落列表。
        """
        text = self.comma_semicolon_re.sub(r"\1\n", text)
        return [segment.strip() for segment in text.split("\n") if segment.strip()]

    def _split_by_spaces(self, text: str) -> List[str]:
        """按空格分割超长段落。

        Args:
            text (str): 输入段落。

        Returns:
            List[str]: 分割后的段落列表。
        """
        words = text.split()
        segments = []
        current_segment = ""
        for word in words:
            if len(current_segment) + len(word) + 1 <= self.max_sentence_length:
                current_segment += word + " "
            else:
                segments.append(current_segment.strip())
                current_segment = word + " "
        if current_segment:
            segments.append(current_segment.strip())
        return segments