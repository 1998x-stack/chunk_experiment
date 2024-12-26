import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/" + ".."))

import random
from typing import Callable, List
from IPython.display import HTML, display


class TextHighlighter:
    """
    一个用于在Jupyter Notebook中高亮显示长文本中特定部分的类。

    该类可以随机选择长文本中的一个子串，使用指定的分块API将其分块，并以不同颜色高亮显示每个分块，
    确保相邻分块的颜色不同，以提高可读性。

    Attributes:
        long_text (str): 输入的长文本。
        chunking_api (Callable[[str], List[str]]): 用于将文本分块的API函数。
        max_length (int): 要随机选取的连续文本的最大长度。
        colors (List[str]): 用于高亮显示的颜色列表。
    """

    def __init__(
        self, long_text: str, chunking_api: Callable[[str], List[str]], max_length: int
    ):
        """
        初始化TextHighlighter实例。

        Args:
            long_text (str): 输入的长文本。
            chunking_api (Callable[[str], List[str]]): 用于将文本分块的API函数。
            max_length (int): 要随机选取的连续文本的最大长度。

        Raises:
            ValueError: 如果max_length不是正整数。
            TypeError: 如果chunking_api不是可调用的函数。
        """
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("max_length必须是一个正整数。")
        if not callable(chunking_api):
            raise TypeError("chunking_api必须是一个可调用的函数。")

        self.long_text: str = long_text
        self.chunking_api: Callable[[str], List[str]] = chunking_api
        self.max_length: int = max_length
        self.colors: List[str] = [
            "lightblue",
            "lightgreen",
            "lightpink",
            "lightyellow",
            "lightcoral",
            "lightsalmon",
            "lightseagreen",
            "lightsteelblue",
            "lightgoldenrodyellow",
            "lightcyan",
        ]

    def get_random_chunk(self) -> str:
        """
        随机选取长文本中一个长度为max_length的连续文本片段。

        Returns:
            str: 选取的文本片段。

        Raises:
            ValueError: 如果长文本为空。
        """
        text_length = len(self.long_text)
        if text_length == 0:
            raise ValueError("长文本为空，无法选取文本片段。")
        if text_length <= self.max_length:
            print(f"长文本长度小于或等于max_length ({self.max_length})，返回整个文本。")
            return self.long_text

        start_idx = random.randint(0, text_length - self.max_length)
        selected_text = self.long_text[start_idx : start_idx + self.max_length]
        print(f"随机选取的文本片段起始索引: {start_idx}, 长度: {len(selected_text)}")
        return selected_text

    def chunk_text(self, text: str, wrapper_func=None) -> List[str]:
        """
        使用分块API将文本分块。

        Args:
            text (str): 要分块的文本。

        Returns:
            List[str]: 分块后的文本列表。

        Raises:
            ValueError: 如果分块结果为空。
        """

        chunks = self.chunking_api(text)
        if wrapper_func==None:
            pass
        else:
            chunks = wrapper_func(chunks)
        if not chunks:
            return []
        print(f"文本已分成 {len(chunks)} 块。")
        return chunks

    def generate_colored_html(self, chunks: List[str]) -> str:
        """
        生成带有不同颜色的HTML字符串，用于高亮显示分块。

        Args:
            chunks (List[str]): 分块后的文本列表。

        Returns:
            str: 带有颜色的HTML字符串。
        """
        html_chunks = []
        num_colors = len(self.colors)
        previous_color = ""

        for idx, chunk in enumerate(chunks):
            color = self.colors[idx % num_colors]
            # 确保当前颜色与前一个颜色不同
            if color == previous_color:
                color = self.colors[(idx + 1) % num_colors]
            html_chunk = f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px; margin: 1px;">{chunk}</span>'
            html_chunks.append(html_chunk)
            previous_color = color

        return " ".join(html_chunks)

    def display_highlighted_text(self, wrapper_func=None) -> None:
        """
        执行随机选取、分块和高亮显示的完整流程，并在Jupyter Notebook中展示结果。

        Raises:
            Exception: 如果在任何步骤中发生错误。
        """
        try:
            # 随机选取文本片段
            selected_text = self.get_random_chunk()
            print(f"选取的文本片段:\n{selected_text}\n")

            # 分块
            chunks = self.chunk_text(selected_text)
            if wrapper_func is not None:
                chunks = wrapper_func(chunks)

            print("分块结果:")
            for i, chunk in enumerate(chunks, 1):
                print(f"Chunk {i}: {chunk}")

            # 生成带颜色的HTML
            colored_html = self.generate_colored_html(chunks)

            # 在Jupyter Notebook中显示
            display(HTML(colored_html)) #TODO： 如何把\n\t显示出来？

        except Exception as e:
            print(f"发生错误: {e}")

    def save_highlighted_text(self, output_file: str, wrapper_func=None) -> None:
        """
        执行随机选取、分块和高亮显示的完整流程，并将结果保存到指定文件。

        Args:
            output_file (str): 保存结果的文件路径。

        Raises:
            Exception: 如果在任何步骤中发生错误。
        """
        # TODO: 最好保存为PNG
        # TODO: output_file保留重要的参数，data/test_result
        pass