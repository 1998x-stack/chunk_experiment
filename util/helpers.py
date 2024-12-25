import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/" + ".."))

from typing import List


def semantic_chunks_to_text(semantic_chunks: List) -> List[str]:
    """
    将 SemanticChunk 对象列表转换为纯文本列表。

    Args:
        semantic_chunks (List[SemanticChunk]): SemanticChunk 对象的列表。

    Returns:
        List[str]: 提取的纯文本组成的列表。
    """
    if not semantic_chunks:
        raise ValueError("提供的 SemanticChunk 列表为空。")

    if not all(hasattr(chunk, "text") for chunk in semantic_chunks):
        raise ValueError("列表中的对象不是有效的 SemanticChunk。请检查输入格式。")

    # 遍历 SemanticChunk 对象，将它们的 `text` 属性提取为字符串
    return [chunk.text for chunk in semantic_chunks]


def process_batch_chunk_output(batch_chunk_output):
    chunks = []
    chunk_list, _ = batch_chunk_output
    for i, chunk_group in enumerate(chunk_list):
        for j, chunk in enumerate(chunk_group):
            chunks.append(chunk)
    return chunks


def string2list(single_string):
    return [single_string]
