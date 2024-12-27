import os
import time
from util.embedding_api import EmbeddingClient
from src.semantic_chunk import SemanticChunker, EmbeddingModel
from src.recursive_chunk import RecursiveCharacterTextSplitter
from util.chunk_highlight import TextHighlighter

from util.helpers import semantic_chunks_to_text, process_batch_chunk_output


def get_pathsplitter(splittertype, language, chunk_size, chunk_overlap):
    en_files_path = "chunk_experiment/data/general/en"
    zh_files_path = "chunk_experiment/data/general/zh"

    if language == "en":
        file_paths = en_files_path
    elif language == "zh":
        file_paths = zh_files_path

    TEST_URL = "https://ai-platform-cloud-proxy.polymas.com/ai/common/kb-get-embedding"
    embedding_client = EmbeddingClient(embedding_url=TEST_URL)
    embedding_model = EmbeddingModel(embedding_client)

    semanticcumulativechunker = SemanticChunker(
        embedding_model=embedding_model,
        min_characters_per_sentence=5,
        similarity_threshold=None,  # 使用动态计算阈值
        similarity_percentile=90,
        similarity_window=1,
        mode="cumulative",
        initial_sentences=1,
        min_sentences=1,
        chunk_size=chunk_size,
        min_chunk_size=20,
        threshold_step=0.05,
        sep="🐮🍺",
    ).chunk

    # window
    semanticwindowchunker = SemanticChunker(
        embedding_model=embedding_model,
        min_characters_per_sentence=5,
        similarity_threshold=None,  # 使用动态计算阈值
        similarity_percentile=90,
        similarity_window=1,
        mode="window",
        initial_sentences=1,
        min_sentences=1,
        chunk_size=chunk_size,
        min_chunk_size=20,
        threshold_step=0.05,
        sep="🐮🍺",
    ).chunk

    recursivesplitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        sep_type="chunk_size",
        separators=["\n\n", "\n", "。", "？", "！", "，", " ", ""],
    )
    recursivesentencesplitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        sep_type="sentence"
    )
    recursivechunker = recursivesplitter.split_text
    recursivesentencechunker = recursivesentencesplitter.split_text

    if splittertype == "semanticcumulative":
        splitter = semanticcumulativechunker
    elif splittertype == "semanticwindow":
        splitter = semanticwindowchunker
    elif splittertype == "recursive":
        splitter = recursivechunker
    elif splittertype == "recursivesentence":
        splitter = recursivesentencechunker

    return file_paths, splitter, splittertype


def test_chunk(type, language, chunk_size, chunk_overlap):

    paths, splitter, _ = get_pathsplitter(type, language, chunk_size, chunk_overlap)

    file_paths = [
        os.path.join(paths, f)
        for f in os.listdir(paths)
        if f.endswith(".txt") and os.path.isfile(os.path.join(paths, f))
    ]

    total_time = 0
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            sample_text = f.read()
            print(f"Processing file: {file_path} | Length: {len(sample_text)}")
            sample_text = sample_text[:20000]  # Limit to first 20,000 characters
            # print(f"Processing file: {file_path} | Length: {len(sample_text)}")
        start_time = time.time()
        chunks = splitter(sample_text)
        end_time = time.time()
        total_time += end_time - start_time

    average_time = total_time / len(file_paths) if file_paths else 0
    print(f"平均运行时间:{average_time:.4f}秒")


def save_highlight(type, language, chunk_size, chunk_overlap, max_len=1000):
    paths, splitter, chunk_type = get_pathsplitter(
        type, language, chunk_size, chunk_overlap
    )
    file_paths = [
        os.path.join(paths, f)
        for f in os.listdir(paths)
        if f.endswith(".txt") and os.path.isfile(os.path.join(paths, f))
    ]

    with open(file_paths[0], "r", encoding="utf-8") as f:
        sample_text = f.read()

    if type == "semanticcumulative" or type == "semanticwindow":
        wrapper_func = semantic_chunks_to_text
    elif type == "recursivesentence":
        wrapper_func = process_batch_chunk_output
        sample_text = [sample_text]
    else:
        wrapper_func = None

    highlighter = TextHighlighter(
        long_text=sample_text, chunking_api=splitter, max_length=max_len
    )

    output_dir = "chunk_experiment/data/test_result/general"
    if language=="zh":
        output_dir += "/zh"
    elif language=="en":
        output_dir += "/en"
    
    if type=="semanticcumulative":
        output_dir += "/semanticcumulative"
    elif type=="semanticwindow":
        output_dir += "/semanticwindow"
    elif type=="recursive":
        output_dir += "/recursive"
    elif type=="recursivesentence":
        output_dir += "/recursivesentence"
        
    highlighter.save_highlighted_text(output_dir, chunk_type, wrapper_func=wrapper_func)
    # 显示高亮文本
    # highlighter.display_highlighted_text()


def main():
    file_type = "semanticcumulative"
    language = "en"
    chunk_size = 1000
    chunk_overlap = 100
    test_chunk(file_type, language, chunk_size, chunk_overlap)
    save_highlight(file_type, language, chunk_size, chunk_overlap)


if __name__ == "__main__":
    main()
