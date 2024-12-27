import os
import time
from src.html_chunk import HTMLSectionSplitter
from src.pythoncode_chunk import PythonCodeTextSplitter
from src.markdown_chunk import MarkdownTextSplitter
from src.latex_chunk import LatexTextSplitter
from util.chunk_highlight import TextHighlighter


def get_pathsplitter(type, chunk_size, chunk_overlap):
    # Define file paths
    html_files_path = "chunk_experiment/data/html/arxiv"
    latex_files_path = "chunk_experiment/data/latex/arxiv"
    md_files_path = "chunk_experiment/data/markdown/markdown-documentation-transformers"
    python_files_path = "chunk_experiment/data/python"

    # Initialize splitters
    html_splitter = HTMLSectionSplitter(
        headers_to_split_on=[
            ("h1", "Header 1"),
            ("h2", "Header 2"),
            ("h3", "Header 3"),
        ]
    )
    python_splitter = PythonCodeTextSplitter(chunk_size, chunk_overlap)
    md_splitter = MarkdownTextSplitter(chunk_size, chunk_overlap)
    tex_splitter = LatexTextSplitter(chunk_size, chunk_overlap)

    # Select splitter and path based on file type
    if type == ".html":
        return html_files_path, html_splitter, "HTMLSectionSplitter"
    elif type == ".tex":
        return latex_files_path, tex_splitter, "LatexTextSplitter"
    elif type == ".md":
        return md_files_path, md_splitter, "MarkdownTextSplitter"
    elif type == ".py":
        return python_files_path, python_splitter, "PythonCodeTextSplitter"
    else:
        raise ValueError(f"Unsupported file type: {type}")


def test_chunk(type, chunk_size, chunk_overlap):

    paths, splitter, _ = get_pathsplitter(type, chunk_size, chunk_overlap)
    # Gather file paths
    file_paths = [os.path.join(paths, f) for f in os.listdir(paths) if f.endswith(type)]
    file_paths = file_paths[:1]  # Limit to first 100 files

    total_time = 0

    if type == ".tex":
        encoding = "latin1"
    else:
        encoding = "utf-8"
    for file_path in file_paths:
        with open(file_path, "r", encoding=encoding) as f:
            sample_text = f.read()
            # sample_text = sample_text[:20000]  # Limit to first 20,000 characters
            # print(f"Processing file: {file_path} | Length: {len(sample_text)}")

        # Specific preprocessing for HTML files
        if type == ".html":
            sample_text = sample_text.replace(
                '<?xml version="1.0" encoding="UTF-8"?>', ""
            )

        start_time = time.time()
        chunks = splitter.split_text(sample_text)
        end_time = time.time()
        total_time += end_time - start_time

    # for idx, split_doc in enumerate(chunks, 1):
    #     print(f"Document {idx}:")
    #     print(f"Metadata: {split_doc.metadata}")
    #     print(f"Content: {split_doc.page_content}\n")
    average_time = total_time / len(file_paths) if file_paths else 0
    print(
        f"Processed {len(file_paths)} documents. Average runtime: {average_time:.4f} seconds"
    )
    print(f"Last processed chunks: {chunks}")


def save_highlight(type, chunk_size, chunk_overlap, max_len=2000):
    paths, splitter, chunk_type = get_pathsplitter(type, chunk_size, chunk_overlap)
    file_paths = [os.path.join(paths, f) for f in os.listdir(paths) if f.endswith(type)]

    if type == ".tex":
        encoding = "latin1"
    else:
        encoding = "utf-8"
    with open(file_paths[0], "r", encoding=encoding) as f:
        sample_text = f.read()

    highlighter = TextHighlighter(
        long_text=sample_text, chunking_api=splitter.split_text, max_length=max_len
    )

    output_dir = "chunk_experiment/data/test_result"
    if type == ".tex":
        output_dir+= "/latex"
    elif type == ".md":
        output_dir+= "/markdown"
    elif type == ".py":
        output_dir+= "/python"
    elif type == ".html":
        output_dir+= "/html"
    
    highlighter.save_highlighted_text(output_dir, chunk_type)
    highlighter.display_highlighted_text()


def main():
    file_type = ".tex"
    chunk_size = 100
    chunk_overlap = 20
    test_chunk(file_type, chunk_size, chunk_overlap)
    save_highlight(file_type, chunk_size, chunk_overlap)


if __name__ == "__main__":
    main()
