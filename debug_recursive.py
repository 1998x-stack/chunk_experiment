#!/usr/bin/env python
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.recursive_chunk import RecursiveCharacterTextSplitter

# Create a short test
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    sep_type="chunk_size",
    separators=["\n\n", "\n", "。", "？", "！", "，", " ", ""]
)

text = "This is a test sentence. This is another sentence. And a third one."
print(f"Input text: {text}")
print(f"Text length: {len(text)}")

chunks = splitter.split_text(text)
print(f"Chunks: {chunks}")
print(f"Type of chunks: {type(chunks)}")
print(f"Length of chunks list: {len(chunks)}")

for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk} (type: {type(chunk)})")