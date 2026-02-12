import sys
import os
import numpy as np
from unittest.mock import Mock

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.enhanced_semantic_chunk import EnhancedSemanticChunker, EnhancedEmbeddingModel, create_default_enhanced_chunker
from src.enhanced_recursive_chunk import EnhancedRecursiveCharacterTextSplitter, create_default_enhanced_recursive_splitter
from util.enhanced_sentence_split import EnhancedGeneralTextSplitter, create_default_enhanced_splitter


def test_enhanced_semantic_chunker():
    """Test the enhanced semantic chunker."""
    print("Testing Enhanced Semantic Chunker...")
    
    # Create mock embedding client
    mock_client = Mock()
    mock_client.get_embeddings.return_value = {
        "data": {
            "resultList": [
                [0.1, 0.2, 0.3, 0.4], [0.4, 0.5, 0.6, 0.7], 
                [0.7, 0.8, 0.9, 1.0], [0.2, 0.3, 0.4, 0.5],
                [0.5, 0.6, 0.7, 0.8]
            ]
        }
    }
    
    # Create enhanced embedding model
    embedding_model = EnhancedEmbeddingModel(mock_client, embedding_dim=4)
    
    # Create enhanced semantic chunker
    chunker = EnhancedSemanticChunker(
        embedding_model=embedding_model,
        chunk_size=100,
        min_characters_per_sentence=5,
        similarity_threshold=0.5,
        similarity_percentile=85,
        mode="cumulative",
        context_window_size=2,
        adaptive_threshold=True,
        enable_optimization=True
    )
    
    # Test text
    test_text = (
        "This is the first sentence. It contains important information. "
        "The second sentence builds upon the first. Together they form a coherent thought. "
        "Here is a third idea that connects to the previous sentences. "
        "This part discusses a different topic entirely. It introduces a new concept. "
        "The final sentences wrap up the discussion."
    )
    
    # Chunk the text
    chunks = chunker.chunk(test_text)
    
    print(f"  Original text length: {len(test_text)}")
    print(f"  Number of chunks: {len(chunks)}")
    
    if chunks:
        print(f"  Average chunk size: {sum(len(chunk.text) for chunk in chunks) / len(chunks):.1f}")
        print(f"  Average semantic coherence: {sum(chunk.semantic_coherence for chunk in chunks) / len(chunks):.3f}")
        
        for i, chunk in enumerate(chunks):
            print(f"    Chunk {i+1}: {len(chunk.text)} chars, coherence={chunk.semantic_coherence:.3f}")
            print(f"      Text preview: '{chunk.text[:50]}...' if len(chunk.text) > 50 else chunk.text")
    
    # Test statistics
    stats = chunker.get_statistics(chunks)
    print(f"  Statistics: {stats}")
    
    print("  ✓ Enhanced Semantic Chunker test completed\n")


def test_enhanced_recursive_splitter():
    """Test the enhanced recursive splitter."""
    print("Testing Enhanced Recursive Splitter...")
    
    # Create enhanced splitter
    splitter = EnhancedRecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        separators=["\n\n", "\n", "。", "？", "！", ".", "?", "!", "，", ",", " ", ""],
        min_chunk_size=10,
        chunk_strategy="balanced",
        strip_separators=True
    )
    
    # Test text
    test_text = (
        "This is the first paragraph. It contains several sentences that belong together. "
        "The second sentence continues the thought.\n\n"
        "This is the second paragraph. It introduces a new topic. "
        "Additional details are provided in this section.\n\n"
        "The third paragraph concludes with final thoughts. "
        "All ideas come together in this closing section."
    )
    
    # Split the text
    chunks = splitter.split_text(test_text, sep_type="chunk_size")
    
    print(f"  Original text length: {len(test_text)}")
    print(f"  Number of chunks: {len(chunks)}")
    
    if chunks:
        print(f"  Average chunk size: {sum(len(chunk) for chunk in chunks) / len(chunks):.1f}")
        
        for i, chunk in enumerate(chunks):
            print(f"    Chunk {i+1}: {len(chunk)} chars")
            print(f"      Text preview: '{chunk[:50]}...' if len(chunk) > 50 else chunk")
    
    # Test statistics
    stats = splitter.get_statistics(test_text, chunks)
    print(f"  Statistics: {stats}")
    
    print("  ✓ Enhanced Recursive Splitter test completed\n")


def test_enhanced_sentence_splitter():
    """Test the enhanced sentence splitter."""
    print("Testing Enhanced Sentence Splitter...")
    
    # Create enhanced splitter
    splitter = EnhancedGeneralTextSplitter(
        language="mixed",
        max_sentence_length=100,
        min_sentence_length=5,
        split_long_sentences=True,
        enable_hybrid_splitting=True
    )
    
    # Test text
    test_text = (
        "这是一个测试句子。它包含一些重要信息。第二个句子延续了第一个句子的想法。"
        "This is an English sentence. It follows the Chinese sentences. "
        "Here is another English sentence that continues the thought. "
        "这是第三个中文句子，用来测试混合语言支持。"
    )
    
    # Split the text
    sentences = splitter.split_text(test_text)
    
    print(f"  Original text length: {len(test_text)}")
    print(f"  Number of sentences: {len(sentences)}")
    
    if sentences:
        print(f"  Average sentence length: {sum(len(s) for s in sentences) / len(sentences):.1f}")
        
        for i, sentence in enumerate(sentences):
            print(f"    Sentence {i+1}: {len(sentence)} chars")
            print(f"      Text: '{sentence}'")
    
    # Test batch chunking
    chunks, counts = splitter.batch_chunk([test_text], max_length=80, overlap_size=10)
    print(f"  Batch chunking - Chunks: {len(chunks[0]) if chunks else 0}")
    
    # Test statistics
    stats = splitter.get_statistics(test_text)
    print(f"  Statistics: {stats}")
    
    print("  ✓ Enhanced Sentence Splitter test completed\n")


def test_default_creators():
    """Test the default creator functions."""
    print("Testing Default Creator Functions...")
    
    # Test default enhanced chunker
    chunker = create_default_enhanced_chunker(chunk_size=150)
    print(f"  Default chunker created with chunk_size: {chunker.chunk_size}")
    
    # Test default recursive splitter
    splitter = create_default_enhanced_recursive_splitter(chunk_size=150)
    print(f"  Default recursive splitter created with chunk_size: {splitter.chunk_size}")
    
    # Test default sentence splitter
    sent_splitter = create_default_enhanced_splitter(language="mixed")
    print(f"  Default sentence splitter created for language: {sent_splitter.language}")
    
    print("  ✓ Default creators test completed\n")


def test_performance_comparison():
    """Compare performance of enhanced vs original algorithms."""
    print("Performance Comparison Test...")
    
    # Create test text
    test_text = (
        "This is a sample text for performance testing. It contains multiple sentences. "
        "Each sentence adds to the overall length. The text continues with more content. "
        "Additional sentences make the text longer. More content is added here. "
        "Yet another sentence contributes to the total. The text keeps growing. "
        "More sentences are added for testing purposes. The content expands further. "
        "Additional text increases the length. More content makes it longer. "
        "Yet more sentences are included. The text continues to grow. "
        "More content is added for testing. The text gets even longer. "
        "Additional sentences contribute. More text is included. "
        "The sample text continues. More content is added. "
        "Sentences keep being added. The text grows longer. "
        "More content is included. The text expands. "
        "Sentences add to length. Content increases. "
        "Text continues growing. Size increases. "
    ) * 5  # Repeat to make it longer
    
    # Test enhanced semantic chunker
    mock_client = Mock()
    # Generate enough embeddings for the test text
    word_count = len(test_text.split())
    embeddings_list = [[float(j % 128)/100.0 for j in range(128)] for _ in range(min(word_count, 50))]
    mock_client.get_embeddings.return_value = {
        "data": {
            "resultList": embeddings_list
        }
    }
    
    embedding_model = EnhancedEmbeddingModel(mock_client, embedding_dim=128)
    enhanced_chunker = EnhancedSemanticChunker(
        embedding_model=embedding_model,
        chunk_size=100,
        mode="cumulative",
        adaptive_threshold=True
    )
    
    import time
    start_time = time.time()
    enhanced_chunks = enhanced_chunker.chunk(test_text)
    enhanced_time = time.time() - start_time
    
    print(f"  Enhanced chunker: {len(enhanced_chunks)} chunks in {enhanced_time:.4f}s")
    
    # Compare with default creator
    default_chunker = create_default_enhanced_chunker(chunk_size=100)
    start_time = time.time()
    default_chunks = default_chunker.chunk(test_text)
    default_time = time.time() - start_time
    
    print(f"  Default chunker: {len(default_chunks)} chunks in {default_time:.4f}s")
    
    print("  ✓ Performance comparison completed\n")


def run_all_tests():
    """Run all tests for enhanced algorithms."""
    print("=" * 60)
    print("ENHANCED ALGORITHMS TEST SUITE")
    print("=" * 60)
    
    try:
        test_enhanced_semantic_chunker()
        test_enhanced_recursive_splitter()
        test_enhanced_sentence_splitter()
        test_default_creators()
        test_performance_comparison()
        
        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        print("TESTS FAILED! ✗")
        print("=" * 60)


if __name__ == "__main__":
    run_all_tests()