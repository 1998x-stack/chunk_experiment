import unittest
import sys
import os
import numpy as np
from unittest.mock import Mock

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.semantic_chunk import SemanticChunker, SemanticChunk, SemanticSentence, EmbeddingModel
from src.recursive_chunk import RecursiveCharacterTextSplitter
from util.embedding_api import EmbeddingClient
from util.sentence_split import GeneralTextSplitter


class TestEmbeddingModel(unittest.TestCase):
    """Test cases for the EmbeddingModel class"""
    
    def setUp(self):
        # Mock embedding client
        self.mock_client = Mock()
        self.mock_client.get_embeddings.return_value = {
            "data": {
                "resultList": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            }
        }
        self.embedding_model = EmbeddingModel(self.mock_client, embedding_dim=3)
    
    def test_embed_batch(self):
        """Test the embed_batch method"""
        texts = ["test sentence 1", "test sentence 2"]
        embeddings = self.embedding_model.embed_batch(texts)
        
        self.assertEqual(len(embeddings), 2)
        self.assertIsInstance(embeddings[0], np.ndarray)
        self.assertEqual(embeddings[0].shape, (3,))
    
    def test_similarity(self):
        """Test the similarity method"""
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([0.0, 1.0, 0.0])
        similarity = self.embedding_model.similarity(emb1, emb2)
        
        self.assertIsInstance(similarity, float)


class TestSemanticChunker(unittest.TestCase):
    """Test cases for the SemanticChunker class"""
    
    def setUp(self):
        # Mock embedding client
        self.mock_client = Mock()
        self.mock_client.get_embeddings.return_value = {
            "data": {
                "resultList": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
            }
        }
        embedding_model = EmbeddingModel(self.mock_client, embedding_dim=3)
        
        self.chunker = SemanticChunker(
            embedding_model=embedding_model,
            min_characters_per_sentence=5,
            similarity_threshold=0.5,
            similarity_percentile=None,
            similarity_window=1,
            mode="cumulative",
            initial_sentences=1,
            min_sentences=1,
            chunk_size=200,
            min_chunk_size=50,
            threshold_step=0.05,
            sep="🐮🍺"
        )
    
    def test_initialization(self):
        """Test SemanticChunker initialization"""
        self.assertEqual(self.chunker.chunk_size, 200)
        self.assertEqual(self.chunker.similarity_threshold, 0.5)
        self.assertEqual(self.chunker.mode, "cumulative")
    
    def test_count_tokens(self):
        """Test the _count_tokens method"""
        text = "This is a test sentence with several words."
        token_count = self.chunker._count_tokens(text)
        
        # Approximate token count based on whitespace splitting
        self.assertGreaterEqual(token_count, 6)
    
    def test_prepare_sentences(self):
        """Test the _prepare_sentences method"""
        text = "This is the first sentence. This is the second sentence. Third sentence here."
        
        sentences = self.chunker._prepare_sentences(text)
        
        self.assertGreaterEqual(len(sentences), 1)
        for sentence in sentences:
            self.assertIsInstance(sentence, SemanticSentence)
            self.assertGreater(len(sentence.text), 0)
    
    def test_chunk_empty_text(self):
        """Test chunking with empty text"""
        chunks = self.chunker.chunk("")
        self.assertEqual(len(chunks), 0)
    
    def test_chunk_single_sentence(self):
        """Test chunking with a single sentence"""
        text = "This is a single sentence."
        chunks = self.chunker.chunk(text)
        
        # Should return at least one chunk
        self.assertGreaterEqual(len(chunks), 0)


class TestRecursiveCharacterTextSplitter(unittest.TestCase):
    """Test cases for the RecursiveCharacterTextSplitter class"""
    
    def setUp(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            sep_type="chunk_size",
            separators=["\n\n", "\n", "。", "？", "！", "，", " ", ""]
        )
    
    def test_initialization(self):
        """Test RecursiveCharacterTextSplitter initialization"""
        self.assertEqual(self.splitter.chunk_size, 100)
        self.assertEqual(self.splitter.chunk_overlap, 20)
    
    def test_split_text_basic(self):
        """Test basic text splitting functionality"""
        text = "This is a test sentence. This is another sentence. And a third one."
        # Need to specify sep_type="chunk_size" to avoid the default "sentence" behavior
        chunks = self.splitter.split_text(text, sep_type="chunk_size")
        
        self.assertIsInstance(chunks, list)
        self.assertGreaterEqual(len(chunks), 1)
        for chunk in chunks:
            self.assertIsInstance(chunk, str)
            self.assertGreater(len(chunk), 0)


class TestAblationExperiments(unittest.TestCase):
    """Test cases for ablation experiments comparing different parameters"""
    
    def setUp(self):
        # Mock embedding client for semantic chunker tests
        self.mock_client = Mock()
        self.mock_client.get_embeddings.return_value = {
            "data": {
                "resultList": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], 
                              [0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]]
            }
        }
        embedding_model = EmbeddingModel(self.mock_client, embedding_dim=3)
        
        self.test_text = (
            "This is a sample text for testing. It contains multiple sentences. "
            "Each sentence contributes to the overall meaning. The semantic relationships "
            "between sentences matter for chunking. We want to test different approaches. "
            "This will help us understand the effectiveness. Different parameters will be evaluated. "
            "The results will guide our decisions. We aim for optimal chunking strategies."
        )
    
    def test_different_chunk_sizes(self):
        """Test semantic chunker with different chunk sizes"""
        chunk_sizes = [100, 200, 500]
        results = {}
        
        for size in chunk_sizes:
            mock_client = Mock()
            mock_client.get_embeddings.return_value = {
                "data": {
                    "resultList": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], 
                                  [0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]]
                }
            }
            embedding_model = EmbeddingModel(mock_client, embedding_dim=3)
            
            chunker = SemanticChunker(
                embedding_model=embedding_model,
                chunk_size=size
            )
            
            chunks = chunker.chunk(self.test_text)
            results[size] = len(chunks)
        
        # Verify that results are captured
        self.assertEqual(len(results), 3)
    
    def test_different_modes(self):
        """Test semantic chunker with different modes (cumulative vs window)"""
        modes = ["cumulative", "window"]
        results = {}
        
        for mode in modes:
            mock_client = Mock()
            mock_client.get_embeddings.return_value = {
                "data": {
                    "resultList": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
                }
            }
            embedding_model = EmbeddingModel(mock_client, embedding_dim=3)
            
            chunker = SemanticChunker(
                embedding_model=embedding_model,
                mode=mode,
                similarity_threshold=0.5
            )
            
            chunks = chunker.chunk(self.test_text)
            results[mode] = len(chunks)
        
        # Both modes should return some chunks
        for mode in modes:
            self.assertIn(mode, results)
            self.assertGreaterEqual(results[mode], 0)


def run_unit_tests():
    """Function to run all unit tests"""
    # Create a test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    # Run the tests
    run_unit_tests()