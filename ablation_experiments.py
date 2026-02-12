import os
import sys
import json
import time
import numpy as np
from datetime import datetime
from unittest.mock import Mock
from typing import Dict, List, Tuple

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.semantic_chunk import SemanticChunker, EmbeddingModel
from src.recursive_chunk import RecursiveCharacterTextSplitter
from util.embedding_api import EmbeddingClient


class AblationExperimentRunner:
    """Class to run systematic ablation experiments on chunking algorithms"""
    
    def __init__(self, test_data_path: str = "data/general/en"):
        self.test_data_path = test_data_path
        self.results = []
        
    def load_test_data(self, max_files: int = 5, max_chars: int = 10000) -> List[Tuple[str, str]]:
        """Load test data from the specified directory"""
        test_files = []
        
        # Find txt files in the test directory
        for filename in os.listdir(self.test_data_path):
            if filename.endswith('.txt'):
                filepath = os.path.join(self.test_data_path, filename)
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()[:max_chars]  # Limit content size
                    test_files.append((filename, content))
                    
                if len(test_files) >= max_files:
                    break
        
        return test_files
    
    def run_semantic_chunker_experiment(self, text: str, params: Dict) -> Dict:
        """Run experiment with semantic chunker using specified parameters"""
        # Create mock embedding client
        mock_client = Mock()
        # Generate mock embeddings - we'll need as many as the number of sentences
        # Estimate number of sentences by counting sentence terminators
        import re
        sentences = re.split(r'[.!?。！？]', text)
        n_sentences = len([s for s in sentences if s.strip()])
        
        # If no sentences detected, use a reasonable default
        n_sentences = max(n_sentences, 10)
        
        # Generate mock embeddings
        mock_embeddings = [[float(j)/100.0 for j in range(128)] for _ in range(n_sentences)]
        mock_client.get_embeddings.return_value = {
            "data": {
                "resultList": mock_embeddings
            }
        }
        
        embedding_model = EmbeddingModel(mock_client, embedding_dim=128)
        
        chunker = SemanticChunker(
            embedding_model=embedding_model,
            chunk_size=params.get('chunk_size', 200),
            min_characters_per_sentence=params.get('min_characters_per_sentence', 5),
            similarity_threshold=params.get('similarity_threshold', None),
            similarity_percentile=params.get('similarity_percentile', 90),
            similarity_window=params.get('similarity_window', 1),
            mode=params.get('mode', 'cumulative'),
            initial_sentences=params.get('initial_sentences', 1),
            min_sentences=params.get('min_sentences', 1),
            min_chunk_size=params.get('min_chunk_size', 50),
            threshold_step=params.get('threshold_step', 0.05),
            sep=params.get('sep', "🐮🍺")
        )
        
        start_time = time.time()
        chunks = chunker.chunk(text)
        end_time = time.time()
        
        return {
            'algorithm': 'semantic',
            'params': params,
            'n_chunks': len(chunks),
            'avg_chunk_size': np.mean([len(chunk.text) for chunk in chunks]) if chunks else 0,
            'total_processing_time': end_time - start_time,
            'text_length': len(text)
        }
    
    def run_recursive_chunker_experiment(self, text: str, params: Dict) -> Dict:
        """Run experiment with recursive chunker using specified parameters"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=params.get('chunk_size', 200),
            chunk_overlap=params.get('chunk_overlap', 50),
            sep_type=params.get('sep_type', 'chunk_size'),
            separators=params.get('separators', ["\n\n", "\n", "。", "？", "！", "，", " ", ""])
        )
        
        start_time = time.time()
        chunks = splitter.split_text(text, sep_type=params.get('sep_type', 'chunk_size'))
        end_time = time.time()
        
        return {
            'algorithm': 'recursive',
            'params': params,
            'n_chunks': len(chunks),
            'avg_chunk_size': np.mean([len(chunk) for chunk in chunks]) if chunks else 0,
            'total_processing_time': end_time - start_time,
            'text_length': len(text)
        }
    
    def run_parameter_ablation_study(self):
        """Run systematic experiments varying different parameters"""
        # Load test data
        test_data = self.load_test_data()
        print(f"Loaded {len(test_data)} test files for experiments")
        
        # Define parameter combinations to test
        param_combinations = []
        
        # Test different chunk sizes for both algorithms
        for chunk_size in [100, 200, 500, 1000]:
            # Semantic chunker variations
            param_combinations.append({
                'algorithm': 'semantic',
                'chunk_size': chunk_size,
                'mode': 'cumulative',
                'similarity_threshold': None,  # Use dynamic threshold
                'similarity_percentile': 90
            })
            
            param_combinations.append({
                'algorithm': 'semantic',
                'chunk_size': chunk_size,
                'mode': 'window',
                'similarity_threshold': None,  # Use dynamic threshold
                'similarity_percentile': 90
            })
            
            # Recursive chunker variations
            param_combinations.append({
                'algorithm': 'recursive',
                'chunk_size': chunk_size,
                'chunk_overlap': chunk_size // 5,  # 20% overlap
                'sep_type': 'chunk_size'
            })
        
        # Test different similarity thresholds for semantic chunker
        for threshold in [0.3, 0.5, 0.7, 0.9]:
            param_combinations.append({
                'algorithm': 'semantic',
                'chunk_size': 500,
                'mode': 'cumulative',
                'similarity_threshold': threshold,
                'similarity_percentile': None
            })
        
        # Test different overlap values for recursive chunker
        for overlap_ratio in [0.1, 0.2, 0.3, 0.5]:
            chunk_size = 500
            param_combinations.append({
                'algorithm': 'recursive',
                'chunk_size': chunk_size,
                'chunk_overlap': int(chunk_size * overlap_ratio),
                'sep_type': 'chunk_size'
            })
        
        print(f"Running {len(param_combinations)} parameter combinations...")
        
        # Run experiments
        for i, params in enumerate(param_combinations):
            print(f"Running experiment {i+1}/{len(param_combinations)}: {params['algorithm']} with chunk_size={params['chunk_size']}")
            
            exp_results = {'experiment_id': i, 'params': params, 'results': []}
            
            for filename, text in test_data:
                try:
                    if params['algorithm'] == 'semantic':
                        result = self.run_semantic_chunker_experiment(text, {k: v for k, v in params.items() if k != 'algorithm'})
                    else:
                        result = self.run_recursive_chunker_experiment(text, {k: v for k, v in params.items() if k != 'algorithm'})
                    
                    result['filename'] = filename
                    exp_results['results'].append(result)
                except Exception as e:
                    print(f"Error running experiment with file {filename}: {e}")
                    continue
            
            # Calculate aggregate metrics for this experiment
            if exp_results['results']:
                avg_n_chunks = np.mean([r['n_chunks'] for r in exp_results['results']])
                avg_chunk_size = np.mean([r['avg_chunk_size'] for r in exp_results['results']])
                avg_processing_time = np.mean([r['total_processing_time'] for r in exp_results['results']])
                
                exp_results['aggregate'] = {
                    'avg_n_chunks': avg_n_chunks,
                    'avg_chunk_size': avg_chunk_size,
                    'avg_processing_time': avg_processing_time
                }
                
                self.results.append(exp_results)
        
        return self.results
    
    def compare_algorithms(self):
        """Compare performance between semantic and recursive chunkers"""
        # Run comparison experiments
        test_data = self.load_test_data(max_files=3)  # Use fewer files for quick comparison
        comparison_results = []
        
        common_params = {
            'chunk_size': 500,
            'chunk_overlap': 100
        }
        
        for filename, text in test_data:
            print(f"Comparing algorithms on file: {filename}")
            
            # Run semantic chunker (cumulative mode)
            semantic_params = {
                **common_params,
                'mode': 'cumulative',
                'similarity_threshold': None,
                'similarity_percentile': 90
            }
            semantic_result = self.run_semantic_chunker_experiment(text, semantic_params)
            semantic_result['filename'] = filename
            
            # Run semantic chunker (window mode)
            semantic_params_window = {
                **common_params,
                'mode': 'window',
                'similarity_threshold': None,
                'similarity_percentile': 90
            }
            semantic_window_result = self.run_semantic_chunker_experiment(text, semantic_params_window)
            semantic_window_result['filename'] = filename
            
            # Run recursive chunker
            recursive_params = {
                **common_params,
                'sep_type': 'chunk_size'
            }
            recursive_result = self.run_recursive_chunker_experiment(text, recursive_params)
            recursive_result['filename'] = filename
            
            comparison_results.append({
                'filename': filename,
                'semantic_cumulative': semantic_result,
                'semantic_window': semantic_window_result,
                'recursive': recursive_result
            })
        
        return comparison_results
    
    def save_results(self, results: List[Dict], filename: str = None):
        """Save experimental results to a JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experiment_results_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {filename}")
        return filename


def run_ablation_experiments():
    """Main function to run the ablation experiments"""
    print("Starting ablation experiments...")
    
    # Initialize the experiment runner
    runner = AblationExperimentRunner()
    
    # Run parameter ablation study
    print("\n1. Running parameter ablation study...")
    param_results = runner.run_parameter_ablation_study()
    
    # Save parameter study results
    runner.save_results(param_results, "param_ablation_results.json")
    
    # Run algorithm comparison
    print("\n2. Running algorithm comparison...")
    comparison_results = runner.compare_algorithms()
    
    # Save comparison results
    runner.save_results(comparison_results, "algorithm_comparison_results.json")
    
    # Print summary
    print("\n3. Experiment Summary:")
    print(f"- Parameter ablation study completed with {len(param_results)} configurations tested")
    print(f"- Algorithm comparison completed on {len(comparison_results)} files")
    
    # Print some key findings
    if comparison_results:
        first_comparison = comparison_results[0]  # Look at first file comparison
        
        print("\nSample comparison results (first file):")
        print(f"  Semantic (cumulative) - Chunks: {first_comparison['semantic_cumulative']['n_chunks']}, "
              f"Avg size: {first_comparison['semantic_cumulative']['avg_chunk_size']:.1f}, "
              f"Time: {first_comparison['semantic_cumulative']['total_processing_time']:.3f}s")
        
        print(f"  Semantic (window) - Chunks: {first_comparison['semantic_window']['n_chunks']}, "
              f"Avg size: {first_comparison['semantic_window']['avg_chunk_size']:.1f}, "
              f"Time: {first_comparison['semantic_window']['total_processing_time']:.3f}s")
        
        print(f"  Recursive - Chunks: {first_comparison['recursive']['n_chunks']}, "
              f"Avg size: {first_comparison['recursive']['avg_chunk_size']:.1f}, "
              f"Time: {first_comparison['recursive']['total_processing_time']:.3f}s")
    
    return param_results, comparison_results


if __name__ == "__main__":
    # Run the experiments
    param_results, comparison_results = run_ablation_experiments()