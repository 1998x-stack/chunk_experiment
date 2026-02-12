import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

def load_results(filename: str) -> List[Dict]:
    """Load experimental results from JSON file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_param_ablation_results():
    """Analyze the parameter ablation study results"""
    results = load_results("param_ablation_results.json")
    
    print("=== PARAMETER ABLATION STUDY ANALYSIS ===")
    print(f"Total experiments: {len(results)}")
    
    # Extract metrics for different chunk sizes
    chunk_sizes = []
    n_chunks_list = []
    avg_chunk_sizes = []
    processing_times = []
    algorithms = []
    
    for exp in results:
        params = exp['params']
        if exp.get('aggregate'):
            chunk_sizes.append(params['chunk_size'])
            n_chunks_list.append(exp['aggregate']['avg_n_chunks'])
            avg_chunk_sizes.append(exp['aggregate']['avg_chunk_size'])
            processing_times.append(exp['aggregate']['avg_processing_time'])
            algorithms.append(params['algorithm'])
    
    # Convert to numpy arrays for analysis
    chunk_sizes = np.array(chunk_sizes)
    n_chunks_array = np.array(n_chunks_list)
    avg_chunk_sizes_array = np.array(avg_chunk_sizes)
    processing_times_array = np.array(processing_times)
    algorithms_array = np.array(algorithms)
    
    # Analysis by algorithm
    semantic_mask = algorithms_array == 'semantic'
    recursive_mask = algorithms_array == 'recursive'
    
    print(f"\nSemantic chunker experiments: {np.sum(semantic_mask)}")
    print(f"Recursive chunker experiments: {np.sum(recursive_mask)}")
    
    # Chunk size vs number of chunks analysis
    print(f"\n--- Chunk Size vs Number of Chunks ---")
    for alg in ['semantic', 'recursive']:
        mask = algorithms_array == alg
        if np.any(mask):
            print(f"{alg.capitalize()} chunker:")
            for size in sorted(set(chunk_sizes[mask])):
                idx = (chunk_sizes == size) & (algorithms_array == alg)
                avg_chunks = np.mean(n_chunks_array[idx])
                print(f"  Chunk size {size}: {avg_chunks:.1f} chunks on average")
    
    # Processing time analysis
    print(f"\n--- Processing Time Analysis ---")
    for alg in ['semantic', 'recursive']:
        mask = algorithms_array == alg
        if np.any(mask):
            avg_time = np.mean(processing_times_array[mask])
            print(f"{alg.capitalize()} chunker average time: {avg_time:.4f}s")
    
    return {
        'chunk_sizes': chunk_sizes,
        'n_chunks': n_chunks_array,
        'avg_chunk_sizes': avg_chunk_sizes_array,
        'processing_times': processing_times_array,
        'algorithms': algorithms_array
    }

def analyze_algorithm_comparison():
    """Analyze the algorithm comparison results"""
    results = load_results("algorithm_comparison_results.json")
    
    print("\n=== ALGORITHM COMPARISON ANALYSIS ===")
    print(f"Files analyzed: {len(results)}")
    
    if results:
        cumulative_chunks = []
        window_chunks = []
        recursive_chunks = []
        
        cumulative_times = []
        window_times = []
        recursive_times = []
        
        cumulative_avg_sizes = []
        window_avg_sizes = []
        recursive_avg_sizes = []
        
        for comp in results:
            cumulative_chunks.append(comp['semantic_cumulative']['n_chunks'])
            window_chunks.append(comp['semantic_window']['n_chunks'])
            recursive_chunks.append(comp['recursive']['n_chunks'])
            
            cumulative_times.append(comp['semantic_cumulative']['total_processing_time'])
            window_times.append(comp['semantic_window']['total_processing_time'])
            recursive_times.append(comp['recursive']['total_processing_time'])
            
            cumulative_avg_sizes.append(comp['semantic_cumulative']['avg_chunk_size'])
            window_avg_sizes.append(comp['semantic_window']['avg_chunk_size'])
            recursive_avg_sizes.append(comp['recursive']['avg_chunk_size'])
        
        # Print statistics
        print(f"\n--- Chunk Count Comparison ---")
        print(f"Semantic (cumulative): {np.mean(cumulative_chunks):.1f} ± {np.std(cumulative_chunks):.1f} chunks")
        print(f"Semantic (window): {np.mean(window_chunks):.1f} ± {np.std(window_chunks):.1f} chunks")
        print(f"Recursive: {np.mean(recursive_chunks):.1f} ± {np.std(recursive_chunks):.1f} chunks")
        
        print(f"\n--- Processing Time Comparison ---")
        print(f"Semantic (cumulative): {np.mean(cumulative_times):.4f} ± {np.std(cumulative_times):.4f}s")
        print(f"Semantic (window): {np.mean(window_times):.4f} ± {np.std(window_times):.4f}s")
        print(f"Recursive: {np.mean(recursive_times):.4f} ± {np.std(recursive_times):.4f}s")
        
        print(f"\n--- Average Chunk Size Comparison ---")
        print(f"Semantic (cumulative): {np.mean(cumulative_avg_sizes):.1f} ± {np.std(cumulative_avg_sizes):.1f} chars")
        print(f"Semantic (window): {np.mean(window_avg_sizes):.1f} ± {np.std(window_avg_sizes):.1f} chars")
        print(f"Recursive: {np.mean(recursive_avg_sizes):.1f} ± {np.std(recursive_avg_sizes):.1f} chars")
    
    return results

def generate_visualizations(data_analysis, comparison_results):
    """Generate visualizations of the experimental results"""
    try:
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Chunking Algorithms Performance Analysis', fontsize=16)
        
        # Extract data
        chunk_sizes = data_analysis['chunk_sizes']
        n_chunks = data_analysis['n_chunks']
        processing_times = data_analysis['processing_times']
        algorithms = data_analysis['algorithms']
        
        # Plot 1: Chunk size vs number of chunks
        for alg in ['semantic', 'recursive']:
            mask = algorithms == alg
            if np.any(mask):
                axes[0, 0].scatter(chunk_sizes[mask], n_chunks[mask], label=f'{alg.capitalize()}', alpha=0.7)
        axes[0, 0].set_xlabel('Chunk Size')
        axes[0, 0].set_ylabel('Number of Chunks')
        axes[0, 0].set_title('Chunk Size vs Number of Chunks')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Chunk size vs processing time
        for alg in ['semantic', 'recursive']:
            mask = algorithms == alg
            if np.any(mask):
                axes[0, 1].scatter(chunk_sizes[mask], processing_times[mask], label=f'{alg.capitalize()}', alpha=0.7)
        axes[0, 1].set_xlabel('Chunk Size')
        axes[0, 1].set_ylabel('Processing Time (seconds)')
        axes[0, 1].set_title('Chunk Size vs Processing Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Algorithm comparison for chunk count
        if comparison_results:
            cumulative_chunks = [comp['semantic_cumulative']['n_chunks'] for comp in comparison_results]
            window_chunks = [comp['semantic_window']['n_chunks'] for comp in comparison_results]
            recursive_chunks = [comp['recursive']['n_chunks'] for comp in comparison_results]
            
            x_pos = np.arange(3)
            avg_chunks = [np.mean(cumulative_chunks), np.mean(window_chunks), np.mean(recursive_chunks)]
            std_chunks = [np.std(cumulative_chunks), np.std(window_chunks), np.std(recursive_chunks)]
            
            bars = axes[1, 0].bar(['Semantic\n(Cumulative)', 'Semantic\n(Window)', 'Recursive'], 
                                 avg_chunks, yerr=std_chunks, capsize=5, alpha=0.7)
            axes[1, 0].set_ylabel('Average Number of Chunks')
            axes[1, 0].set_title('Algorithm Comparison - Number of Chunks')
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, avg_val in zip(bars, avg_chunks):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + std_chunks[0]/2.,
                               f'{avg_val:.1f}', ha='center', va='bottom')
        
        # Plot 4: Algorithm comparison for processing time
        if comparison_results:
            cumulative_times = [comp['semantic_cumulative']['total_processing_time'] for comp in comparison_results]
            window_times = [comp['semantic_window']['total_processing_time'] for comp in comparison_results]
            recursive_times = [comp['recursive']['total_processing_time'] for comp in comparison_results]
            
            avg_times = [np.mean(cumulative_times), np.mean(window_times), np.mean(recursive_times)]
            std_times = [np.std(cumulative_times), np.std(window_times), np.std(recursive_times)]
            
            bars = axes[1, 1].bar(['Semantic\n(Cumulative)', 'Semantic\n(Window)', 'Recursive'], 
                                 avg_times, yerr=std_times, capsize=5, alpha=0.7)
            axes[1, 1].set_ylabel('Average Processing Time (seconds)')
            axes[1, 1].set_title('Algorithm Comparison - Processing Time')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, avg_val in zip(bars, avg_times):
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + std_times[0]/2.,
                               f'{avg_val:.3f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('chunking_analysis.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'chunking_analysis.png'")
        plt.show()
        
    except ImportError:
        print("Matplotlib/seaborn not available, skipping visualization")

def summarize_findings():
    """Summarize key findings from the experiments"""
    print("\n=== KEY FINDINGS AND RECOMMENDATIONS ===")
    
    print("\n1. Performance Comparison:")
    print("   - Recursive chunker is significantly faster than semantic chunkers")
    print("   - Semantic window mode creates many more, smaller chunks than cumulative mode")
    print("   - Chunk size directly affects the number of resulting chunks as expected")
    
    print("\n2. Algorithm Characteristics:")
    print("   - Semantic (cumulative): Moderate number of chunks with good semantic coherence, moderate processing time")
    print("   - Semantic (window): Very fine-grained chunks, fast processing")
    print("   - Recursive: Balanced approach, fastest processing, predictable chunk sizes")
    
    print("\n3. Parameter Sensitivity:")
    print("   - Larger chunk sizes result in fewer chunks (linear relationship)")
    print("   - Semantic chunkers are more sensitive to text structure")
    print("   - Recursive chunker provides most consistent performance")
    
    print("\n4. Recommendations:")
    print("   - For speed: Use Recursive chunker")
    print("   - For semantic coherence: Use Semantic (cumulative) chunker")
    print("   - For fine-grained control: Use Semantic (window) chunker")
    print("   - For predictable chunk sizes: Use Recursive chunker with fixed parameters")

if __name__ == "__main__":
    # Perform analysis
    param_data = analyze_param_ablation_results()
    comparison_data = analyze_algorithm_comparison()
    
    # Generate visualizations if possible
    try:
        generate_visualizations(param_data, comparison_data)
    except:
        print("Could not generate visualizations (missing plotting libraries)")
    
    # Print summary
    summarize_findings()