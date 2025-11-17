"""
Topic Inspection and Visualization Module

This module provides utilities for inspecting and visualizing topics learned
by the TopicGCN model. It includes functions to:
- Display top words for each topic
- Find documents closest to each topic
- Visualize topic-topic similarity heatmaps
- Analyze topic distributions across documents
"""

import os
import numpy as np
import pickle
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: Matplotlib/Seaborn not available. Visualization disabled.")

from topic_model import TopicModel, load_documents_from_file
from sklearn.metrics.pairwise import cosine_similarity


def load_topic_model(dataset: str, graph_path: str = "data/graph") -> TopicModel:
    """
    Load a saved topic model.
    
    Args:
        dataset: Dataset name
        graph_path: Path to graph directory
    
    Returns:
        Loaded TopicModel instance
    """
    topic_model_path = f"{graph_path}/{dataset}_topic_model.pkl"
    
    if not os.path.exists(topic_model_path):
        raise FileNotFoundError(f"Topic model not found: {topic_model_path}")
    
    # Load model metadata first to get num_topics
    with open(topic_model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    num_topics = model_data.get('num_topics', 50)
    topic_model = TopicModel(num_topics=num_topics)
    topic_model.load(topic_model_path)
    
    return topic_model


def print_topic_words(topic_model: TopicModel, 
                      top_n: int = 20,
                      num_topics_to_show: Optional[int] = None):
    """
    Print top words for each topic.
    
    Args:
        topic_model: Trained TopicModel instance
        top_n: Number of top words to display per topic
        num_topics_to_show: Number of topics to display (None = all)
    """
    print(f"\n{'='*80}")
    print(f"TOP WORDS FOR EACH TOPIC (Top {top_n} words)")
    print(f"{'='*80}\n")
    
    topic_words = topic_model.get_topic_word_distribution(top_n=top_n)
    num_topics = len(topic_words)
    
    if num_topics_to_show is None:
        num_topics_to_show = num_topics
    
    for topic_id in range(min(num_topics_to_show, num_topics)):
        words = topic_words[topic_id]
        print(f"Topic {topic_id:3d}: ", end="")
        word_str = ", ".join([f"{word}({prob:.3f})" for word, prob in words[:10]])
        print(word_str)
    
    print(f"\n{'='*80}\n")


def find_documents_for_topic(topic_model: TopicModel,
                             documents: List[str],
                             topic_id: int,
                             top_n: int = 10) -> List[Tuple[int, float, str]]:
    """
    Find documents most closely associated with a specific topic.
    
    Args:
        topic_model: Trained TopicModel instance
        documents: List of document strings
        topic_id: ID of the topic to analyze
        top_n: Number of top documents to return
    
    Returns:
        List of (doc_index, topic_weight, document_preview) tuples
    """
    # Get document-topic distribution
    doc_topic_dist = topic_model.get_document_topic_distribution(documents)
    
    # Get topic weights for all documents
    topic_weights = doc_topic_dist[:, topic_id]
    
    # Get top N documents
    top_indices = np.argsort(topic_weights)[-top_n:][::-1]
    
    results = []
    for idx in top_indices:
        weight = topic_weights[idx]
        doc_preview = documents[idx][:100] + "..." if len(documents[idx]) > 100 else documents[idx]
        results.append((idx, weight, doc_preview))
    
    return results


def print_documents_for_topic(topic_model: TopicModel,
                              documents: List[str],
                              topic_id: int,
                              top_n: int = 5):
    """
    Print documents most closely associated with a specific topic.
    
    Args:
        topic_model: Trained TopicModel instance
        documents: List of document strings
        topic_id: ID of the topic to analyze
        top_n: Number of top documents to display
    """
    print(f"\n{'='*80}")
    print(f"TOP {top_n} DOCUMENTS FOR TOPIC {topic_id}")
    print(f"{'='*80}\n")
    
    doc_results = find_documents_for_topic(topic_model, documents, topic_id, top_n)
    
    for rank, (doc_idx, weight, doc_preview) in enumerate(doc_results, 1):
        print(f"Rank {rank} (Weight: {weight:.4f}, Doc ID: {doc_idx}):")
        print(f"  {doc_preview}")
        print()
    
    print(f"{'='*80}\n")


def plot_topic_similarity_heatmap(topic_model: TopicModel,
                                  save_path: Optional[str] = None,
                                  figsize: Tuple[int, int] = (12, 10)):
    """
    Plot a heatmap of topic-topic similarity.
    
    Args:
        topic_model: Trained TopicModel instance
        save_path: Path to save the figure (None = display only)
        figsize: Figure size (width, height)
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Cannot plot heatmap.")
        return
    
    # Get topic embeddings
    if topic_model.topic_embeddings is None:
        topic_model.get_topic_embeddings(top_n=20)
    
    topic_embeddings = topic_model.topic_embeddings
    
    # Compute similarity matrix
    sim_matrix = cosine_similarity(topic_embeddings)
    
    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(sim_matrix, 
                cmap='viridis',
                square=True,
                cbar_kws={'label': 'Cosine Similarity'},
                xticklabels=range(topic_model.num_topics),
                yticklabels=range(topic_model.num_topics))
    
    plt.title('Topic-Topic Similarity Heatmap', fontsize=16, fontweight='bold')
    plt.xlabel('Topic ID', fontsize=12)
    plt.ylabel('Topic ID', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_topic_distribution(topic_model: TopicModel,
                               documents: List[str],
                               top_n_topics: int = 5):
    """
    Analyze topic distribution across documents.
    
    Args:
        topic_model: Trained TopicModel instance
        documents: List of document strings
        top_n_topics: Number of top topics to analyze
    """
    print(f"\n{'='*80}")
    print(f"TOPIC DISTRIBUTION ANALYSIS")
    print(f"{'='*80}\n")
    
    # Get document-topic distribution
    doc_topic_dist = topic_model.get_document_topic_distribution(documents)
    
    num_docs = doc_topic_dist.shape[0]
    num_topics = doc_topic_dist.shape[1]
    
    # Statistics per topic
    topic_stats = []
    for topic_id in range(num_topics):
        topic_weights = doc_topic_dist[:, topic_id]
        mean_weight = np.mean(topic_weights)
        max_weight = np.max(topic_weights)
        num_docs_above_threshold = np.sum(topic_weights > 0.02)
        
        topic_stats.append({
            'topic_id': topic_id,
            'mean_weight': mean_weight,
            'max_weight': max_weight,
            'num_docs': num_docs_above_threshold
        })
    
    # Sort by mean weight
    topic_stats.sort(key=lambda x: x['mean_weight'], reverse=True)
    
    print(f"Top {top_n_topics} topics by average document weight:\n")
    for i, stats in enumerate(topic_stats[:top_n_topics], 1):
        print(f"{i}. Topic {stats['topic_id']:3d}: "
              f"Mean={stats['mean_weight']:.4f}, "
              f"Max={stats['max_weight']:.4f}, "
              f"Docs above threshold={stats['num_docs']}")
    
    # Document statistics
    print(f"\nDocument statistics:")
    print(f"  Total documents: {num_docs}")
    print(f"  Average topics per document (weight > 0.02): "
          f"{np.mean([np.sum(doc_topic_dist[i] > 0.02) for i in range(num_docs)]):.2f}")
    
    print(f"\n{'='*80}\n")


def inspect_topics(dataset: str,
                   graph_path: str = "data/graph",
                   clean_corpus_path: str = "data/text_dataset/clean_corpus",
                   top_n_words: int = 20,
                   top_n_docs: int = 5,
                   plot_heatmap: bool = True,
                   heatmap_save_path: Optional[str] = None,
                   save_to_file: bool = True,
                   output_dir: str = "results"):
    """
    Complete topic inspection pipeline.
    
    Args:
        dataset: Dataset name
        graph_path: Path to graph directory
        clean_corpus_path: Path to clean corpus directory
        top_n_words: Number of top words to display per topic
        top_n_docs: Number of top documents to display per topic
        plot_heatmap: Whether to plot topic similarity heatmap
        heatmap_save_path: Path to save heatmap (None = display only)
        save_to_file: Whether to save inspection results to a text file
        output_dir: Directory to store inspection artifacts
    """
    print(f"\n{'='*80}")
    print(f"TOPIC INSPECTION FOR DATASET: {dataset.upper()}")
    print(f"{'='*80}\n")
    
    # Prepare output file if saving
    output_lines = []
    if save_to_file:
        output_lines.append("="*80 + "\n")
        output_lines.append(f"TOPIC INSPECTION FOR DATASET: {dataset.upper()}\n")
        output_lines.append("="*80 + "\n\n")
    
    # Load topic model
    print("Loading topic model...")
    topic_model = load_topic_model(dataset, graph_path)
    
    # Load documents
    print("Loading documents...")
    content_path = f"{clean_corpus_path}/{dataset}.txt"
    documents = load_documents_from_file(content_path)
    print(f"Loaded {len(documents)} documents\n")
    
    # Print topic words and capture output
    import io
    import sys
    
    # Capture print output for topic words
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    print_topic_words(topic_model, top_n=top_n_words)
    topic_words_output = buffer.getvalue()
    sys.stdout = old_stdout
    print(topic_words_output)
    if save_to_file:
        output_lines.append(topic_words_output)
    
    # Capture topic distribution analysis
    sys.stdout = buffer = io.StringIO()
    analyze_topic_distribution(topic_model, documents)
    distribution_output = buffer.getvalue()
    sys.stdout = old_stdout
    print(distribution_output)
    if save_to_file:
        output_lines.append(distribution_output)
    
    # Print documents for top topics
    print("Documents for top 3 topics:\n")
    if save_to_file:
        output_lines.append("Documents for top 3 topics:\n\n")
    
    for topic_id in range(min(3, topic_model.num_topics)):
        sys.stdout = buffer = io.StringIO()
        print_documents_for_topic(topic_model, documents, topic_id, top_n=top_n_docs)
        doc_output = buffer.getvalue()
        sys.stdout = old_stdout
        print(doc_output)
        if save_to_file:
            output_lines.append(doc_output)
    
    # Plot heatmap if requested
    if plot_heatmap and MATPLOTLIB_AVAILABLE:
        if heatmap_save_path is None:
            os.makedirs(output_dir, exist_ok=True)
            heatmap_save_path = f"{output_dir}/{dataset}_topic_heatmap.png"
        print("Generating topic similarity heatmap...")
        plot_topic_similarity_heatmap(topic_model, save_path=heatmap_save_path)
        if save_to_file:
            output_lines.append(f"Heatmap saved to: {heatmap_save_path}\n")
    
    print(f"\n{'='*80}")
    print("Topic inspection complete!")
    print(f"{'='*80}\n")
    
    # Save to file if requested
    if save_to_file:
        results_dir = output_dir
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        output_file = f"{results_dir}/{dataset}_topic_inspection.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(output_lines)
            f.write("="*80 + "\n")
            f.write("Topic inspection complete!\n")
            f.write("="*80 + "\n")
        
        print(f"Inspection results saved to: {output_file}")


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Inspect topics from TopicGCN model")
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., R8, 20ng)')
    parser.add_argument('--graph_path', type=str, default='data/graph',
                       help='Path to graph directory (default: data/graph)')
    parser.add_argument('--corpus_path', type=str, default='data/text_dataset/clean_corpus',
                       help='Path to clean corpus directory (default: data/text_dataset/clean_corpus)')
    parser.add_argument('--top_n_words', type=int, default=20,
                       help='Number of top words to display per topic (default: 20)')
    parser.add_argument('--top_n_docs', type=int, default=5,
                       help='Number of top documents to display per topic (default: 5)')
    parser.add_argument('--no_heatmap', action='store_true',
                       help='Skip plotting topic similarity heatmap')
    parser.add_argument('--heatmap_path', type=str, default=None,
                       help='Path to save heatmap (default: graph_path/dataset_topic_heatmap.png)')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save inspection results to file')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save inspection artifacts (default: results)')
    
    args = parser.parse_args()
    
    inspect_topics(
        dataset=args.dataset,
        graph_path=args.graph_path,
        clean_corpus_path=args.corpus_path,
        top_n_words=args.top_n_words,
        top_n_docs=args.top_n_docs,
        plot_heatmap=not args.no_heatmap,
        heatmap_save_path=args.heatmap_path,
        save_to_file=not args.no_save,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()

