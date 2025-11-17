"""
TopicGCN Graph Construction Script

This script builds a Document-Topic-Topic graph for TopicGCN. The graph
contains:
- Document nodes: indices [0, num_docs)
- Topic nodes: indices [num_docs, num_docs + num_topics)
- Document-topic edges with weights from the LDA topic distribution (θ_dk)
- Topic-topic edges with weights given by cosine similarity of topic embeddings

The workflow:
1. Load the cleaned corpus for the target dataset
2. Fit an LDA topic model (optionally train Word2Vec for embeddings)
3. Create document-topic edges filtered by θ_dk threshold
4. Create topic-topic edges filtered by cosine similarity threshold
5. Save the resulting weighted graph and the trained topic model
"""

import os
from typing import List, Optional

import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from topic_model import TopicModel, load_documents_from_file
from utils import print_graph_detail


class TopicGraphBuilder:
    """
    Build a Document-Topic-Topic graph for a specific dataset.

    Args:
        dataset: Dataset name (e.g., "mr", "20ng", "R8")
        num_topics: Number of topics (K) for LDA
        doc_topic_threshold: Minimum θ_dk to keep a document-topic edge
        topic_topic_threshold: Minimum cosine similarity to keep a topic-topic edge
        use_word2vec: Whether to train Word2Vec for topic embeddings
        min_df: Minimum document frequency for LDA vocabulary
        max_df: Maximum document frequency (fraction) for LDA vocabulary
    """

    def __init__(
        self,
        dataset: str,
        num_topics: int = 50,
        doc_topic_threshold: float = 0.02,
        topic_topic_threshold: float = 0.3,
        use_word2vec: bool = True,
        min_df: int = 2,
        max_df: float = 0.95,
    ):
        self.dataset = dataset
        self.num_topics = num_topics
        self.doc_topic_threshold = doc_topic_threshold
        self.topic_topic_threshold = topic_topic_threshold
        self.use_word2vec = use_word2vec
        self.min_df = min_df
        self.max_df = max_df

        self.clean_corpus_path = "data/text_dataset/clean_corpus"
        self.graph_path = "data/graph"
        os.makedirs(self.graph_path, exist_ok=True)

        self.documents = load_documents_from_file(
            os.path.join(self.clean_corpus_path, f"{dataset}.txt")
        )
        self.num_docs = len(self.documents)

        print(f"\n==> Building TopicGCN graph for dataset: {dataset} <==")
        print(f"Documents: {self.num_docs}")
        print(f"Topics: {num_topics}")
        print(f"Document-topic threshold: {doc_topic_threshold}")
        print(f"Topic-topic threshold: {topic_topic_threshold}")

        self.graph = nx.Graph()
        self.topic_model = TopicModel(num_topics=num_topics, random_state=42)

        self._build_topic_model()
        self._add_document_topic_edges()
        self._add_topic_topic_edges()
        self._save_outputs()
    
    def _build_topic_model(self) -> None:
        print("\n==> Fitting LDA model <==")
        self.topic_model.fit(self.documents, min_df=self.min_df, max_df=self.max_df)

        if self.use_word2vec:
            self.topic_model.fit_word2vec(self.documents, vector_size=100)

        self.topic_model.get_topic_embeddings(top_n=20)
        self.doc_topic_dist = self.topic_model.get_document_topic_distribution(self.documents)
        self.topic_embeddings = self.topic_model.topic_embeddings

        print(f"Document-topic matrix: {self.doc_topic_dist.shape}")
        print(f"Topic embedding matrix: {self.topic_embeddings.shape}")

    def _add_document_topic_edges(self) -> None:
        print("\n==> Adding document-topic edges <==")
        edge_count = 0

        for doc_idx in tqdm(range(self.num_docs), desc="Document-topic"):
            for topic_idx in range(self.num_topics):
                weight = self.doc_topic_dist[doc_idx, topic_idx]
                if weight < self.doc_topic_threshold:
                    continue

                topic_node_idx = self.num_docs + topic_idx
                self.graph.add_edge(doc_idx, topic_node_idx, weight=float(weight))
                edge_count += 1

        print(f"Document-topic edges: {edge_count}")
        print_graph_detail(self.graph)

    def _add_topic_topic_edges(self) -> None:
        print("\n==> Adding topic-topic edges <==")
        similarity_matrix = cosine_similarity(self.topic_embeddings)

        edge_count = 0
        for i in tqdm(range(self.num_topics), desc="Topic-topic"):
            for j in range(i + 1, self.num_topics):
                similarity = similarity_matrix[i, j]
                if similarity <= self.topic_topic_threshold:
                    continue

                topic_i = self.num_docs + i
                topic_j = self.num_docs + j
                self.graph.add_edge(topic_i, topic_j, weight=float(similarity))
                edge_count += 1

        print(f"Topic-topic edges: {edge_count}")
        print_graph_detail(self.graph)

    def _export_protege_csvs(self) -> None:
        """
        Export nodes and edges as CSV files for Protégé visualization.
        
        Creates two files:
        - {dataset}_topic_nodes.csv: node_id, node_type, label
        - {dataset}_topic_edges.csv: source_id, target_id, edge_type, weight
        """
        import csv
        
        nodes_file = os.path.join(self.graph_path, f"{self.dataset}_topic_nodes.csv")
        edges_file = os.path.join(self.graph_path, f"{self.dataset}_topic_edges.csv")
        
        # Get topic words for labels
        topic_words_dict = self.topic_model.get_topic_word_distribution(top_n=10)
        
        # Export nodes
        with open(nodes_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['node_id', 'node_type', 'label'])
            
            # Document nodes
            for doc_idx in range(self.num_docs):
                # Create a short label from document (first 50 chars)
                doc_label = self.documents[doc_idx][:50].replace('\n', ' ').replace(',', ' ')
                if len(self.documents[doc_idx]) > 50:
                    doc_label += '...'
                writer.writerow([doc_idx, 'document', f'Doc_{doc_idx}: {doc_label}'])
            
            # Topic nodes
            for topic_idx in range(self.num_topics):
                topic_node_idx = self.num_docs + topic_idx
                top_words = [word for word, _ in topic_words_dict[topic_idx][:5]]
                topic_label = f'Topic_{topic_idx}: {", ".join(top_words)}'
                writer.writerow([topic_node_idx, 'topic', topic_label])
        
        # Export edges
        with open(edges_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['source_id', 'target_id', 'edge_type', 'weight'])
            
            for u, v, data in self.graph.edges(data=True):
                weight = data.get('weight', 1.0)
                
                # Determine edge type
                if u < self.num_docs and v >= self.num_docs:
                    edge_type = 'doc-topic'
                elif u >= self.num_docs and v < self.num_docs:
                    edge_type = 'doc-topic'
                elif u >= self.num_docs and v >= self.num_docs:
                    edge_type = 'topic-topic'
                else:
                    edge_type = 'doc-doc'  # Should not happen in our graph
                
                writer.writerow([u, v, edge_type, f'{weight:.6f}'])
        
        print(f"Protégé CSV files exported:")
        print(f"  Nodes: {nodes_file}")
        print(f"  Edges: {edges_file}")

    def _save_outputs(self) -> None:
        graph_file = os.path.join(self.graph_path, f"{self.dataset}_topic.txt")
        model_file = os.path.join(self.graph_path, f"{self.dataset}_topic_model.pkl")

        nx.write_weighted_edgelist(self.graph, graph_file)
        self.topic_model.save(model_file)
        self._export_protege_csvs()

        print(f"\nGraph saved to: {graph_file}")
        print(f"Topic model saved to: {model_file}")
        print(f"Total nodes: {self.graph.number_of_nodes()}")
        print(f"Total edges: {self.graph.number_of_edges()}\n")


def build_graphs(
    datasets: Optional[List[str]] = None,
    num_topics: int = 50,
    doc_topic_threshold: float = 0.02,
    topic_topic_threshold: float = 0.3,
    use_word2vec: bool = True,
    min_df: int = 2,
    max_df: float = 0.95,
) -> None:
    target_datasets = datasets or ["mr", "ohsumed", "R52", "R8", "20ng"]

    for dataset in target_datasets:
        try:
            TopicGraphBuilder(
                dataset=dataset,
                num_topics=num_topics,
                doc_topic_threshold=doc_topic_threshold,
                topic_topic_threshold=topic_topic_threshold,
                use_word2vec=use_word2vec,
                min_df=min_df,
                max_df=max_df,
            )
        except Exception as exc:
            print(f"Error building graph for {dataset}: {exc}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Document-Topic-Topic graphs for TopicGCN")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (default: build all supported datasets)")
    parser.add_argument("--num_topics", type=int, default=50,
                        help="Number of topics (default: 50)")
    parser.add_argument("--doc_topic_threshold", type=float, default=0.02,
                        help="Minimum θ_dk to keep document-topic edge (default: 0.02)")
    parser.add_argument("--topic_topic_threshold", type=float, default=0.3,
                        help="Minimum cosine similarity for topic-topic edge (default: 0.3)")
    parser.add_argument("--no_word2vec", action="store_true",
                        help="Disable Word2Vec embeddings (use topic-word distributions instead)")
    parser.add_argument("--min_df", type=int, default=2,
                        help="Minimum document frequency for LDA vocabulary (default: 2)")
    parser.add_argument("--max_df", type=float, default=0.95,
                        help="Maximum document frequency fraction for LDA vocabulary (default: 0.95)")

    cli_args = parser.parse_args()

    datasets = [cli_args.dataset] if cli_args.dataset else None
    build_graphs(
        datasets=datasets,
        num_topics=cli_args.num_topics,
        doc_topic_threshold=cli_args.doc_topic_threshold,
        topic_topic_threshold=cli_args.topic_topic_threshold,
        use_word2vec=not cli_args.no_word2vec,
        min_df=cli_args.min_df,
        max_df=cli_args.max_df,
    )
