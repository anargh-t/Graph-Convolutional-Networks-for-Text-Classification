"""
TopicGCN Training Script

This script implements the training pipeline for Topic-aware Graph
Convolutional Networks. It loads the document-topic graph, builds topic-based
features, trains the GCN model, and evaluates performance.

Main components:
- PrepareData: Loads graph data, features, and labels
- TopicGCNTrainer: Handles model training and evaluation
- Main function: Orchestrates the training process
"""

import gc
import warnings
from time import time

import networkx as nx
import numpy as np
import pandas as pd
import torch as th
from sklearn.model_selection import train_test_split

# Import custom modules
from layer import GCN
from utils import accuracy
from utils import macro_f1
from utils import CudaUse
from utils import EarlyStopping
from utils import LogResult
from utils import preprocess_adj
from utils import print_graph_detail
from utils import read_file
from utils import return_seed

# Configure PyTorch for reproducibility and performance
th.backends.cudnn.deterministic = True  # Ensure reproducible results
th.backends.cudnn.benchmark = True      # Optimize for consistent input sizes
warnings.filterwarnings("ignore")       # Suppress warnings for cleaner output


def get_train_test(target_fn):
    """
    Parse the dataset file to separate training and test samples.
    
    Args:
        target_fn (str): Path to the dataset file containing labels and splits
        
    Returns:
        tuple: (train_lst, test_lst) - Lists of indices for train and test samples
        
    Note:
        The dataset file format is: index\t{split}\t{label}
        Split can be "train", "training", "20news-bydate-train" for training samples
    """
    train_lst = list()
    test_lst = list()
    
    # Read the dataset file line by line
    with read_file(target_fn, mode="r") as fin:
        for indx, item in enumerate(fin):
            # Split each line: index\t{split}\t{label}
            split_info = item.split("\t")[1]
            
            # Check if this is a training sample
            if split_info in {"train", "training", "20news-bydate-train"}:
                train_lst.append(indx)
            else:
                test_lst.append(indx)

    return train_lst, test_lst


class PrepareData:
    """
    Data preparation class for TopicGCN training.
    
    This class loads the pre-built document-topic graph, constructs the adjacency
    matrix, builds topic-aware node features, and prepares labels/splits for the
    GCN model.
    """
    
    def __init__(self, args):
        """
        Initialize data preparation.
        
        Args:
            args: Namespace containing dataset name and training hyperparameters.
        """
        print("prepare data")
        self.graph_path = "data/graph"
        self.args = args
        
        graph_filename = f"{self.graph_path}/{args.dataset}_topic.txt"
        print(f"Loading TopicGCN graph: {graph_filename}")

        # Load the pre-built graph from file
        graph = nx.read_weighted_edgelist(graph_filename, nodetype=int)
        print_graph_detail(graph)
        try:
            # Try the old method first
            adj = nx.to_scipy_sparse_matrix(graph,
                                            nodelist=list(range(graph.number_of_nodes())),
                                            weight='weight',
                                            dtype=np.float32)  # Use float32 to save memory
        except AttributeError:
            try:
                # For newer NetworkX versions, use adjacency_matrix
                adj = nx.adjacency_matrix(graph,
                                          nodelist=list(range(graph.number_of_nodes())),
                                          weight='weight',
                                          dtype=np.float32)  # Use float32 to save memory
            except Exception as e:
                print(f"Error creating adjacency matrix: {e}")
                print("Trying alternative method...")
                # Alternative: create sparse matrix manually
                import scipy.sparse as sp
                n_nodes = graph.number_of_nodes()
                edges = list(graph.edges(data=True))
                
                if len(edges) == 0:
                    adj = sp.identity(n_nodes, dtype=np.float32)
                else:
                    # Create COO matrix manually
                    row_indices = []
                    col_indices = []
                    data = []
                    
                    for edge in edges:
                        u, v, weight_data = edge
                        weight = weight_data.get('weight', 1.0)
                        row_indices.append(u)
                        col_indices.append(v)
                        data.append(weight)
                        
                        # Add symmetric edge
                        if u != v:
                            row_indices.append(v)
                            col_indices.append(u)
                            data.append(weight)
                    
                    adj = sp.coo_matrix((data, (row_indices, col_indices)), 
                                       shape=(n_nodes, n_nodes), 
                                       dtype=np.float32)

        # Ensure adjacency matrix is symmetric (undirected graph)
        # This operation ensures A[i,j] = A[j,i] by taking the maximum of both values
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        # Preprocess adjacency matrix (normalize and convert to PyTorch sparse tensor)
        self.adj = preprocess_adj(adj, is_sparse=True)

        # ========== FEATURE MATRIX CONSTRUCTION ==========
        self._build_topic_features(graph)
    
    def _build_topic_features(self, graph):
        """
        Build feature matrix for TopicGCN mode.
        
        Document nodes: Use topic distribution vector θ_d (dimension K)
        Topic nodes: Use topic embedding vector (dimension = embedding size)
        """
        from topic_model import TopicModel
        
        print("\n==> Building topic-based features <==")
        
        # Load topic model
        topic_model_path = f"{self.graph_path}/{self.args.dataset}_topic_model.pkl"
        topic_model = TopicModel(num_topics=getattr(self.args, 'num_topics', 50))
        topic_model.load(topic_model_path)
        
        num_topics = topic_model.num_topics
        
        # Get document-topic distribution
        clean_corpus_path = "data/text_dataset/clean_corpus"
        content_path = f"{clean_corpus_path}/{self.args.dataset}.txt"
        from topic_model import load_documents_from_file
        documents = load_documents_from_file(content_path)
        doc_topic_dist = topic_model.get_document_topic_distribution(documents)
        
        # Get topic embeddings
        if topic_model.topic_embeddings is None:
            topic_model.get_topic_embeddings(top_n=20)
        topic_embeddings = topic_model.topic_embeddings
        
        num_docs = doc_topic_dist.shape[0]
        embedding_dim = topic_embeddings.shape[1]
        
        print(f"Number of documents: {num_docs}")
        print(f"Number of topics: {num_topics}")
        print(f"Topic embedding dimension: {embedding_dim}")
        
        # Determine feature dimension
        # For documents: use topic distribution (K dimensions)
        # For topics: use embeddings (embedding_dim dimensions)
        # We'll pad to the maximum dimension for compatibility
        self.nfeat_dim = max(num_topics, embedding_dim)
        
        # Build feature matrix
        # Documents: [0, num_docs-1] -> topic distributions
        # Topics: [num_docs, num_docs+num_topics-1] -> embeddings
        feature_matrix = np.zeros((num_docs + num_topics, self.nfeat_dim), dtype=np.float32)
        
        # Document features: topic distributions (normalized to sum to 1)
        for doc_idx in range(num_docs):
            topic_dist = doc_topic_dist[doc_idx]
            # Normalize to ensure sum = 1
            topic_dist = topic_dist / (topic_dist.sum() + 1e-8)
            feature_matrix[doc_idx, :num_topics] = topic_dist
        
        # Topic features: embeddings
        for topic_idx in range(num_topics):
            if embedding_dim <= self.nfeat_dim:
                feature_matrix[num_docs + topic_idx, :embedding_dim] = topic_embeddings[topic_idx]
            else:
                # If embedding is larger, truncate
                feature_matrix[num_docs + topic_idx, :] = topic_embeddings[topic_idx, :self.nfeat_dim]
        
        # Normalize features (L2 normalization)
        from sklearn.preprocessing import normalize
        feature_matrix = normalize(feature_matrix, norm='l2', axis=1)
        
        # Convert to PyTorch sparse tensor
        # For efficiency, convert dense to sparse
        import scipy.sparse as sp
        feature_sparse = sp.csr_matrix(feature_matrix)
        
        # Convert to COO format to get row and col indices
        feature_coo = feature_sparse.tocoo()
        row = feature_coo.row
        col = feature_coo.col
        data = feature_coo.data
        
        indices = th.from_numpy(np.vstack((row, col)).astype(np.int64))
        values = th.FloatTensor(data)
        shape = th.Size(feature_matrix.shape)
        
        self.features = th.sparse.FloatTensor(indices, values, shape)
        
        print(f"Feature matrix shape: {self.features.shape}")
        print(f"Feature dimension: {self.nfeat_dim}")

        # ========== LABEL PROCESSING ==========
        
        # Load labels from the dataset file
        # Format: index\t{split}\t{label}
        target_fn = f"data/text_dataset/{self.args.dataset}.txt"
        target = np.array(pd.read_csv(target_fn,
                                      sep="\t",
                                      header=None)[2])  # Column 2 contains labels
        
        # Create label-to-ID mapping for classification
        # Convert string labels to integer IDs for neural network training
        target2id = {label: indx for indx, label in enumerate(set(target))}
        self.target = [target2id[label] for label in target]
        self.nclass = len(target2id)  # Number of unique classes

        # ========== TRAIN/TEST SPLIT ==========
        
        # Separate training and test samples based on the split information in the dataset
        self.train_lst, self.test_lst = get_train_test(target_fn)


class TopicGCNTrainer:
    """
    TopicGCN Model Training and Evaluation Class
    
    This class handles the complete training pipeline for the TopicGCN model,
    including model initialization, training loop, validation, and testing.
    It implements early stopping, logging, and performance evaluation.
    """
    
    def __init__(self, args, model, pre_data):
        """
        Initialize the TopicGCN trainer.
        
        Args:
            args: Command line arguments with hyperparameters
            model: The GCN model class to instantiate
            pre_data: Prepared data object containing graph, features, and labels
        """
        self.args = args
        self.model = model
        self.device = args.device

        # Training hyperparameters
        self.max_epoch = self.args.max_epoch
        self.set_seed()  # Set random seeds for reproducibility

        self.dataset = args.dataset
        self.predata = pre_data
        self.earlystopping = EarlyStopping(args.early_stopping)  # Early stopping mechanism

    def set_seed(self):
        th.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)

    def fit(self):
        self.prepare_data()
        self.model = self.model(nfeat=self.nfeat_dim,
                                nhid=self.args.nhid,
                                nclass=self.nclass,
                                dropout=self.args.dropout)
        print(self.model.parameters)
        self.model = self.model.to(self.device)

        self.optimizer = th.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = th.nn.CrossEntropyLoss()

        self.model_param = sum(param.numel() for param in self.model.parameters())
        print('# model parameters:', self.model_param)
        self.convert_tensor()

        start = time()
        self.train()
        self.train_time = time() - start

    @classmethod
    def set_description(cls, desc):
        string = ""
        for key, value in desc.items():
            if isinstance(value, int):
                string += f"{key}:{value} "
            else:
                string += f"{key}:{value:.4f} "
        print(string)

    def prepare_data(self):
        self.adj = self.predata.adj
        self.nfeat_dim = self.predata.nfeat_dim
        self.features = self.predata.features
        self.target = self.predata.target
        self.nclass = self.predata.nclass

        self.train_lst, self.val_lst = train_test_split(self.predata.train_lst,
                                                        test_size=self.args.val_ratio,
                                                        shuffle=True,
                                                        random_state=self.args.seed)
        self.test_lst = self.predata.test_lst

    def convert_tensor(self):
        self.model = self.model.to(self.device)
        self.adj = self.adj.to(self.device)
        self.features = self.features.to(self.device)
        self.target = th.tensor(self.target).long().to(self.device)
        self.train_lst = th.tensor(self.train_lst).long().to(self.device)
        self.val_lst = th.tensor(self.val_lst).long().to(self.device)

    def train(self):
        # Store training history for saving
        self.training_history = []
        
        for epoch in range(self.max_epoch):
            self.model.train()
            self.optimizer.zero_grad()

            logits = self.model.forward(self.features, self.adj)
            loss = self.criterion(logits[self.train_lst],
                                  self.target[self.train_lst])

            loss.backward()
            self.optimizer.step()

            val_desc = self.val(self.val_lst)

            desc = dict(**{"epoch"     : epoch,
                           "train_loss": loss.item(),
                           }, **val_desc)

            self.set_description(desc)
            
            # Store epoch results
            self.training_history.append(desc)

            if self.earlystopping(val_desc["val_loss"]):
                break

    @th.no_grad()
    def val(self, x, prefix="val"):
        self.model.eval()
        with th.no_grad():
            logits = self.model.forward(self.features, self.adj)
            loss = self.criterion(logits[x],
                                  self.target[x])
            acc = accuracy(logits[x],
                           self.target[x])
            f1, precision, recall = macro_f1(logits[x],
                                             self.target[x],
                                             num_classes=self.nclass)

            desc = {
                f"{prefix}_loss": loss.item(),
                "acc"           : acc,
                "macro_f1"      : f1,
                "precision"     : precision,
                "recall"        : recall,
            }
        return desc

    @th.no_grad()
    def test(self):
        self.test_lst = th.tensor(self.test_lst).long().to(self.device)
        test_desc = self.val(self.test_lst, prefix="test")
        test_desc["train_time"] = self.train_time
        test_desc["model_param"] = self.model_param
        return test_desc


def main(dataset, times, output_dir="results"):
    """
    Main training function.
    
    Args:
        dataset: Dataset name
        times: Number of training runs
        output_dir: Directory to write structured outputs/logs
    """
    # Create args namespace for downstream components
    class Args:
        pass
    args = Args()
    args.dataset = dataset
    args.output_dir = output_dir

    args.device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
    args.nhid = 200
    args.max_epoch = 200
    args.dropout = 0.5
    args.val_ratio = 0.1
    args.early_stopping = 10
    args.lr = 0.02
    model = GCN

    print(vars(args))
    print("\n==> Training in TopicGCN mode <==")

    predata = PrepareData(args)
    cudause = CudaUse()

    record = LogResult()
    seed_lst = list()
    all_training_histories = []  # Store all training histories
    
    for ind, seed in enumerate(return_seed(times)):
        print(f"\n\n==> {ind}, seed:{seed}")
        args.seed = seed
        seed_lst.append(seed)

        framework = TopicGCNTrainer(model=model, args=args, pre_data=predata)
        framework.fit()
        
        # Evaluate once per run
        test_results = framework.test()
        
        # Store training history
        all_training_histories.append({
            'seed': seed,
            'history': framework.training_history,
            'test_results': test_results
        })

        if th.cuda.is_available():
            gpu_mem = cudause.gpu_mem_get(_id=0)
            record.log_single(key="gpu_mem", value=gpu_mem)

        record.log(test_results)

        del framework
        gc.collect()

        if th.cuda.is_available():
            th.cuda.empty_cache()

    print("==> seed set:")
    print(seed_lst)
    record.show_str()
    
    # Save results to file(s)
    import os
    import json
    results_dir = output_dir
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    mode_suffix = "_topic"
    results_file = f"{results_dir}/{dataset}{mode_suffix}_training_results.txt"
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"TRAINING RESULTS: {dataset.upper()}{mode_suffix.upper()}\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Dataset: {dataset}\n")
        f.write("Mode: TopicGCN\n")
        f.write(f"Number of runs: {times}\n")
        f.write(f"Seeds used: {seed_lst}\n\n")
        
        f.write("Hyperparameters:\n")
        f.write(f"  Hidden dimension: {args.nhid}\n")
        f.write(f"  Max epochs: {args.max_epoch}\n")
        f.write(f"  Dropout: {args.dropout}\n")
        f.write(f"  Learning rate: {args.lr}\n")
        f.write(f"  Validation ratio: {args.val_ratio}\n")
        f.write(f"  Early stopping patience: {args.early_stopping}\n\n")
        
        f.write("Results Summary:\n")
        f.write("-"*80 + "\n")
        for key, value_lst in record.result.items():
            if key == 'train_time':
                value = np.mean(value_lst)
                f.write(f"{key}:\n")
                f.write(f"  Mean: {value:.4f} seconds\n")
                f.write(f"  Max: {max(value_lst):.4f} seconds\n")
                f.write(f"  Min: {min(value_lst):.4f} seconds\n\n")
            elif key == 'model_param':
                value = np.mean(value_lst)
                f.write(f"{key}:\n")
                f.write(f"  Mean: {int(value)}\n")
                f.write(f"  Max: {int(max(value_lst))}\n")
                f.write(f"  Min: {int(min(value_lst))}\n\n")
            else:
                value = np.mean(value_lst)
                f.write(f"{key}:\n")
                f.write(f"  Mean: {value:.4f}\n")
                f.write(f"  Max: {max(value_lst):.4f}\n")
                f.write(f"  Min: {min(value_lst):.4f}\n\n")
        
        # Write detailed training history for each run
        f.write("\n" + "="*80 + "\n")
        f.write("DETAILED TRAINING HISTORY\n")
        f.write("="*80 + "\n\n")
        
        for run_idx, run_data in enumerate(all_training_histories):
            f.write(f"\nRun {run_idx + 1} (Seed: {run_data['seed']}):\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Epoch':<8} {'Train Loss':<12} {'Val Loss':<12} {'Val Acc':<10} {'Val F1':<10} {'Val Prec':<10} {'Val Rec':<10}\n")
            f.write("-"*80 + "\n")
            
            for epoch_data in run_data['history']:
                epoch = epoch_data.get('epoch', 0)
                train_loss = epoch_data.get('train_loss', 0)
                val_loss = epoch_data.get('val_loss', 0)
                val_acc = epoch_data.get('acc', 0)
                val_f1 = epoch_data.get('macro_f1', 0)
                val_prec = epoch_data.get('precision', 0)
                val_rec = epoch_data.get('recall', 0)
                
                f.write(f"{epoch:<8} {train_loss:<12.4f} {val_loss:<12.4f} {val_acc:<10.4f} "
                       f"{val_f1:<10.4f} {val_prec:<10.4f} {val_rec:<10.4f}\n")
            
            # Write test results
            test_res = run_data['test_results']
            f.write("\nTest Results:\n")
            f.write(f"  Test Loss: {test_res.get('test_loss', 0):.4f}\n")
            f.write(f"  Test Accuracy: {test_res.get('acc', 0):.4f}\n")
            f.write(f"  Test Macro F1: {test_res.get('macro_f1', 0):.4f}\n")
            f.write(f"  Test Precision: {test_res.get('precision', 0):.4f}\n")
            f.write(f"  Test Recall: {test_res.get('recall', 0):.4f}\n")
            f.write(f"  Training Time: {test_res.get('train_time', 0):.2f} seconds\n")
            f.write(f"  Model Parameters: {test_res.get('model_param', 0)}\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
    
    print(f"\nResults saved to: {results_file}")
    
    # Save structured summary as JSON
    summary = {
        "dataset": dataset,
        "mode": "TopicGCN",
        "runs": times,
        "seeds": seed_lst,
        "hyperparams": {
            "hidden_dim": args.nhid,
            "max_epoch": args.max_epoch,
            "dropout": args.dropout,
            "learning_rate": args.lr,
            "val_ratio": args.val_ratio,
            "early_stopping": args.early_stopping,
        },
        "metrics": {
            key: {
                "mean": float(np.mean(vals)),
                "max": float(np.max(vals)),
                "min": float(np.min(vals)),
            } for key, vals in record.result.items()
        },
        "runs_detail": all_training_histories,
    }
    json_path = f"{results_dir}/{dataset}{mode_suffix}_training_results.json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(summary, jf, indent=2)
    print(f"Structured summary saved to: {json_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Train TopicGCN")
    parser.add_argument('--dataset', type=str, default='R8',
                       help='Dataset name (default: R8)')
    parser.add_argument('--times', type=int, default=1,
                       help='Number of training runs (default: 1)')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save training artifacts (default: results)')
    
    args = parser.parse_args()
    main(args.dataset, args.times, output_dir=args.output_dir)
