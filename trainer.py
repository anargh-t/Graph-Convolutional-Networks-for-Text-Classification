"""
TextGCN Training Script

This script implements the training pipeline for Graph Convolutional Networks
for Text Classification. It loads preprocessed graphs, trains GCN models,
and evaluates performance on text classification tasks.

Main components:
- PrepareData: Loads and preprocesses graph data
- TextGCNTrainer: Handles model training and evaluation
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
from utils import parameter_parser
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
    Data preparation class for TextGCN training.
    
    This class handles loading and preprocessing of graph data, features,
    and labels for the TextGCN model. It creates the necessary data structures
    for training including adjacency matrices, feature matrices, and label vectors.
    """
    
    def __init__(self, args):
        """
        Initialize data preparation.
        
        Args:
            args: Command line arguments containing dataset name and other parameters
        """
        print("prepare data")
        self.graph_path = "data/graph"
        self.args = args

        # Load the pre-built graph from file
        # The graph contains document-word and word-word edges with weights
        graph = nx.read_weighted_edgelist(f"{self.graph_path}/{args.dataset}.txt"
                                          , nodetype=int)
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
        
        # In TextGCN, we use identity matrix as features
        # Each node (document or word) has a unique one-hot feature vector
        # This is a common approach in graph neural networks when no additional features are available
        self.nfeat_dim = graph.number_of_nodes()
        
        # Create identity matrix as sparse tensor
        # Row indices: [0, 1, 2, ..., n-1]
        row = list(range(self.nfeat_dim))
        # Column indices: [0, 1, 2, ..., n-1] (same as row for identity matrix)
        col = list(range(self.nfeat_dim))
        # Values: all 1.0 (diagonal elements)
        value = [1.] * self.nfeat_dim
        shape = (self.nfeat_dim, self.nfeat_dim)
        
        # Convert to PyTorch sparse tensor format
        indices = th.from_numpy(
                np.vstack((row, col)).astype(np.int64))
        values = th.FloatTensor(value)
        shape = th.Size(shape)

        self.features = th.sparse.FloatTensor(indices, values, shape)

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


class TextGCNTrainer:
    """
    TextGCN Model Training and Evaluation Class
    
    This class handles the complete training pipeline for the TextGCN model,
    including model initialization, training loop, validation, and testing.
    It implements early stopping, logging, and performance evaluation.
    """
    
    def __init__(self, args, model, pre_data):
        """
        Initialize the TextGCN trainer.
        
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


def main(dataset, times):
    args = parameter_parser()
    args.dataset = dataset

    args.device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
    args.nhid = 200
    args.max_epoch = 200
    args.dropout = 0.5
    args.val_ratio = 0.1
    args.early_stopping = 10
    args.lr = 0.02
    model = GCN

    print(args)

    predata = PrepareData(args)
    cudause = CudaUse()

    record = LogResult()
    seed_lst = list()
    for ind, seed in enumerate(return_seed(times)):
        print(f"\n\n==> {ind}, seed:{seed}")
        args.seed = seed
        seed_lst.append(seed)

        framework = TextGCNTrainer(model=model, args=args, pre_data=predata)
        framework.fit()

        if th.cuda.is_available():
            gpu_mem = cudause.gpu_mem_get(_id=0)
            record.log_single(key="gpu_mem", value=gpu_mem)

        record.log(framework.test())

        del framework
        gc.collect()

        if th.cuda.is_available():
            th.cuda.empty_cache()

    print("==> seed set:")
    print(seed_lst)
    record.show_str()


if __name__ == '__main__':
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # for d in ["mr", "ohsumed", "R52", "R8", "20ng"]:
    #     main(d)
    main("R8", 1)  # Run R8 dataset
    # main("ohsumed")
    # main("R8", 1)
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
