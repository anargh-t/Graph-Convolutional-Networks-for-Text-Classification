"""
TextGCN Neural Network Layers

This module implements the Graph Convolutional Network (GCN) layers used in TextGCN.
The implementation follows the standard GCN architecture from:
"Semi-Supervised Classification with Graph Convolutional Networks" (Kipf & Welling, 2017)

Key components:
- GraphConvolution: Single GCN layer implementation
- GCN: Two-layer GCN model for text classification

The GCN layers perform message passing on the heterogeneous graph where:
- Nodes represent documents and words
- Edges represent document-word and word-word relationships
- Features are learned through graph convolution operations
"""

import math
import torch as th

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Graph Convolutional Layer implementation.
    
    This layer implements the standard graph convolution operation:
    H^(l+1) = σ(D̃^(-1/2) Ã D̃^(-1/2) H^(l) W^(l))
    
    Where:
    - H^(l) is the node feature matrix at layer l
    - Ã = A + I (adjacency matrix with self-loops)
    - D̃ is the degree matrix of Ã
    - W^(l) is the learnable weight matrix
    - σ is the activation function (applied in the GCN model)
    
    Reference: "Semi-Supervised Classification with Graph Convolutional Networks" (Kipf & Welling, 2017)
    """

    def __init__(self, in_features, out_features, bias=True):
        """
        Initialize the graph convolution layer.
        
        Args:
            in_features (int): Number of input features per node
            out_features (int): Number of output features per node
            bias (bool): Whether to use bias term
        """
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Learnable weight matrix: transforms features from in_features to out_features
        self.weight = Parameter(th.FloatTensor(in_features, out_features))
        
        # Optional bias term
        if bias:
            self.bias = Parameter(th.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize layer parameters using Xavier/Glorot initialization.
        
        The weights are initialized uniformly in the range [-stdv, stdv]
        where stdv = 1/sqrt(out_features). This helps with gradient flow.
        """
        # Calculate standard deviation for uniform initialization
        stdv = 1. / math.sqrt(self.weight.size(1))
        
        # Initialize weight matrix uniformly
        self.weight.data.uniform_(-stdv, stdv)
        
        # Initialize bias uniformly (if present)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, infeatn, adj):
        """
        Forward pass of the graph convolution layer.
        
        The forward pass performs:
        1. Linear transformation: support = H^(l) W^(l)
        2. Graph convolution: output = Ã support
        3. Add bias: output = output + bias (if present)
        
        Args:
            infeatn (torch.sparse.FloatTensor): Input node features H^(l)
            adj (torch.sparse.FloatTensor): Normalized adjacency matrix Ã
            
        Returns:
            torch.Tensor: Output node features H^(l+1)
        """
        # Step 1: Linear transformation of input features
        # This applies the learnable weight matrix to transform features
        support = th.spmm(infeatn, self.weight)
        
        # Step 2: Graph convolution - aggregate neighbor information
        # This performs the actual graph convolution operation
        output = th.spmm(adj, support)
        
        # Step 3: Add bias term (if present)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        """
        String representation of the layer.
        
        Returns:
            str: Layer name with input and output dimensions
        """
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(Module):
    """
    Two-layer Graph Convolutional Network for Text Classification.
    
    This is the main TextGCN model that combines two graph convolution layers
    with ReLU activation and dropout for text classification tasks.
    
    Architecture:
    - Input: Node features (identity matrix for TextGCN)
    - Layer 1: GraphConvolution(nfeat -> nhid) + ReLU + Dropout
    - Layer 2: GraphConvolution(nhid -> nclass)
    - Output: Class logits for each node
    
    The model learns to classify documents by propagating information through
    the heterogeneous graph containing documents and words.
    """
    
    def __init__(self, nfeat, nhid, nclass, dropout):
        """
        Initialize the two-layer GCN model.
        
        Args:
            nfeat (int): Number of input features per node
            nhid (int): Number of hidden features per node
            nclass (int): Number of output classes
            dropout (float): Dropout probability for regularization
        """
        super(GCN, self).__init__()
        
        # First graph convolution layer: input features -> hidden features
        self.gc1 = GraphConvolution(nfeat, nhid)
        
        # Second graph convolution layer: hidden features -> output classes
        self.gc2 = GraphConvolution(nhid, nclass)
        
        # Dropout rate for regularization
        self.dropout = dropout

    def forward(self, x, adj):
        """
        Forward pass of the two-layer GCN model.
        
        The forward pass performs:
        1. First GCN layer: H^(1) = ReLU(Ã H^(0) W^(1))
        2. Dropout: H^(1) = Dropout(H^(1))
        3. Second GCN layer: H^(2) = Ã H^(1) W^(2)
        
        Args:
            x (torch.sparse.FloatTensor): Input node features H^(0)
            adj (torch.sparse.FloatTensor): Normalized adjacency matrix Ã
            
        Returns:
            torch.Tensor: Output logits for classification H^(2)
        """
        # First graph convolution layer with ReLU activation
        x = self.gc1(x, adj)        # Graph convolution: nfeat -> nhid
        x = th.relu(x)              # ReLU activation for non-linearity
        
        # Apply dropout for regularization (only during training)
        x = th.dropout(x, self.dropout, train=self.training)
        
        # Second graph convolution layer (no activation for final output)
        x = self.gc2(x, adj)        # Graph convolution: nhid -> nclass
        
        return x  # Return class logits
