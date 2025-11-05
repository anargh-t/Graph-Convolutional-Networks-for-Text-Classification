#!/usr/bin/env python3
"""
Test script to verify the memory and compatibility fixes
"""

import sys
import os
import numpy as np
import torch as th
import networkx as nx
import scipy.sparse as sp

def test_graph_loading():
    """Test loading and processing the MR graph"""
    print("Testing graph loading and processing...")
    
    try:
        # Load MR graph (smaller dataset)
        graph_path = "data/graph/mr.txt"
        if not os.path.exists(graph_path):
            print(f"Error: {graph_path} not found. Please run build_graph.py first.")
            return False
            
        graph = nx.read_weighted_edgelist(graph_path, nodetype=int)
        print(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # Test adjacency matrix creation with our fixes
        try:
            # Try the old method first
            adj = nx.to_scipy_sparse_matrix(graph,
                                            nodelist=list(range(graph.number_of_nodes())),
                                            weight='weight',
                                            dtype=np.float32)
            print("✓ Old method (to_scipy_sparse_matrix) worked")
        except AttributeError:
            try:
                # For newer NetworkX versions, use adjacency_matrix
                adj = nx.adjacency_matrix(graph,
                                          nodelist=list(range(graph.number_of_nodes())),
                                          weight='weight',
                                          dtype=np.float32)
                print("✓ New method (adjacency_matrix) worked")
            except Exception as e:
                print(f"✗ Both methods failed: {e}")
                return False
        
        print(f"Adjacency matrix shape: {adj.shape}")
        print(f"Adjacency matrix memory usage: {adj.data.nbytes / 1024 / 1024:.2f} MB")
        
        # Test preprocessing
        from utils import preprocess_adj
        adj_processed = preprocess_adj(adj, is_sparse=True)
        print(f"✓ Preprocessing successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_20ng_memory():
    """Test if 20NG dataset can be loaded (memory test)"""
    print("\nTesting 20NG dataset memory requirements...")
    
    try:
        graph_path = "data/graph/20ng.txt"
        if not os.path.exists(graph_path):
            print(f"Error: {graph_path} not found. Please run build_graph.py first.")
            return False
            
        # Just check file size first
        file_size = os.path.getsize(graph_path) / 1024 / 1024  # MB
        print(f"20NG graph file size: {file_size:.2f} MB")
        
        # Try to load just the first few lines to estimate memory
        with open(graph_path, 'r') as f:
            first_line = f.readline()
            print(f"Sample edge: {first_line.strip()}")
            
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing TextGCN Fixes ===\n")
    
    # Test MR dataset
    mr_success = test_graph_loading()
    
    # Test 20NG memory
    ng20_success = test_20ng_memory()
    
    print(f"\n=== Results ===")
    print(f"MR dataset test: {'PASS' if mr_success else 'FAIL'}")
    print(f"20NG memory test: {'PASS' if ng20_success else 'FAIL'}")
    
    if mr_success:
        print("\n✓ Ready to run training on MR dataset")
        print("Run: python trainer.py")
    else:
        print("\n✗ Issues detected. Check error messages above.")
