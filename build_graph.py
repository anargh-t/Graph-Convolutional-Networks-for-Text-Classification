"""
TextGCN Graph Construction Script

This script builds the heterogeneous graph for TextGCN by creating two types of edges:
1. Document-Word edges: Based on TF-IDF weights
2. Word-Word edges: Based on PMI (Pointwise Mutual Information) scores

The graph construction process:
1. Load cleaned text corpus
2. Compute TF-IDF features for document-word relationships
3. Compute PMI scores for word-word co-occurrence relationships
4. Build NetworkX graph with weighted edges
5. Save graph to file for training

Key concepts:
- TF-IDF: Term Frequency-Inverse Document Frequency for document-word weights
- PMI: Pointwise Mutual Information for word co-occurrence relationships
- Window-based co-occurrence: Words appearing within a sliding window are considered related
"""

import os
from collections import Counter

import networkx as nx

import itertools
import math
from collections import defaultdict
from time import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from utils import print_graph_detail


def get_window(content_lst, window_size):
    """
    Extract sliding windows from text documents for PMI calculation.
    
    This function processes each document and creates sliding windows of words
    to capture local co-occurrence patterns. Words appearing in the same window
    are considered to be related.
    
    Args:
        content_lst: List of documents (each document is a string or list of words)
        window_size: Size of the sliding window for co-occurrence detection
        
    Returns:
        tuple: (word_window_freq, word_pair_count, windows_len)
            - word_window_freq: Dictionary of word frequencies across all windows
            - word_pair_count: Dictionary of word pair co-occurrence counts
            - windows_len: Total number of windows processed
    """
    # Track word frequencies within windows
    word_window_freq = defaultdict(int)  # w(i): word frequency in windows
    # Track word pair co-occurrence counts
    word_pair_count = defaultdict(int)  # w(i,j): word pair co-occurrence count
    windows_len = 0
    
    # Process each document
    for words in tqdm(content_lst, desc="Split by window"):
        windows = list()

        # Convert string to word list if needed
        if isinstance(words, str):
            words = words.split()
        length = len(words)

        # Create windows based on document length
        if length <= window_size:
            # If document is shorter than window, use entire document as one window
            windows.append(words)
        else:
            # Create sliding windows of specified size
            for j in range(length - window_size + 1):
                window = words[j: j + window_size]
                # Remove duplicates within each window
                windows.append(list(set(window)))

        # Process each window
        for window in windows:
            # Count word frequencies in this window
            for word in window:
                word_window_freq[word] += 1

            # Count word pair co-occurrences in this window
            for word_pair in itertools.combinations(window, 2):
                word_pair_count[word_pair] += 1

        windows_len += len(windows)
        
    return word_window_freq, word_pair_count, windows_len


def cal_pmi(W_ij, W, word_freq_1, word_freq_2):
    """
    Calculate Pointwise Mutual Information (PMI) between two words.
    
    PMI measures the association between two words based on their co-occurrence
    frequency compared to their individual frequencies. Higher PMI indicates
    stronger association between words.
    
    Formula: PMI(i,j) = log(P(i,j) / (P(i) * P(j)))
    Where:
    - P(i,j) = W_ij / W (joint probability)
    - P(i) = word_freq_1 / W (marginal probability of word i)
    - P(j) = word_freq_2 / W (marginal probability of word j)
    
    Args:
        W_ij: Co-occurrence count of word pair (i,j)
        W: Total number of windows
        word_freq_1: Frequency of word i across all windows
        word_freq_2: Frequency of word j across all windows
        
    Returns:
        float: PMI score between the two words
    """
    # Calculate marginal probabilities
    p_i = word_freq_1 / W  # Probability of word i appearing in a window
    p_j = word_freq_2 / W  # Probability of word j appearing in a window
    p_i_j = W_ij / W       # Joint probability of words i and j appearing together
    
    # Calculate PMI: log(P(i,j) / (P(i) * P(j)))
    pmi = math.log(p_i_j / (p_i * p_j))

    return pmi


def count_pmi(windows_len, word_pair_count, word_window_freq, threshold):
    """
    Calculate PMI scores for all word pairs and filter by threshold.
    
    This function processes all word pairs that co-occurred in windows,
    calculates their PMI scores, and keeps only those above the threshold.
    This filtering helps remove weak associations and reduces graph sparsity.
    
    Args:
        windows_len: Total number of windows processed
        word_pair_count: Dictionary of word pair co-occurrence counts
        word_window_freq: Dictionary of individual word frequencies
        threshold: Minimum PMI threshold for keeping word pairs
        
    Returns:
        list: List of [word1, word2, pmi_score] tuples for word pairs above threshold
    """
    word_pmi_lst = list()
    
    # Process each word pair that co-occurred
    for word_pair, W_i_j in tqdm(word_pair_count.items(), desc="Calculate pmi between words"):
        # Get individual word frequencies
        word_freq_1 = word_window_freq[word_pair[0]]
        word_freq_2 = word_window_freq[word_pair[1]]

        # Calculate PMI for this word pair
        pmi = cal_pmi(W_i_j, windows_len, word_freq_1, word_freq_2)
        
        # Only keep word pairs with PMI above threshold
        if pmi <= threshold:
            continue
            
        # Add to result list: [word1, word2, pmi_score]
        word_pmi_lst.append([word_pair[0], word_pair[1], pmi])
        
    return word_pmi_lst


def get_pmi_edge(content_lst, window_size=20, threshold=0.):
    """
    Extract PMI-based word-word edges from text corpus.
    
    This function orchestrates the PMI calculation process:
    1. Load text corpus
    2. Extract sliding windows
    3. Calculate PMI scores for word pairs
    4. Return filtered word-word edges
    
    Args:
        content_lst: List of documents or path to text file
        window_size: Size of sliding window for co-occurrence detection
        threshold: Minimum PMI threshold for keeping edges
        
    Returns:
        tuple: (pmi_edge_lst, pmi_time)
            - pmi_edge_lst: List of [word1, word2, pmi_score] edges
            - pmi_time: Time taken for PMI calculation
    """
    # Handle file path input
    if isinstance(content_lst, str):
        content_lst = list(open(content_lst, "r"))
    print("pmi read file len:", len(content_lst))

    # Start timing PMI calculation
    pmi_start = time()
    
    # Extract sliding windows and count co-occurrences
    word_window_freq, word_pair_count, windows_len = get_window(content_lst,
                                                                window_size=window_size)

    # Calculate PMI scores and filter by threshold
    pmi_edge_lst = count_pmi(windows_len, word_pair_count, word_window_freq, threshold)
    print("Total number of edges between word:", len(pmi_edge_lst))
    
    # Calculate total time
    pmi_time = time() - pmi_start
    return pmi_edge_lst, pmi_time


class BuildGraph:
    """
    Main class for building TextGCN heterogeneous graph.
    
    This class orchestrates the complete graph construction process:
    1. Creates document-word edges using TF-IDF weights
    2. Creates word-word edges using PMI scores
    3. Combines both types of edges into a single graph
    4. Saves the graph to file for training
    
    The resulting graph contains:
    - Document nodes (indices 0 to num_docs-1)
    - Word nodes (indices num_docs to num_docs+num_words-1)
    - Document-word edges with TF-IDF weights
    - Word-word edges with PMI weights
    """
    
    def __init__(self, dataset):
        """
        Initialize graph construction for a specific dataset.
        
        Args:
            dataset (str): Name of the dataset (e.g., 'mr', '20ng', 'R8')
        """
        # Set up file paths
        clean_corpus_path = "data/text_dataset/clean_corpus"
        self.graph_path = "data/graph"
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.graph_path):
            os.makedirs(self.graph_path)

        # Initialize word-to-ID mapping dictionary
        self.word2id = dict()  # Maps words to their node indices
        self.dataset = dataset
        print(f"\n==> Current dataset: {dataset} <==")

        # Initialize empty NetworkX graph
        self.g = nx.Graph()

        # Set path to cleaned corpus file
        self.content = f"{clean_corpus_path}/{dataset}.txt"

        # Build the graph by adding both types of edges
        self.get_tfidf_edge()  # Add document-word edges
        self.get_pmi_edge()    # Add word-word edges
        self.save()            # Save graph to file

    def get_pmi_edge(self):
        pmi_edge_lst, self.pmi_time = get_pmi_edge(self.content, window_size=20, threshold=0.0)
        print("pmi time:", self.pmi_time)

        for edge_item in pmi_edge_lst:
            word_indx1 = self.node_num + self.word2id[edge_item[0]]
            word_indx2 = self.node_num + self.word2id[edge_item[1]]
            if word_indx1 == word_indx2:
                continue
            self.g.add_edge(word_indx1, word_indx2, weight=edge_item[2])

        print_graph_detail(self.g)

    def get_tfidf_edge(self):
        # 获得tfidf权重矩阵（sparse）和单词列表
        tfidf_vec = self.get_tfidf_vec()

        count_lst = list()  # 统计每个句子的长度
        for ind, row in tqdm(enumerate(tfidf_vec),
                             desc="generate tfidf edge"):
            count = 0
            for col_ind, value in zip(row.indices, row.data):
                word_ind = self.node_num + col_ind
                self.g.add_edge(ind, word_ind, weight=value)
                count += 1
            count_lst.append(count)

        print_graph_detail(self.g)

    def get_tfidf_vec(self):
        """
        学习获得tfidf矩阵，及其对应的单词序列
        :param content_lst:
        :return:
        """
        start = time()
        text_tfidf = Pipeline([
            ("vect", CountVectorizer(min_df=1,
                                     max_df=1.0,
                                     token_pattern=r"\S+",
                                     )),
            ("tfidf", TfidfTransformer(norm=None,
                                       use_idf=True,
                                       smooth_idf=False,
                                       sublinear_tf=False
                                       ))
        ])

        tfidf_vec = text_tfidf.fit_transform(open(self.content, "r"))

        self.tfidf_time = time() - start
        print("tfidf time:", self.tfidf_time)
        print("tfidf_vec shape:", tfidf_vec.shape)
        print("tfidf_vec type:", type(tfidf_vec))

        self.node_num = tfidf_vec.shape[0]

        # Map words
        try:
            vocab_lst = text_tfidf["vect"].get_feature_names_out()
        except AttributeError:
            vocab_lst = text_tfidf["vect"].get_feature_names()
        print("vocab_lst len:", len(vocab_lst))
        for ind, word in enumerate(vocab_lst):
            self.word2id[word] = ind

        self.vocab_lst = vocab_lst

        return tfidf_vec

    def save(self):
        print("total time:", self.pmi_time + self.tfidf_time)
        nx.write_weighted_edgelist(self.g,
                                   f"{self.graph_path}/{self.dataset}.txt")

        print("\n")


def main():
    BuildGraph("mr")
    BuildGraph("ohsumed")
    BuildGraph("R52")
    BuildGraph("R8")
    BuildGraph("20ng")


if __name__ == '__main__':
    main()
