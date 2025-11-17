"""
Topic Modeling Module for Document-Topic-Topic Graph Construction

This module implements Latent Dirichlet Allocation (LDA) for topic modeling
and generates topic embeddings based on word2vec representations of top topic words.

Key components:
- LDA topic modeling using sklearn
- Topic distribution extraction (document-topic and topic-word)
- Topic embedding generation using word2vec
- Integration with graph construction pipeline
"""

import os
import numpy as np
from time import time
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

try:
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False
    print("Warning: Gensim not available. Topic embeddings will use simple averaging.")


class TopicModel:
    """
    Topic modeling class using Latent Dirichlet Allocation (LDA).
    
    This class provides:
    - Document-topic distributions (θ_dk)
    - Topic-word distributions
    - Topic embeddings based on word2vec
    """
    
    def __init__(self, 
                 num_topics: int = 50,
                 random_state: int = 42,
                 max_iter: int = 20,
                 learning_method: str = 'batch',
                 n_components: Optional[int] = None):
        """
        Initialize the topic model.
        
        Args:
            num_topics: Number of topics to extract (K)
            random_state: Random seed for reproducibility
            max_iter: Maximum number of iterations for LDA
            learning_method: 'batch' or 'online' learning
            n_components: Alias for num_topics (for compatibility)
        """
        if n_components is not None:
            num_topics = n_components
        
        self.num_topics = num_topics
        self.random_state = random_state
        self.max_iter = max_iter
        self.learning_method = learning_method
        
        # Will be set during fitting
        self.lda_model = None
        self.vectorizer = None
        self.vocabulary_ = None
        self.topic_word_distribution = None
        self.topic_embeddings = None
        self.word2vec_model = None
        
    def fit(self, documents: List[str], min_df: int = 2, max_df: float = 0.95):
        """
        Fit the LDA model on a collection of documents.
        
        Args:
            documents: List of documents (each as a string of space-separated words)
            min_df: Minimum document frequency for vocabulary
            max_df: Maximum document frequency (as fraction) for vocabulary
        """
        print(f"\n==> Fitting LDA model with {self.num_topics} topics <==")
        start_time = time()
        
        # Convert documents to list of word lists if needed
        if isinstance(documents[0], str):
            doc_words = [doc.split() for doc in documents]
        else:
            doc_words = documents
        
        # Create vocabulary and document-term matrix
        self.vectorizer = CountVectorizer(
            min_df=min_df,
            max_df=max_df,
            token_pattern=r'\S+',  # Match any non-whitespace sequence
            lowercase=False
        )
        
        # Convert documents back to strings for vectorizer
        doc_strings = [' '.join(doc) for doc in doc_words]
        doc_term_matrix = self.vectorizer.fit_transform(doc_strings)
        
        self.vocabulary_ = self.vectorizer.get_feature_names_out()
        print(f"Vocabulary size: {len(self.vocabulary_)}")
        print(f"Document-term matrix shape: {doc_term_matrix.shape}")
        
        # Fit LDA model
        self.lda_model = LatentDirichletAllocation(
            n_components=self.num_topics,
            random_state=self.random_state,
            max_iter=self.max_iter,
            learning_method=self.learning_method,
            verbose=1
        )
        
        self.lda_model.fit(doc_term_matrix)
        
        # Store training documents for later use
        self.training_documents = documents
        
        # Extract topic-word distributions
        self.topic_word_distribution = self.lda_model.components_  # Shape: (num_topics, vocab_size)
        # Normalize to probabilities
        self.topic_word_distribution = self.topic_word_distribution / \
            self.topic_word_distribution.sum(axis=1, keepdims=True)
        
        fit_time = time() - start_time
        print(f"LDA fitting completed in {fit_time:.2f} seconds")
        
        return self
    
    def get_document_topic_distribution(self, documents: Optional[List[str]] = None) -> np.ndarray:
        """
        Get document-topic distribution matrix θ_dk.
        
        Args:
            documents: Optional list of documents. If None, uses training documents.
        
        Returns:
            numpy array of shape (num_documents, num_topics) where each row
            is the topic distribution for a document (θ_d)
        """
        if self.lda_model is None:
            raise ValueError("Model must be fitted first. Call fit() method.")
        
        if documents is None:
            # Use training documents
            if not hasattr(self, 'training_documents'):
                raise ValueError("No documents provided and no training documents stored.")
            documents = self.training_documents
        
        # Convert to document-term matrix
        if isinstance(documents[0], str):
            doc_strings = documents
        else:
            doc_strings = [' '.join(doc) for doc in documents]
        
        doc_term_matrix = self.vectorizer.transform(doc_strings)
        
        # Get document-topic distributions
        doc_topic_dist = self.lda_model.transform(doc_term_matrix)  # Shape: (num_docs, num_topics)
        
        return doc_topic_dist
    
    def get_topic_word_distribution(self, top_n: int = 20) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get topic-word distributions with top words for each topic.
        
        Args:
            top_n: Number of top words to return per topic
        
        Returns:
            Dictionary mapping topic_id -> list of (word, probability) tuples
        """
        if self.topic_word_distribution is None:
            raise ValueError("Model must be fitted first. Call fit() method.")
        
        topic_words = {}
        vocab_list = list(self.vocabulary_)
        
        for topic_id in range(self.num_topics):
            # Get word probabilities for this topic
            word_probs = self.topic_word_distribution[topic_id]
            
            # Get top N words
            top_indices = np.argsort(word_probs)[-top_n:][::-1]
            top_words = [(vocab_list[idx], word_probs[idx]) for idx in top_indices]
            
            topic_words[topic_id] = top_words
        
        return topic_words
    
    def fit_word2vec(self, documents: List[str], 
                     vector_size: int = 100,
                     window: int = 5,
                     min_count: int = 2,
                     workers: int = 4):
        """
        Train a Word2Vec model on the documents for topic embeddings.
        
        Args:
            documents: List of documents (as strings or word lists)
            vector_size: Dimension of word embeddings
            window: Context window size
            min_count: Minimum word frequency
            workers: Number of worker threads
        """
        if not GENSIM_AVAILABLE:
            print("Warning: Gensim not available. Using simple averaging for embeddings.")
            return
        
        print(f"\n==> Training Word2Vec model for topic embeddings <==")
        start_time = time()
        
        # Convert to list of word lists
        if isinstance(documents[0], str):
            sentences = [doc.split() for doc in documents]
        else:
            sentences = documents
        
        # Train Word2Vec model
        self.word2vec_model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=0,  # CBOW
            epochs=10
        )
        
        train_time = time() - start_time
        print(f"Word2Vec training completed in {train_time:.2f} seconds")
        print(f"Vocabulary size in Word2Vec: {len(self.word2vec_model.wv)}")
    
    def get_topic_embeddings(self, top_n: int = 20) -> np.ndarray:
        """
        Generate topic embeddings as average of top-N word embeddings.
        
        Args:
            top_n: Number of top words to use for embedding
        
        Returns:
            numpy array of shape (num_topics, embedding_dim) containing
            topic embeddings
        """
        if self.topic_word_distribution is None:
            raise ValueError("Model must be fitted first. Call fit() method.")
        
        topic_words = self.get_topic_word_distribution(top_n=top_n)
        vocab_list = list(self.vocabulary_)
        
        embeddings = []
        
        for topic_id in range(self.num_topics):
            top_words = topic_words[topic_id]
            
            if GENSIM_AVAILABLE and self.word2vec_model is not None:
                # Use Word2Vec embeddings
                word_vectors = []
                for word, prob in top_words:
                    if word in self.word2vec_model.wv:
                        # Weight by probability
                        word_vectors.append(self.word2vec_model.wv[word] * prob)
                
                if len(word_vectors) > 0:
                    # Average weighted embeddings
                    topic_emb = np.mean(word_vectors, axis=0)
                else:
                    # Fallback: random embedding
                    topic_emb = np.random.randn(100)
            else:
                # Fallback: use topic-word distribution as embedding
                # This is a simple approach when Word2Vec is not available
                topic_emb = self.topic_word_distribution[topic_id]
            
            embeddings.append(topic_emb)
        
        self.topic_embeddings = np.array(embeddings)
        
        print(f"Topic embeddings shape: {self.topic_embeddings.shape}")
        return self.topic_embeddings
    
    def save(self, filepath: str):
        """Save the topic model to disk."""
        import pickle
        
        model_data = {
            'lda_model': self.lda_model,
            'vectorizer': self.vectorizer,
            'vocabulary_': self.vocabulary_,
            'topic_word_distribution': self.topic_word_distribution,
            'topic_embeddings': self.topic_embeddings,
            'num_topics': self.num_topics,
            'word2vec_model': self.word2vec_model
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Topic model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load the topic model from disk."""
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.lda_model = model_data['lda_model']
        self.vectorizer = model_data['vectorizer']
        self.vocabulary_ = model_data['vocabulary_']
        self.topic_word_distribution = model_data['topic_word_distribution']
        self.topic_embeddings = model_data.get('topic_embeddings')
        self.num_topics = model_data['num_topics']
        self.word2vec_model = model_data.get('word2vec_model')
        
        print(f"Topic model loaded from {filepath}")


def load_documents_from_file(filepath: str) -> List[str]:
    """
    Load documents from a text file (one document per line).
    
    Args:
        filepath: Path to the text file
    
    Returns:
        List of document strings
    """
    documents = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                documents.append(line)
    
    return documents

